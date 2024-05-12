import torch
import triton
import triton.language as tl

##
## constants
##

BLOCK_SIZE_N = 16
BLOCK_SIZE_M = 16
BLOCK_SIZE_K = 16

##
## functions
##

ceil_div = lambda x, y: (x + y - 1) // y

##
## quantize
##

# requires K divisible by BLOCK_SIZE_K
# requires BLOCK_SIZE_K divisble by QFACT
# assumes contiguous data layout
@triton.jit
def quantize_kernel(
    X, Y, N, K, K1,
    scale, zero_point,
    BITS: tl.constexpr,
    QFACT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
):
    # data type
    dtype = X.dtype.element_ty

    # quantization params
    QMASK = (1 << BITS) - 1
    QMASK_FLT = tl.full((), QMASK, dtype=dtype)

    # load block data
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # deswizzle k block indices
    bk  = tl.arange(0, BLOCK_SIZE_K )
    bk1 = tl.arange(0, BLOCK_SIZE_K1)
    x_shift = BITS * (bk % QFACT)

    # get row indices for each axis
    rn  = pid_n * BLOCK_SIZE_N  + tl.arange(0, BLOCK_SIZE_N)
    rk  = pid_k * BLOCK_SIZE_K  + bk
    rk1 = pid_k * BLOCK_SIZE_K1 + bk1

    # load quantized data
    mask_x = rn[:, None] < N
    mask_y = rn[:, None] < N

    # get first data pointer
    X1 = X + (rn[:, None] * K  + rk [None, :])
    Y1 = Y + (rn[:, None] * K1 + rk1[None, :])

    # load data
    x = tl.load(X1, mask=mask_x)

    # quantize data
    xf = tl.clamp(x / scale + zero_point, 0.0, QMASK_FLT)
    xi = (xf + 0.5).to(tl.uint8) # round to nearest
    xq = xi << x_shift

    # compress quantized data
    mat = tl.reshape(xq, (BLOCK_SIZE_N, BLOCK_SIZE_K1, QFACT))
    out = tl.sum(mat, axis=2)

    # store quantized data
    tl.store(Y1, out, mask=mask_y)

##
## dequantize
##

# requires K divisible by BLOCK_SIZE_K
# requires BLOCK_SIZE_K divisble by QFACT
# assumes contiguous data layout
@triton.jit
def dequantize_kernel(
    X, Y, N, K, K1,
    scale, zero_point,
    BITS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
):
    # quantization params
    QFACT = 8 // BITS
    QMASK = (1 << BITS) - 1
    QMASK_INT = tl.full((), QMASK, dtype=tl.uint8)

    # output params
    dtype = Y.dtype.element_ty
    scale_ty = tl.full((), scale, dtype=dtype)
    zero_point_ty = tl.full((), zero_point, dtype=dtype)

    # load block data
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # get k block indices
    bk = tl.arange(0, BLOCK_SIZE_K)
    bk1, bq1 = bk // QFACT, bk % QFACT
    x_shift = BITS * bq1

    # get row indices for each axis
    rn  = pid_n * BLOCK_SIZE_N  + tl.arange(0, BLOCK_SIZE_N)
    rk  = pid_k * BLOCK_SIZE_K  + bk
    rk1 = pid_k * BLOCK_SIZE_K1 + bk1

    # load quantized data
    mask_x = rn[:, None] < N
    mask_y = rn[:, None] < N

    # get data pointers
    X1 = X + (rn[:, None] * K1 + rk1[None, :])
    Y1 = Y + (rn[:, None] * K  + rk [None, :])

    # load data
    x = tl.load(X1, mask=mask_x)

    # unpack quantized data
    xi = (x >> x_shift) & QMASK_INT
    xf = scale_ty * (xi.to(dtype) - zero_point_ty)

    # store dequantized data
    tl.store(Y1, xf, mask=mask_y)

##
## matmul
##

# requires K divisible by BLOCK_SIZE_K
@triton.jit
def matmul_kernel(
    A, B, C, N, M, K,
    stride_an, stride_ak,
    stride_bk, stride_bm,
    stride_cn, stride_cm,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # data type
    dtype = C.dtype.element_ty

    # load program ids
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # get row indices for each axis
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # create read/write masks
    mask_a = rn[:, None] < N
    mask_b = rm[None, :] < M
    mask_c = (rn[:, None] < N) & (rm[None, :] < M)

    # the memory addresses of first block elemenets
    A1 = A + (rn[:, None] * stride_an + rk[None, :] * stride_ak)
    B1 = B + (rk[:, None] * stride_bk + rm[None, :] * stride_bm)
    C1 = C + (rn[:, None] * stride_cn + rm[None, :] * stride_cm)

    # allocate accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=dtype)

    # initialize and iteratively update accumulator
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A1, mask=mask_a)
        b = tl.load(B1, mask=mask_b)

        # block level matrix multiplication
        acc += tl.dot(a, b, out_dtype=dtype)
    
        # increment pointers of A and B
        A1 += BLOCK_SIZE_K * stride_ak
        B1 += BLOCK_SIZE_K * stride_bk

    # write back result
    tl.store(C1, acc, mask=mask_c)

# requires K divisible by BLOCK_SIZE_K
# requires BLOCK_SIZE_K divisble by QFACT
@triton.jit
def matmul_quant_kernel(
    A, B, C, N, M, K, K1,
    stride_an, stride_ak,
    stride_bk, stride_bm,
    stride_cn, stride_cm,
    scale, zero_point,
    BITS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
):
    # quantization parameters
    QFACT = 8 // BITS
    QMASK = (1 << BITS) - 1
    QMASK_INT = tl.full((), QMASK, dtype=tl.uint8)

    # data type
    dtype = C.dtype.element_ty
    zero_point_ty = tl.full((), zero_point, dtype=dtype)
    scale_ty = tl.full((), scale, dtype=dtype)

    # load program ids
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # get indices for each axis
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # deswizzle k between index and shifter
    rk = tl.arange(0, BLOCK_SIZE_K)
    rk1, rq1 = rk // QFACT, rk % QFACT
    a_shift = BITS * rq1

    # create read/write masks
    mask_a = rn[:, None] < N
    mask_b = rm[None, :] < M
    mask_c = (rn[:, None] < N) & (rm[None, :] < M)

    # the memory addresses of first block elemenets
    A1 = A + (rn[:, None] * stride_an + rk1[None, :] * stride_ak)
    B1 = B + (rk[:, None] * stride_bk + rm [None, :] * stride_bm)
    C1 = C + (rn[:, None] * stride_cn + rm [None, :] * stride_cm)

    # allocate accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=dtype)

    # iteratively update accumulator
    for k in range(0, K, BLOCK_SIZE_K):
        aq = tl.load(A1, mask=mask_a)
        b = tl.load(B1, mask=mask_b)

        # unpack a values
        ai = (aq >> a_shift) & QMASK_INT
        a = ai.to(dtype) - zero_point_ty

        # do the actual matmul
        acc += tl.dot(a, b, out_dtype=dtype)
    
        # increment A by BLOCK_SIZE_K
        A1 += BLOCK_SIZE_K1 * stride_ak
        B1 += BLOCK_SIZE_K  * stride_bk

    # scale output
    acc *= scale_ty

    # write back result
    tl.store(C1, acc, mask=mask_c)

##
## interfaces
##

def quantize(x, bits, scale, zero_point):
    # tensor information
    device = x.device
    N, K = x.shape

    # shape params
    QFACT = 8 // bits
    K1 = K // QFACT
    BLOCK_SIZE_K1 = BLOCK_SIZE_K // QFACT

    assert(x.is_contiguous())
    assert(K % BLOCK_SIZE_K == 0)
    assert(BLOCK_SIZE_K % QFACT == 0)

    # output shape
    y = torch.empty((N, K1), device=device, dtype=torch.uint8)

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(K, BLOCK_SIZE_K)
    quantize_kernel[grid](
        x, y, N, K, K1,
        scale, zero_point,
        BITS=bits, QFACT=QFACT,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_K1=BLOCK_SIZE_K1,
    )

    # return result
    return y

def dequantize(x, dtype, bits, scale, zero_point):
    assert(x.is_contiguous())

    # tensor information
    device = x.device
    N, K1 = x.shape

    # shape params
    QFACT = 8 // bits
    K = QFACT * K1
    BLOCK_SIZE_K1 = BLOCK_SIZE_K // QFACT

    assert(x.is_contiguous())
    assert(K % BLOCK_SIZE_K == 0)
    assert(BLOCK_SIZE_K % QFACT == 0)

    # output shape
    y = torch.empty((N, K), device=device, dtype=dtype)

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(K, BLOCK_SIZE_K)
    dequantize_kernel[grid](
        x, y, N, K, K1,
        scale, zero_point, BITS=bits,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_K1=BLOCK_SIZE_K1,
    )

    # return result
    return y

def matmul(a, b, dtype=None, bits=None, scale=None, zero_point=None):
    # device information
    assert(a.device == b.device)
    device = a.device

    # detect dtype
    if dtype is None:
        dtype = b.dtype

    # shape information
    N, Ka = a.size()
    Kb, M = b.size()

    # dtype information
    if bits is not None:
        QFACT = 8 // bits
        BLOCK_SIZE_K1 = BLOCK_SIZE_K // QFACT
        assert(Kb == Ka * QFACT)
        assert(bits in [1, 2, 4, 8])
        assert(a.dtype == torch.uint8)
        assert(scale is not None)
        assert(zero_point is not None)
        K1, K = Ka, Kb
    else:
        assert(Ka == Kb)
        K = Ka

    # output allocate
    c = torch.zeros((N, M), device=device, dtype=dtype)

    # stride information
    san, sak = a.stride()
    sbk, sbm = b.stride()
    scn, scm = c.stride()

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(M, BLOCK_SIZE_M)
    if bits is None:
        matmul_kernel[grid](
            a, b, c, N, M, K,
            san, sak, sbk, sbm, scn, scm,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    else:
        matmul_quant_kernel[grid](
            a, b, c, N, M, K, K1,
            san, sak, sbk, sbm, scn, scm,
            scale=scale, zero_point=zero_point, BITS=bits,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_K1=BLOCK_SIZE_K1,
        )

    # return result
    return c
