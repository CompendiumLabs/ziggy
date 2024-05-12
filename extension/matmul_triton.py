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
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
):
    # data type
    dtype = X.dtype.element_ty

    # quantization params
    QFACT = 8 // BITS
    QMASK = (1 << BITS) - 1
    QMASK_FLT = tl.full((), QMASK, dtype=dtype)

    # load block data
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # get row indices for each axis
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = pid_k * BLOCK_SIZE_K + QFACT * tl.arange(0, BLOCK_SIZE_K1)

    # get first data pointer
    X1 = X + (rn[:, None] * K + rk[None, :])

    # create output tensor
    out = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K1), dtype=tl.uint8)

    # quantize data
    for q in range(0, 8, BITS):
        q_ty = tl.full((), q, dtype=tl.uint8)
        x = tl.load(X1)
        xf = tl.clamp(x / scale + zero_point, 0.0, QMASK_FLT)
        xi = (xf + 0.5).to(tl.uint8) # round to nearest
        out |= xi << q_ty
        X1 += 1

    # store quantized data
    rk1 = pid_k * BLOCK_SIZE_K1 + tl.arange(0, BLOCK_SIZE_K1)
    mask1 = (rn[:, None] < N) & (rk1[None, :] < K1)
    Y = Y + (rn[:, None] * K1 + rk1[None, :])
    tl.store(Y, out, mask=mask1)

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

    # get row indices for each axis
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk1 = pid_k * BLOCK_SIZE_K1 + tl.arange(0, BLOCK_SIZE_K1)
    rk = pid_k * BLOCK_SIZE_K + QFACT * tl.arange(0, BLOCK_SIZE_K1)

    # load quantized data
    mask1 = (rn[:, None] < N) & (rk1[None, :] < K1)
    X1 = X + (rn[:, None] * K1 + rk1[None, :])
    x = tl.load(X1, mask=mask1)

    # create output tensor
    out = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K1), dtype=dtype)

    # quantize data
    for q in range(0, 8, BITS):
        # unpack quantized data
        q_ty = tl.full((), q, dtype=tl.uint8)
        xi = (x >> q_ty) & QMASK_INT
        xf = xi.to(dtype)
        out = scale_ty * (xf - zero_point_ty)

        # store dequantized data
        mask = (rn[:, None] < N) & (rk[None, :] < K)
        Y1 = Y + (rn[:, None] * K + rk[None, :])
        tl.store(Y1, out, mask=mask)
        rk += 1

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
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rk = tl.arange(0, BLOCK_SIZE_K)

    # the memory addresses of first block elemenets
    A1 = A + (rn[:, None] * stride_an + rk[None, :] * stride_ak)
    B1 = B + (rk[:, None] * stride_bk + rm[None, :] * stride_bm)

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=dtype)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A1)
        b = tl.load(B1)

        # block level matrix multiplication
        acc += tl.dot(a, b, out_dtype=dtype)
    
        # increment pointers of A and B
        A1 += BLOCK_SIZE_K * stride_ak
        B1 += BLOCK_SIZE_K * stride_bk

    # write back result
    C = C + (rn[:, None] * stride_cn + rm[None, :] * stride_cm)
    mask = (rn[:, None] < N) & (rm[None, :] < M)
    tl.store(C, acc, mask=mask)

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
    rn0 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm0 = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn, rm = rn0 % N, rm0 % M

    # deswizzle k between index and shifter
    rk = tl.arange(0, BLOCK_SIZE_K)
    rk1 = rk // QFACT
    rq1 = BITS * (rk % QFACT)

    # the memory addresses of first block elemenets
    A1 = A + (rn[:, None] * stride_an + rk1[None, :] * stride_ak)
    B1 = B + (rk[:, None] * stride_bk + rm [None, :] * stride_bm)

    # allocate accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=dtype)

    # iteratively update accumulator
    for k in range(0, K, BLOCK_SIZE_K):
        aq = tl.load(A1)
        b = tl.load(B1)

        # unpack a values
        ai = (aq >> rq1) & QMASK_INT
        a = ai.to(dtype) - zero_point_ty

        # do the actual matmul
        acc += tl.dot(a, b, out_dtype=dtype)
    
        # increment A by BLOCK_SIZE_K
        A1 += BLOCK_SIZE_K1 * stride_ak
        B1 += BLOCK_SIZE_K  * stride_bk

    # scale output
    acc *= scale_ty

    # write back result
    C1 = C + (rn[:, None] * stride_cn + rm[None, :] * stride_cm)
    mask = (rn0[:, None] < N) & (rm0[None, :] < M)
    tl.store(C1, acc, mask=mask)

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
        scale, zero_point, BITS=bits,
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
