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

def ceil_div(x, y):
    return (x + y - 1) // y

@triton.jit
def clamp(x, a, b):
    return tl.maximum(a, tl.minimum(b, x))

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
    scale_ty = tl.full((), scale, dtype=dtype)
    zero_point_ty = tl.full((), zero_point, dtype=dtype)

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
    xf = clamp(x / scale_ty + zero_point_ty, 0.0, QMASK_FLT)
    xi = tl.math.rint(xf).to(tl.uint8) # round-to-nearest-even
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
    # output params
    dtype = Y.dtype.element_ty
    scale_ty = tl.full((), scale, dtype=dtype)
    zero_point_ty = tl.full((), zero_point, dtype=dtype)

    # quantization params
    QFACT = 8 // BITS
    QMASK = (1 << BITS) - 1
    QMASK_INT = tl.full((), QMASK, dtype=tl.uint8)

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
def matmul_float_kernel(
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
    mask_c = mask_a & mask_b

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
    # data type
    dtype = C.dtype.element_ty
    zero_point_ty = tl.full((), zero_point, dtype=dtype)
    scale_ty = tl.full((), scale, dtype=dtype)

    # quantization parameters
    QFACT = 8 // BITS
    QMASK = (1 << BITS) - 1
    QMASK_INT = tl.full((), QMASK, dtype=tl.uint8)

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
    mask_c = mask_a & mask_b

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
        b1 = b.to(dtype)
        acc += tl.dot(a, b1, out_dtype=dtype)
    
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

def dequantize(x, bits, scale, zero_point, dtype):
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

def matmul_float(x, y):
    # shape information
    N, Kx = x.size()
    Ky, M = y.size()
    device = x.device
    K = Kx

    assert(x.device == y.device)
    assert(Kx == Ky)

    # output allocate
    z = torch.zeros((N, M), device=device, dtype=torch.float32)

    # stride information
    sxn, sxk = x.stride()
    syk, sym = y.stride()
    szn, szm = z.stride()

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(M, BLOCK_SIZE_M)
    matmul_float_kernel[grid](
        x, y, z, N, M, K,
        sxn, sxk, syk, sym, szn, szm,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # return result
    return z

def matmul_quant(x, y, bits, scale, zero_point):
    # shape information
    N, Kx = x.size()
    Ky, M = y.size()
    QFACT = 8 // bits
    BLOCK_SIZE_K1 = BLOCK_SIZE_K // QFACT
    device = x.device
    K1, K = Kx, Ky

    assert(x.device == y.device)
    assert(x.is_contiguous())
    assert(Ky == Kx * QFACT)
    assert(8 % bits == 0)
    assert(x.dtype == torch.uint8)
    assert(scale is not None)
    assert(zero_point is not None)

    # output allocate
    z = torch.zeros((N, M), device=device, dtype=torch.float32)

    # stride information
    sxn, sxk = x.stride()
    syk, sym = y.stride()
    szn, szm = z.stride()

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(M, BLOCK_SIZE_M)
    matmul_quant_kernel[grid](
        x, y, z, N, M, K, K1,
        sxn, sxk, syk, sym, szn, szm,
        scale=scale, zero_point=zero_point,
        BITS=bits,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_K1=BLOCK_SIZE_K1,
    )

    # return result
    return z
