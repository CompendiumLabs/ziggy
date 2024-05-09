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
    # quantization params
    QFACT = 8 // BITS
    QMASK = (1 << BITS) - 1
    QMASK_TY = tl.full((), QMASK, dtype=tl.uint8)

    # load block data
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # get row indices for each axis
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = pid_k * BLOCK_SIZE_K + QFACT * tl.arange(0, BLOCK_SIZE_K1)

    # get first data pointer
    mask = (rn[:, None] < N) & (rk[None, :] < K)
    X1 = X + (rn[:, None] * K + rk[None, :])

    # create output tensor
    out = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K1), dtype=tl.uint8)

    # quantize data
    for q in range(0, 8, BITS):
        q_ty = tl.full((), q, dtype=tl.uint8)
        x = tl.load(X1)
        xf = tl.clamp(x / scale + zero_point, 0.0, 255.0)
        xi = (xf + 0.5).to(tl.uint8) # round to nearest
        out |= (xi & QMASK_TY) << q_ty
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
    QMASK_TY = tl.full((), QMASK, dtype=tl.uint8)

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
        xi = (x >> q_ty) & QMASK_TY
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

@triton.jit
def matmul_quant_kernel(
    A, B, C, N, M, K,
    stride_an, stride_ak,
    stride_bk, stride_bm,
    stride_cn, stride_cm,
    scale, zero_point,
    BITS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # quantization parameters
    QMASK = (1 << BITS) - 1

    # data type
    dtype = C.dtype.element_ty
    zero_point_ty = tl.full((), zero_point, dtype=dtype)

    # load program ids
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # get row indices for each axis (with wrap around)
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rk = tl.arange(0, BLOCK_SIZE_K)

    # the memory addresses of first block elemenets
    A1 = A + (rn[:, None] * stride_an + rk[None, :] * stride_ak)
    B1 = B + (rk[:, None] * stride_bk + rm[None, :] * stride_bm)

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=dtype)
    for k in range(0, K, BLOCK_SIZE_K):
        aq = tl.load(A1)
        bq = tl.load(B1)

        for q in range(0, 8, BITS):
            # unpack
            ai = (aq >> q) & QMASK
            bi = (bq >> q) & QMASK

            # offset
            a = ai.to(dtype) - zero_point_ty
            b = bi.to(dtype) - zero_point_ty

            # do the actual matmul
            acc += tl.dot(a, b, out_dtype=dtype)
    
        # increment pointers of A and B
        A1 += BLOCK_SIZE_K * stride_ak
        B1 += BLOCK_SIZE_K * stride_bk

    # scale output
    acc *= scale

    # write back result
    C = C + (rn[:, None] * stride_cn + rm[None, :] * stride_cm)
    mask = (rn[:, None] < N) & (rm[None, :] < M)
    tl.store(C, acc, mask=mask)

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

    # output shape
    y = torch.empty((N, K1), device=device, dtype=torch.uint8)

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(K, BLOCK_SIZE_K)
    quantize_kernel[grid](
        x, y, N, K, K1, scale, zero_point, BITS=bits,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_K1=BLOCK_SIZE_K1
    )

    # return result
    return y

def dequantize(x, bits, scale, zero_point, dtype=torch.float16):
    # tensor information
    device = x.device
    N, K1 = x.shape

    # shape params
    QFACT = 8 // bits
    K = QFACT * K1
    BLOCK_SIZE_K1 = BLOCK_SIZE_K // QFACT

    # output shape
    y = torch.empty((N, K), device=device, dtype=dtype)

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(K, BLOCK_SIZE_K)
    dequantize_kernel[grid](
        x, y, N, K, K1, scale, zero_point, BITS=bits,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_K1=BLOCK_SIZE_K1
    )

    # return result
    return y

def matmul(a, b, bits=None, scale=None, zero_point=None, dtype=torch.float16):
    # device information
    assert(a.device == b.device)
    device = a.device

    # dtype information
    if bits is not None:
        assert(bits in [1, 2, 4, 8])
        assert(a.dtype == torch.uint8)
        assert(b.dtype == torch.uint8)
        assert(scale is not None)
        assert(zero_point is not None)

    # shape information
    N, K1 = a.size()
    K2, M = b.size()
    assert(K1 == K2)
    K = K1

    # stride information
    san, sak = a.stride()
    sbk, sbm = b.stride()

    # output shape
    c = torch.zeros((N, M), device=device, dtype=dtype)
    scn, scm = c.stride()

    # call kernel
    grid = ceil_div(N, BLOCK_SIZE_N), ceil_div(M, BLOCK_SIZE_M)
    if bits is None:
        matmul_kernel[grid](
            a, b, c, N, M, K, san, sak, sbk, sbm, scn, scm,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K
        )
    else:
        matmul_quant_kernel[grid](
            a, b, c, N, M, K, san, sak, sbk, sbm, scn, scm,
            scale=scale, zero_point=zero_point, BITS=bits,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K
        )

    # return result
    return c
