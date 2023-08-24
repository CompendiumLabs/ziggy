import torch
import triton
import triton.language as tl

##
## constants
##

BLOCK_SIZE_N = 32
BLOCK_SIZE_M = 32
BLOCK_SIZE_K = 128

##
## functions
##

ceil_div = lambda x, y: (x + y - 1) // y

##
## kernels
##

@triton.jit
def matmul_kernel(
    A, B, C, N, M, K,
    stride_an, stride_ak,
    stride_bk, stride_bm,
    stride_cn, stride_cm,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # load program ids
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # get row indices for each axis
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # indices for a single block
    rk = tl.arange(0, BLOCK_SIZE_K)

    # the memory addresses of first block elemenets
    A = A + (rn[:, None] * stride_an + rk[None, :] * stride_ak)
    B = B + (rk [:, None] * stride_bk  + rm[None, :] * stride_bm)

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A)
        b = tl.load(B)

        # block level matrix multiplication
        acc += tl.dot(a, b)

        # increment pointers of A and B
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk

    # write back result
    C = C + (rn[:, None] * stride_cn + rm[None, :] * stride_cm)
    mask = (rn[:, None] < N) & (rm[None, :] < M)
    tl.store(C, acc, mask=mask)

##
## interfaces
##

def matmul(a, b, dtype=torch.float16):
    # device information
    assert(a.device == b.device)
    device = a.device

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
    matmul_kernel[grid](
        a, b, c, N, M, K, san, sak, sbk, sbm, scn, scm,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # return result
    return c
