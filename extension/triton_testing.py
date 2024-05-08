# triton testing

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 32

ceil_div = lambda x, y: (x + y - 1) // y

@triton.jit
def int_to_half_kernel(x_int, x_hlf, offset, N, BLOCK_SIZE : tl.constexpr):
    pid_n = tl.program_id(0)
    idx_n = (pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) % N
    X_int = tl.load(x_int + idx_n)
    X_hlf = X_int.to(tl.float16)
    X_hlf = X_hlf - offset
    mask = pid_n < N
    tl.store(x_hlf + idx_n, X_hlf, mask=mask)

def int_to_half(x_int, offset=0.0):
    N, = x_int.shape
    x_hlf = torch.empty(N, dtype=torch.float16, device='cuda')
    grid = ceil_div(N, BLOCK_SIZE),
    int_to_half_kernel[grid](x_int, x_hlf, offset, N, BLOCK_SIZE)
    return x_hlf
