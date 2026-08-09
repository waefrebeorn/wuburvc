import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def main():
    n = 1024
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    out = torch.empty(n, device='cuda')
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    err = (out - (x + y)).abs().max().item()
    print('triton add OK, maxerr:', err)
    assert err < 1e-6, 'FAIL'

if __name__ == '__main__':
    main()
