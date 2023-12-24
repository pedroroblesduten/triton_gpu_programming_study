import tabulate
import torch

import triton
import triton.language as tl

@triton.jit
def _dropout(
        x_ptr,
        output_ptr,
        n_elements,
        p,
        seed,
        BLOCK_SIZE: tl.constexpr,
        ):

    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p

    output = tl.where(x_keep, x / (1-p), 0.0)
    tl.store(output_ptr + offsets, output, mask = mask)

def dropout(x, p, seed=42):
    output = torch.empty_like(x)
    assert x.is_contiguous()

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    _dropout[grid](
            x_ptr = x,
            output_ptr = output,
            n_elements = n_elements,
            p = p,
            seed = seed,
            BLOCK_SIZE=1024
            )
    return output

x = torch.randn(size=(10, )).cuda()
# Compare this to the baseline - dropout mask is never instantiated!
output = dropout(x, p=0.5, seed=123)
output2 = dropout(x, p=0.5, seed=123)
output3 = dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))





