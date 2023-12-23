import torch

import triton
import triton.language as tl
from utils import *
import time
import matplotlib.pyplot as plt

@torch.jit.script
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    Softmax formula: softmax(x)_i = exp(x_i) / sum(exp(x_j))
    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0] # encontra o valor máximo em cada linha do tensor x.

    # read MN + M elements ; write MN elements
    z = x - x_max[:, None] # normaliza pelo máximo para evitar problemas de overflow numérico

    # read  MN elements ; write MN elements
    numerator = torch.exp(z) # Calcula a exponencial de cada elemento em z.

    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1) #  Calcula a soma de todas as exponenciais em cada linha

    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None] # Divide cada elemento do numerador pelo denominador correspondente

    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


# --- TRITON SOFTMAX ----
@triton.jit
def softmax_kernel(input_pointer,
                   out_pointer,
                   input_row_stride,
                   out_row_stride,
                   n_cols,
                   BLOCK_SIZE: tl.constexpr,
                   ):
    # the rows of the softmax are independet
    # so we parallelize across those
    row_idx = tl.program_id(0)

    # stride is how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_pointer + (row_idx*input_row_stride)


    # Cada thread dentro do bloco lidará com um elemento diferente da linha.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets # Calcula os ponteiros para cada elemento na linha atual.

    # Carrega os dados da linha atual na memória
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0) # normalizar pelo máximo para evitar overflow

    # calculo de fato da softmax
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # escrevendo os resultados na memória
    output_row_start_ptr = out_pointer + row_idx * out_row_stride # Calcula o ponteiro de início da linha atual nos dados de saída.
    output_ptrs = output_row_start_ptr + col_offsets # Calcula os ponteiros para cada elemento na linha de saída
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols) # Armazena os resultados da softmax nos locais apropriados na memória da GPU


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE, num_warps = calculate_settings_a(n_cols)

    # Allocate output
    y = torch.empty_like(x)

    # # Launch the kernel with calculated settings
    softmax_kernel[(n_rows, )](
            input_pointer = x,
            out_pointer = y,
            input_row_stride = x.stride(0),
            out_row_stride = y.stride(0),
            n_cols = n_cols,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps = num_warps
            )
    return y

# unit def test
torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    torch.cuda.synchronize()  # Ensure GPU syncs before timing

    start_time = time.time()
    for _ in range(100):  # Example: 100 iterations
        if provider == 'torch-native':
            _ = torch.softmax(x, axis=-1)
        elif provider == 'triton':
            _ = softmax(x)
        elif provider == 'torch-jit':
            _ = naive_softmax(x)
    torch.cuda.synchronize()  # Ensure GPU syncs after operation
    end_time = time.time()

    ms = (end_time - start_time) * 1000 / 100  # Convert to milliseconds and average per iteration
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

# Run the benchmark and capture the results
M = 4096  # Example matrix size
Ns = [128 * i for i in range(2, 100)]  # Range of N values
providers = ['triton', 'torch-native', 'torch-jit']
results = {provider: [] for provider in providers}

for N in Ns:
    for provider in providers:
        perf = benchmark(M, N, provider)
        results[provider].append(perf)

# Plot the results
plt.figure(figsize=(10, 6))
for provider, provider_results in results.items():
    plt.plot(Ns, provider_results, label=provider)
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('GB/s')
plt.title('Softmax Performance Comparison')
plt.legend()

# Save the plot
plt.savefig('./results/softmax.png')








