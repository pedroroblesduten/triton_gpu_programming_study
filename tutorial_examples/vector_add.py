import torch
import triton
import triton.language as tl
import time
from utils import *
import matplotlib.pyplot as plt

#### ADD KERNEL FUCTION FROM OPENAI TUTORIALS ####

# decorator Just-In-Time Compilation
# diz que a função deve ser compilada somente no momento de execução
@triton.jit
def add_kernel(x_pointer, # ponteiro para a entrada x
               y_pointer, # ponteiro para a entrada y
               out_pointer, # ponteiro para a saída (resultado da soma)
               n_elements, # numero total de elementos vetores
               BLOCK_SIZE: tl.constexpr, # quantos elementos cada bloco de execucao de processar
               # "bloco" é uma unidade de threads que executam em paralelo.
               #tl.constexpr diz que BLOCK_SIZE deve ser constante em tempo de execução
               ):

    # retorna o identificador único para a thread atual dentro da grade de execução do kernel (PID).
    pid = tl.program_id(axis=0)

    # um grande vetor é dividido em pedaços menores, e cada pedaço é processado por uma instância separada do kernel.
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE) # cria uma sequência de números inteiros começando em 0 e indo até BLOCK_SIZE - 1 + block_start.
    mask = offsets < n_elements #  garantir que o kernel não tente acessar índices fora dos limites do vetor

    # calcula os endereços de memória reais dos elementos do vetor x que precisam ser carregados.
    # Cada elemento em offsets é somado ao x_ptr para obter o endereço exato na memória.
    x = tl.load(x_pointer + offsets, mask=mask)
    y = tl.load(y_pointer + offsets, mask=mask)

    out = x + y

    # escrever os resultados da adição de volta para a memória DRAM.
    tl.store(out_pointer + offsets, out, mask=mask)

# declarando de fato a função add
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x) # precisamos de pre alocar o tensor de saida

    assert x.is_cuda and y.is_cuda and output.is_cuda, 'ERROR: both tensors need to be in cuda device'
    n_elements = output.numel() # calcula o numero total de elementos para a soma
    BLOCK_SIZE, num_warps = calculate_settings(n_elements)

    # triton.cdiv é usado para calcular a dimensao da grade de lançamento
    # aqui estamos calculando quantos blocos de threads são necessários para processar todos os elementos
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # o kernel é indexado com a grade de lançamento grid (quantos blocos de threads serão utilizados)

    add_kernel[grid](x_pointer=x,
                    y_pointer=y,
                    out_pointer=output,
                    n_elements=n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps)
    return output


# Assuming the function 'add' is already defined as per your previous Triton setup.

torch.manual_seed(0)
size = 8500
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

# Time the PyTorch addition
start_time_torch = time.time()
for _ in range(100000):
    output_torch = x + y
torch.cuda.synchronize()  # Ensure CUDA operations are finished
end_time_torch = time.time()
total_time_torch = end_time_torch - start_time_torch

# Time the Triton addition
start_time_triton = time.time()
for _ in range(100000):
    output_triton = add(x, y)
torch.cuda.synchronize()  # Ensure CUDA operations are finished
end_time_triton = time.time()
total_time_triton = end_time_triton - start_time_triton

# Print results
print(f'Total time for PyTorch addition: {total_time_torch:.6f} seconds')
print(f'Total time for Triton addition: {total_time_triton:.6f} seconds')

# Check the maximum difference between the two methods
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


import torch
import triton
import matplotlib.pyplot as plt
import time

# Assuming the 'add' function is already defined as per your Triton setup

def manual_benchmark(size):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)

    # Benchmark Torch
    start = time.time()
    for _ in range(100):  # Number of runs can be adjusted
        _ = x + y
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100

    # Benchmark Triton
    start = time.time()
    for _ in range(100):  # Number of runs can be adjusted
        _ = add(x, y)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100

    gbps = lambda ms: 12 * size / ms / 1e-6
    return gbps(torch_time), gbps(triton_time)

# Sizes to benchmark
sizes = [2**i for i in range(12, 28, 1)]
torch_results = []
triton_results = []

# Running the benchmark
for size in sizes:
    torch_gbps, triton_gbps = manual_benchmark(size)
    torch_results.append(torch_gbps)
    triton_results.append(triton_gbps)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(sizes, torch_results, label='Torch', color='green', linestyle='-')
plt.plot(sizes, triton_results, label='Triton', color='blue', linestyle='-')
plt.xscale('log')
plt.xlabel('Size')
plt.ylabel('GB/s')
plt.title('Vector Add Performance Comparison')
plt.legend()
plt.savefig('./results/vector_add.png')

