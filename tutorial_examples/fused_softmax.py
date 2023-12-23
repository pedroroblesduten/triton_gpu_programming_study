import torch

import triton
import triton.language as tl

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

