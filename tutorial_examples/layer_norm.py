import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False


@triton.jit
def _layer_norm_fwd_fused(
        X, # pointer to the input
        Y, # pointer to the output
        W, # pointer to the weights
        B, # pointer to the biases
        Mean, # pointer to the Mean
        Rstd, # pointer to the 1/std
        stride, # how much to increase the pointer when movind by 1 row
        N, # number of columns in X
        eps, # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
        ):

    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # calcula a media
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # itera sobre as colunas da matriz X em blocos de tamanhao BLOCK_SIZE
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE) # indices das colunas para o blcoo atual
        a = tl.load(X + cols, mask=cols<N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x-mean, 0.0) # calcula o desvio da media
        _var += x * x
    var = tl.sum(_var, axis=0)/N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask)

        x_hat = (x - mean ) * rstd # layer norm
        y = x_hat * w + b # learnable parameters

        tl.store(Y + cols, y, mask=mask)



# -- STAGE 1: _layer_norm_bwd_dx_fuse
# calcula os gradientes da entrada DX e acumula parcialmente
# os gradientes dos pesos DW e DB
# A acumulação é feita em buffers compartilhados, usando uma estratégia de redução paralela para evitar conflitos de escrita.


# estratégia de redução paralela, onde os resultados parciais de diferentes linhas são acumulados de forma eficiente.

@triton.jit
def _layer_norm_bwd_dx_fused(
        DX, # ponteiro para o gradiente da entrada
        DY, # ponteiro para o gradiente da saida
        DW, # ponteiro para a soma partial do gradiente dos pesos
        DB, # ponteiro para a soma parcial do grandiente dos biases
        X, # ponteiro para a entrada
        W, # ponteiro para os pesos
        B, # ponteiro para o biases
        Mean, # ponteiro para a media
        Rstd, # ponteiro para 1/std
        Lock, # ponteiro para o Lock
        stride, # quanto que precisamos aumentar o ponteiro para andar uma linha
        N, # numero de colunas
        eps,
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        ):

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    X += row * stride
    DY += row * stride
    DX += row * stride

    # Configuração de Locks e Ponteiros de Gradientes para Redução Paralela
    lock_id = row % GROUP_SIZE_M #lock usado para controle de acesso concorrente
    Lock += lock_id

    Count = Lock + GROUP_SIZE_M # para rastreas quantas linhas ja escreveram nos buffers de gradiente parcial

    # ajustam os pnteiros para o local correto dos gradientes parcial
    # lock_id * N garante que cada linha tenha seu proprio espaço separado no buffer
    # N garante que cada grupo de linhas comece em um novo "bloco" no buffer.
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    # carrega os dados para a SRAM
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # -- formula do VECTOR-JACOBIAN PRODUCT (VJP) --
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)

    wdy = w * dy # pondeira o gradiente pelos pesos
    wdy = tl.where(mask, wdy, 0.0)

    # calcula constantes intermediarias
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N

    dw = (wdy - (xhat * c1 + c2)) * rstd

    partial_dw = (dy * xhat)


    # -- Acumulação de Somas Parciais para DW e DB --

    # Cálculo dos Gradientes Parciais
    # dL/dw = dy * xhat
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)

    #  lock garante que apenas uma thread da GPU possa escrever nos buffers de gradiente parcial (DW, DB) por vez
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass #  loop de espera ativa onde a thread continua verificando o estado do lock sem realizar nenhuma outra operação

    count = tl.load(Count) # rastreia quantas linhas já escreveram nos buffers de gradiente parcial (DW e DB).

    if count == 0: # primeira thread
        tl.atomic_xchg(Count, 1) # muda o valor do contator para 1
    else:
        # acumulam os gradientes parciais de várias threads.
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)

    # Armazenamento dos Gradientes Parciais e Liberação do Lock
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)


# -- STAGE 2: _layer_norm_bwd_dwdb --
# responsável por consolidar as somas parciais dos gradientes dos pesos (DW) e dos biases (DB)
# em gradientes finais (FINAL_DW, FINAL_DB).

@triton.jit
def _layer_norm_bwd_dwdb(
        DW, # pointer to the partial sum of weights gradientes
        DB, # pointer to the partial sum of biases gradient
        FINAL_DW, # pointer to the weights gradient
        FINAL_DB, # pointer to the biases gradient
        M, # GROUP_SIZE_M
        N, # number of columns,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        ):

    # Declaração e Mapeamento de IDs:
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Inicialização dos Tensores Temporários:
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iteração sobre as Linhas dos Buffers de Gradientes Parciais:
    for i in range(0, M, BLOCK_SIZE_M): # A cada iteração, o loop pega um novo bloco de linhas para processar.
        rows = i + tl.arange(0, BLOCK_SIZE_M) # Calcula os índices das linhas que serão processados nesta iteração específica do loop.
        mask = (rows[:, None] < M) * (cols[None, :] < N) # dentro dos limites do tamanho do grupo M e do número de colunas N
        offs = rows[:, None] * cols[None, :] * M # Calcula os deslocamentos (offsets) no buffer para os índices de linha e coluna especificados.

        # Acumulação das Somas Parciais:
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)

    # Cálculo dos Gradientes Finais e Armazenamento:
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)

    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

# -- CLASSE LAYER NORM --
class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layer_norm_fwd_fused[(M, )](
                x_arg, y, weight, bias, mean, rstd,
                x_arg.stride(0), N, eps,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

        # atualizando o contexto
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors

        N = w.shape[0]
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        # alocando tensores necessarios
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
        _dw = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device)
        _db = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device)
        dw = torch.empty((w.shape[0], ), dtype=w.dtype, device=w.device)
        db = torch.empty((w.shape[0], ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        _layer_norm_bwd_dx_fused[(M, )](  #
            dx, dy, _dw, _db, x, w, b, m, v, locks,  #
            x_arg.stride(0), N, ctx.eps,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, GROUP_SIZE_M, N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128)
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':

        def y_fwd():
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

    if provider == 'torch':

        def y_fwd():
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

    if provider == 'apex':
        apex_layer_norm = apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype)

        def y_fwd():
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':

        def gbps(ms):
            return 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704

        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(1151, 8192, torch.float16)
#bench_layer_norm.run(save_path='.', print_data=True)









































































