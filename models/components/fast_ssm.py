import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ssm_fwd_kernel(
    hx_ptr, u_ptr, out_ptr, s_save_ptr, state_ptr,
    A_ptr, WsT_ptr,
    L,
    stride_hx_b, stride_hx_l, stride_hx_d,
    stride_u_b, stride_u_l, stride_u_n,
    stride_out_b, stride_out_l, stride_out_d,
    stride_s_b, stride_s_l, stride_s_n,
    N: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_n = tl.arange(0, N)
    offs_d = tl.arange(0, D)

    A = tl.load(A_ptr + offs_n[:, None] * N + offs_n[None, :])
    WsT = tl.load(WsT_ptr + offs_n[:, None] * D + offs_d[None, :])

    state = tl.load(state_ptr + pid * N + offs_n)

    for t in range(L):
        offs_hx = pid * stride_hx_b + t * stride_hx_l + offs_d * stride_hx_d
        offs_u = pid * stride_u_b + t * stride_u_l + offs_n * stride_u_n
        offs_out = pid * stride_out_b + t * stride_out_l + offs_d * stride_out_d
        offs_s = pid * stride_s_b + t * stride_s_l + offs_n * stride_s_n

        hx_t = tl.load(hx_ptr + offs_hx)
        u_t = tl.load(u_ptr + offs_u)

        h = hx_t + tl.sum(state[:, None] * WsT, axis=0)
        tl.store(out_ptr + offs_out, h)
        tl.store(s_save_ptr + offs_s, state)

        state = tl.sum(state[None, :] * A, axis=1) + u_t

    tl.store(state_ptr + pid * N + offs_n, state)


@triton.jit
def _ssm_bwd_kernel(
    grad_out_ptr, s_save_ptr,
    grad_hx_ptr, grad_u_ptr, grad_state_ptr,
    A_ptr, WsT_ptr,
    L,
    stride_go_b, stride_go_l, stride_go_d,
    stride_s_b, stride_s_l, stride_s_n,
    stride_ghx_b, stride_ghx_l, stride_ghx_d,
    stride_gu_b, stride_gu_l, stride_gu_n,
    N: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_n = tl.arange(0, N)
    offs_d = tl.arange(0, D)

    A = tl.load(A_ptr + offs_n[:, None] * N + offs_n[None, :])
    WsT = tl.load(WsT_ptr + offs_n[:, None] * D + offs_d[None, :])

    grad_s = tl.zeros([N,], dtype=tl.float32)

    for t in range(L - 1, -1, -1):
        offs_go = pid * stride_go_b + t * stride_go_l + offs_d * stride_go_d
        offs_ghx = pid * stride_ghx_b + t * stride_ghx_l + offs_d * stride_ghx_d
        offs_gu = pid * stride_gu_b + t * stride_gu_l + offs_n * stride_gu_n

        grad_h = tl.load(grad_out_ptr + offs_go)
        tl.store(grad_hx_ptr + offs_ghx, grad_h)
        tl.store(grad_u_ptr + offs_gu, grad_s)

        grad_s = tl.sum(grad_h[None, :] * WsT, axis=1) + tl.sum(A * grad_s[:, None], axis=0)

    tl.store(grad_state_ptr + pid * N + offs_n, grad_s)


class _SSMFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hx, u, state, A, WsT):
        B, L, D = hx.shape
        N = A.shape[-1]

        s_save = torch.empty(B, L, N, device=hx.device, dtype=hx.dtype)
        out = torch.empty_like(hx)

        grid = (B,)
        _ssm_fwd_kernel[grid](
            hx, u, out, s_save, state,
            A.contiguous(), WsT.contiguous(),
            L,
            hx.stride(0), hx.stride(1), hx.stride(2),
            u.stride(0), u.stride(1), u.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            s_save.stride(0), s_save.stride(1), s_save.stride(2),
            N=N, D=D,
        )

        ctx.save_for_backward(s_save, A, WsT)
        ctx.L = L
        ctx.D = D
        ctx.N = N
        return out, state

    @staticmethod
    def backward(ctx, grad_out, grad_state_final):
        s_save, A, WsT = ctx.saved_tensors
        L, D, N = ctx.L, ctx.D, ctx.N
        B = grad_out.shape[0]

        grad_hx = torch.empty(B, L, D, device=grad_out.device, dtype=grad_out.dtype)
        grad_u = torch.empty(B, L, N, device=grad_out.device, dtype=grad_out.dtype)
        grad_state = torch.empty(B, N, device=grad_out.device, dtype=grad_out.dtype)

        grid = (B,)
        _ssm_bwd_kernel[grid](
            grad_out.contiguous(), s_save.contiguous(),
            grad_hx.contiguous(), grad_u.contiguous(), grad_state.contiguous(),
            A.contiguous(), WsT.contiguous(),
            L,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            s_save.stride(0), s_save.stride(1), s_save.stride(2),
            grad_hx.stride(0), grad_hx.stride(1), grad_hx.stride(2),
            grad_u.stride(0), grad_u.stride(1), grad_u.stride(2),
            N=N, D=D,
        )

        return grad_hx, grad_u, grad_state, None, None


class FastSSM(nn.Module):
    def __init__(self, d_model: int = 16, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.fc = nn.Linear(d_model + d_state, d_model, bias=True)
        self.state_proj = nn.Linear(d_model, d_state, bias=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        D, N = self.d_model, self.d_state
        W_fc = self.fc.weight
        W_x = W_fc[:, :D]
        W_s = W_fc[:, D:]
        W_sp = self.state_proj.weight

        A = W_sp @ W_s
        B = W_sp @ W_x
        b = W_sp @ self.fc.bias + self.state_proj.bias

        hx = x @ W_x.T + self.fc.bias
        u = x @ B.T + b

        out, new_state = _SSMFn.apply(hx, u, state, A, W_s.T)
        out = self.norm(out)
        return out, new_state
