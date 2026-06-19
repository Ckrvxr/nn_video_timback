import torch
import torch.nn as nn


class DilatedHDCStream(nn.Module):
    def __init__(self, nf: int = 4, dilations: list[int] | None = None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.down = nn.Conv2d(1, nf, 3, 2, 1)

        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(nf, nf, 3, 1, d, dilation=d))
            convs.append(nn.PReLU(nf))
        self.convs = nn.ModuleList(convs)

        self.up = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(nf, 1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        for mod in self.convs:
            x = mod(x)
        x = self.up(x)
        return 0.1 * torch.tanh(x)


class MergedDilatedHDCStream(nn.Module):
    def __init__(self, num_experts: int, nf: int = 2,
                 dilations: list[int] | None = None):
        super().__init__()
        G = num_experts
        if dilations is None:
            dilations = [1, 2, 4, 32]
        self.down = nn.Conv2d(G, nf * G, 3, 2, 1, groups=G)

        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(nf * G, nf * G, 3, 1, d, dilation=d, groups=G))
            convs.append(nn.PReLU(nf * G))
        self.convs = nn.ModuleList(convs)

        self.up = nn.Sequential(
            nn.Conv2d(nf * G, nf * G * 4, 3, 1, 1, groups=G),
            nn.PixelShuffle(2),
            nn.Conv2d(nf * G, G, 3, 1, 1, groups=G),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = x.repeat_interleave(self.convs[0].groups, dim=1)
        x = self.down(x)
        for m in self.convs:
            x = m(x)
        x = self.up(x)
        return 0.1 * torch.tanh(x)


class ParallelExperts(nn.Module):
    def __init__(self, num_experts: int, nf: int = 4, dilations: list[int] | None = None):
        super().__init__()
        self.num_experts = num_experts
        self.nf = nf
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.dilations = dilations

        self.down_weight = nn.Parameter(torch.zeros(num_experts, nf, 1, 3, 3))
        self.down_bias = nn.Parameter(torch.zeros(num_experts, nf))

        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()
        self.prelu_weights = nn.ParameterList()
        
        for d in dilations:
            self.conv_weights.append(nn.Parameter(torch.zeros(num_experts, nf, nf, 3, 3)))
            self.conv_biases.append(nn.Parameter(torch.zeros(num_experts, nf)))
            self.prelu_weights.append(nn.Parameter(torch.zeros(num_experts, nf)))

        self.up1_weight = nn.Parameter(torch.zeros(num_experts, nf * 4, nf, 3, 3))
        self.up1_bias = nn.Parameter(torch.zeros(num_experts, nf * 4))
        self.up2_weight = nn.Parameter(torch.zeros(num_experts, 1, nf, 3, 3))
        self.up2_bias = nn.Parameter(torch.zeros(num_experts, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_weight, a=5**0.5)
        nn.init.zeros_(self.down_bias)
        for w_c, b_c, w_p in zip(self.conv_weights, self.conv_biases, self.prelu_weights):
            nn.init.kaiming_uniform_(w_c, a=5**0.5)
            nn.init.zeros_(b_c)
            nn.init.constant_(w_p, 0.25)
        nn.init.kaiming_uniform_(self.up1_weight, a=5**0.5)
        nn.init.zeros_(self.up1_bias)
        nn.init.kaiming_uniform_(self.up2_weight, a=5**0.5)
        nn.init.zeros_(self.up2_bias)

    def _apply(self, fn):
        def wrapped_fn(t):
            if t.dim() != 4:
                device = None
                dtype = None
                non_blocking = False
                if hasattr(fn, "__closure__") and fn.__closure__:
                    for cell in fn.__closure__:
                        val = cell.cell_contents
                        if isinstance(val, torch.device):
                            device = val
                        elif isinstance(val, torch.dtype):
                            dtype = val
                        elif isinstance(val, bool):
                            non_blocking = val
                return t.to(device=device, dtype=dtype, non_blocking=non_blocking)
            return fn(t)
        return super()._apply(wrapped_fn)

    def forward(self, x: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = idx.shape[1]
        nf = self.nf

        x_rep = x.repeat(1, k, 1, 1)
        x_rep = x_rep.view(1, B * k, H, W)

        flat_idx = idx.view(-1)  # [B * k]

        import torch.nn.functional as F
        W_down = self.down_weight[flat_idx].view(B * k * nf, 1, 3, 3)
        b_down = self.down_bias[flat_idx].view(B * k * nf)

        x_feat = F.conv2d(x_rep, W_down, b_down, stride=2, padding=1, groups=B * k)

        for i, d in enumerate(self.dilations):
            W_c = self.conv_weights[i][flat_idx].view(B * k * nf, nf, 3, 3)
            b_c = self.conv_biases[i][flat_idx].view(B * k * nf)
            w_p = self.prelu_weights[i][flat_idx].view(B * k * nf)

            x_feat = F.conv2d(x_feat, W_c, b_c, stride=1, padding=d, dilation=d, groups=B * k)
            x_feat = F.prelu(x_feat, w_p)

        W_up1 = self.up1_weight[flat_idx].view(B * k * nf * 4, nf, 3, 3)
        b_up1 = self.up1_bias[flat_idx].view(B * k * nf * 4)
        x_feat = F.conv2d(x_feat, W_up1, b_up1, stride=1, padding=1, groups=B * k)

        H_half, W_half = x_feat.shape[-2], x_feat.shape[-1]
        x_feat = x_feat.view(B * k, nf * 4, H_half, W_half)
        x_feat = F.pixel_shuffle(x_feat, 2)
        x_feat = x_feat.view(1, B * k * nf, H, W)

        W_up2 = self.up2_weight[flat_idx].view(B * k, nf, 3, 3)
        b_up2 = self.up2_bias[flat_idx].view(B * k)
        x_feat = F.conv2d(x_feat, W_up2, b_up2, stride=1, padding=1, groups=B * k)

        out_delta = x_feat.view(B, k, H, W)
        w_res = weights.view(B, k, 1, 1)
        out = (out_delta * w_res).sum(dim=1, keepdim=True)
        return 0.1 * torch.tanh(out)
