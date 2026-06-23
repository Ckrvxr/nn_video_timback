import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect', bias=False)
        self.norm1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect', bias=False)
        self.norm2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.norm2(out)
        return F.relu(x + out, inplace=True)


class DWResBlock(nn.Module):
    """Depthwise-separable ResBlock — DW3×3 + PW1×1 per block.

    ~8.5× fewer params & ~17× less compute than standard ResBlock.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.dw1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect', groups=ch, bias=False)
        self.pw1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(ch)
        self.dw2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect', groups=ch, bias=False)
        self.pw2 = nn.Conv2d(ch, ch, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.dw1(x)
        out = self.pw1(out)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.dw2(out)
        out = self.pw2(out)
        out = self.norm2(out)
        return F.relu(x + out, inplace=True)


class RealTimeUNet4K_PureCNN(nn.Module):
    """Pure CNN UNet for 4K video enhancement — mathematically symmetric.

    5 depthwise-separable ResBlocks, fully symmetric encoder-decoder.
    Accepts arbitrary input size (padded to multiple of 4 internally).

    Input:  [B, 9, H, W] — concat(t-2, t-1, t)
    Output: [B, 3, H, W] — enhanced center frame (residual delta)
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        ch = base_ch

        # ── Stage 1: Input ──
        self.entry = nn.Sequential(
            nn.Conv2d(144, ch, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # ── Stage 2: Encoder ──
        self.down1 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )
        self.res1 = DWResBlock(ch * 2)

        self.down2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),
        )
        self.res2 = DWResBlock(ch * 4)

        # ── Stage 3: Bottleneck (center) — standard ResBlock for full-rank capacity ──
        self.bottleneck = nn.Sequential(*[ResBlock(ch * 4) for _ in range(3)])

        # ── Stage 4: Decoder ──
        self.up1 = nn.Conv2d(ch * 4, ch * 2, 3, padding=1, padding_mode='reflect', bias=False)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )
        self.res1_dec = DWResBlock(ch * 2)

        self.up2 = nn.Conv2d(ch * 2, ch, 3, padding=1, padding_mode='reflect', bias=False)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.res2_dec = DWResBlock(ch)

        # ── Stage 5: Output ──
        self.post_map = nn.Sequential(
            nn.Conv2d(ch, 144, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(144),
        )
        self.pred = nn.Conv2d(9, 3, 3, padding=1, padding_mode='reflect')
        self.delta_scale = nn.Parameter(torch.tensor([0.25, 0.04, 0.04]))

        nn.init.zeros_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def forward(self, x: torch.Tensor):
        x_center = x[:, 6:9]
        s = 4
        pH = (s - x.shape[2] % s) % s
        pW = (s - x.shape[3] % s) % s
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH))

        h = F.pixel_unshuffle(x, s)
        skip1 = self.entry(h)
        h = self.down1(skip1)
        skip2 = self.res1(h)
        h = self.down2(skip2)
        h = self.res2(h)
        h = self.bottleneck(h)

        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.up1(h)
        h = self.fuse1(torch.cat([h, skip2], dim=1))
        h = self.res1_dec(h)

        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.up2(h)
        h = self.fuse2(torch.cat([h, skip1], dim=1))
        h = self.res2_dec(h)

        h = self.post_map(h)
        h = F.pixel_shuffle(h, s)
        h = self.pred(h)

        if pH or pW:
            h = h[:, :, :x_center.shape[2], :x_center.shape[3]]

        delta = torch.tanh(h) * self.delta_scale.view(1, -1, 1, 1)
        return x_center + delta


