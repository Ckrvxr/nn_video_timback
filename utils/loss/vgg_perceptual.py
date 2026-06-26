import torch
import torch.nn as nn
import torchvision.models as models


class VGGDistance(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:16]
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred_rgb.shape
        if H != 224 or W != 224:
            pred_rgb = nn.functional.interpolate(pred_rgb, (224, 224), mode='bilinear', align_corners=False)
            target_rgb = nn.functional.interpolate(target_rgb, (224, 224), mode='bilinear', align_corners=False)
        pred_feat = self.features(pred_rgb)
        target_feat = self.features(target_rgb)
        return (pred_feat - target_feat).pow(2).mean()


def vgg_distance(pred_rgb, target_rgb):
    vgg = VGGDistance().to(pred_rgb.device)
    return vgg(pred_rgb, target_rgb)
