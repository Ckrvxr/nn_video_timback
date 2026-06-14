import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, layers: list[int] = None):
        super().__init__()
        layers = layers or [3, 8, 15, 22]
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features
        self.blocks = nn.ModuleList()
        prev = 0
        for i, l in enumerate(layers):
            self.blocks.append(nn.Sequential(*features[prev:l]))
            for p in self.blocks[-1].parameters():
                p.requires_grad = False
            prev = l

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
            pred = (pred - mean) / std
            target = (target - mean) / std

        loss = 0.0
        for block in self.blocks:
            pred = block(pred)
            target = block(target)
            loss += torch.mean((pred - target) ** 2)
        return loss
