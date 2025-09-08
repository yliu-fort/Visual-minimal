from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm  # for ViT and other backbones
except ImportError:
    timm = None


class VisualClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet18", num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        try:
            self.model = timm.create_model(self.backbone_name, pretrained=pretrained, num_classes=num_classes)
        except ValueError:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # quick test
    x = torch.randn(2, 3, 224, 224)
    model = VisualClassifier("resnet18")
    print(model(x).shape)  # torch.Size([2, 10])
    model = VisualClassifier("efficientnet_b0")
    print(model(x).shape)  # torch.Size([2, 10])
    model = VisualClassifier("vit_tiny_patch16_224")
    print(model(x).shape)  # torch.Size([2, 10])
    model = VisualClassifier("convnext_nano")
    print(model(x).shape)  # torch.Size([2, 10])
