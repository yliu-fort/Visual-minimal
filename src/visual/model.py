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

        if self.backbone_name.startswith("resnet"):
            self.model = getattr(tv_models, self.backbone_name)(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        elif self.backbone_name.startswith("efficientnet"):
            self.model = getattr(tv_models, self.backbone_name)(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

        elif self.backbone_name.startswith("vit"):
            if timm is None:
                raise ImportError("timm is required for ViT models. Install via `pip install timm`.")
            self.model = timm.create_model(self.backbone_name, pretrained=pretrained, num_classes=num_classes)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # quick test
    x = torch.randn(2, 3, 224, 224)
    model = VisualClassifier("resnet18")
    print(model(x).shape)  # torch.Size([2, 10])
    model = VisualClassifier("efficientnet_b3")
    print(model(x).shape)  # torch.Size([2, 10])
    model = VisualClassifier("vit_tiny_patch16_224")
    print(model(x).shape)  # torch.Size([2, 10])
