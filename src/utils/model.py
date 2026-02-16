import torch
import torch.nn as nn
import timm


def build_model(model_name: str = "efficientnet_b0", pretrained=True, drop_rate=0.2) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)


def set_backbone_trainable(m: nn.Module, trainable: bool):
    for _, p in m.named_parameters():
        p.requires_grad = trainable
    # sblocca sempre la head
    for head_name in ["classifier", "fc", "head"]:
        if hasattr(m, head_name):
            for p in getattr(m, head_name).parameters():
                p.requires_grad = True


def find_last_conv_layer(model: nn.Module) -> str:
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
    return last_name
