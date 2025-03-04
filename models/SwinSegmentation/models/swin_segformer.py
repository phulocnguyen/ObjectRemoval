import torch
import torch.nn as nn
from mmseg.models import build_segmentor
from mmcv import Config

class SwinSegFormer(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()

        cfg = Config.fromfile("configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py")
        cfg.model.decode_head.num_classes = num_classes

        self.model = build_segmentor(cfg.model)

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("Loaded pretrained weights from", pretrained_path)

        # Freeze backbone Swin
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Only fine-tuning decode_head (decoder)
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SwinSegFormer(num_classes=2, pretrained_path="segformer_mit-b2.pth").to(device)

print("Model ready!")
