# import torch
# import cv2
# from processing.segmentation import segment_object
# from models.SwinSegmentation.models.swin_segformer import SwinSegFormer

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SwinSegFormer(num_classes=2, pretrained=True).to(device)  # num_classes = 2 (foreground/background)
# model.load_state_dict(torch.load("swin_finetuned.pth", map_location=device))

# image = cv2.imread("sample/test.jpg")
# mask = segment_object(image, model, device)

# cv2.imwrite("segmented_mask.png", mask * 255)  # Lưu mask để kiểm tra
import torch
from models.SwinSegmentation.models.swin_segformer import SwinSegFormer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SwinSegFormer(num_classes=91, pretrained=False).to(device)
checkpoint = torch.load("segformer_mit-b2.pth", map_location=device)

model.load_state_dict(checkpoint, strict=False)  # strict=False nếu bạn fine-tune
print("Checkpoint loaded successfully!")
