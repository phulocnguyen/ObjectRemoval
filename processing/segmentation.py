import torch
import cv2
import numpy as np
from models.SwinSegmentation.models.swin_segformer import SwinSegFormer
from torchvision import transforms

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0) 

def segment_object(image, model, device="cuda"):

    model.eval()
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor) 
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    
    return refined_mask
