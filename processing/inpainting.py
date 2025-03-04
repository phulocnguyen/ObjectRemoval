import cv2
import numpy as np
import torch
from mat.model import MAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MAT.from_pretrained("fenglinglwb/mat").to(device)
model.eval()

def removeObject(image: np.ndarray, mask: np.ndarray) -> np.ndarray:

    if mask is None or mask.sum() == 0:  
        return image  

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    with torch.no_grad():
        inpainted_image = model(image_tensor, mask_tensor)

    inpainted_image = (inpainted_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return inpainted_image
