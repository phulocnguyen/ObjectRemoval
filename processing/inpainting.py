import cv2
import numpy as np
from lama_cleaner.model_manager import ModelManager
import torch
from types import SimpleNamespace

# Load LaMa model
model = ModelManager(name="lama", device="cuda" if torch.cuda.is_available() else "cpu")

def removeObject(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Perform inpainting using LaMa with mask validation."""
    if mask is None or mask.sum() == 0:  # Ensure mask is valid
        return image  # Return original image if no mask is found
    
    config = SimpleNamespace(
        controlnet_method=None,  # Hoặc giá trị phù hợp nếu cần
        hd_strategy="Original",
        sd_scale=0.75,
        sd_mask_blur=7
    )

    return model(image, mask, config=config)


