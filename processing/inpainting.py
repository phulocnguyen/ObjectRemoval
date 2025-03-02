import cv2
import numpy as np
from lama_cleaner.model_manager import ModelManager
import torch
from types import SimpleNamespace


model = ModelManager(name="lama", device="cuda" if torch.cuda.is_available() else "cpu")

def removeObject(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.sum() == 0: 
        return image 
    
    config = SimpleNamespace(
        controlnet_method=None, 
        hd_strategy="Original",
        sd_scale=0.75,
        sd_mask_blur=7
    )

    return model(image, mask, config=config)


