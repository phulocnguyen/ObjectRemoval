import torch
import cv2
import numpy as np
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv import Config

CONFIG_PATH = "configs/swin/swin_base_upernet_ade20k.py"
CHECKPOINT_PATH = "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k/upernet_swin_base_patch4_window7_512x512_160k_ade20k_20201225_195325-9eebe41d.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
swin_model = init_segmentor(Config.fromfile(CONFIG_PATH), CHECKPOINT_PATH, device=DEVICE)

def segment_object(image, bbox):

    x1, y1, x2, y2 = bbox
    cropped_img = image[y1:y2, x1:x2]
    
    if cropped_img.size == 0:
        return None  #
    

    resized_img = cv2.resize(cropped_img, (512, 512))

    mask_result = inference_segmentor(swin_model, resized_img)
    mask = (mask_result[0] > 0).astype(np.uint8)  

    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask_resized

    return full_mask
