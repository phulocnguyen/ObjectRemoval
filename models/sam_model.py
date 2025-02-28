import torch
from segment_anything import sam_model_registry, SamPredictor

checkpoint = "/Users/phulocnguyen/Documents/Workspace/VideoObjectRemoval/sam_vit_h_4b8939.pth"

def load_sam_model(model_type='vit_h', checkpoint=checkpoint):
    device="cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)