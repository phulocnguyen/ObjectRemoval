import torch
from segment_anything import sam_model_registry, SamPredictor

import requests

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
response = requests.get(url, stream=True)

with open("sam_vit_h.pth", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Download completed: sam_vit_h.pth")


def load_sam_model(model_type='vit_h', checkpoint='sam_vit_h.pth'):
    device="cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)