import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.swin_segformer import SwinSegFormer
from utils.dataset import COCOSegmentation, get_dataloader
import os

def setup(rank, world_size):
    """Thiết lập môi trường Multi-GPU"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Dọn dẹp tiến trình Multi-GPU"""
    dist.destroy_process_group()

def train(rank, world_size, num_classes=91, epochs=10, lr=1e-4, batch_size=8):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    model = SwinSegFormer(num_classes, pretrained=True).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    train_loader = get_dataloader(
        "data/images", 
        "data/annotations/instances_train2017.json", 
        batch_size=batch_size,
        distributed=True,
        rank=rank,
        world_size=world_size
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[GPU {rank}] Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "swin_finetuned.pth")
        print("Fine-tuned model saved successfully!")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
