import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ****************************
# Dataset 
# ****************************
class KvasirDataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_id + ".jpg")).convert("L")
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        return img, mask

def load_split(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_loaders(batch_size=2, img_size=192):
    data_root = "Kvasir-SEG"
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")
    split_dir = "Kvasir-SEG-main/Data-split"

    
    if not os.path.exists(split_dir):
        print("Warning: Data splits not found. Please ensure paths are correct.")
        return None, None, None

    train_ids = load_split(os.path.join(split_dir, "train.txt"))
    val_ids = load_split(os.path.join(split_dir, "val.txt"))
     
    split_idx = len(val_ids) // 2
    test_ids = val_ids[split_idx:]
    val_ids = val_ids[:split_idx]             

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_ds = KvasirDataset(train_ids, img_dir, mask_dir, transform, target_transform)
    val_ds = KvasirDataset(val_ids, img_dir, mask_dir, transform, target_transform)
    test_ds = KvasirDataset(test_ids, img_dir, mask_dir, transform, target_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

# ****************************
# Core concepts: Simplified Mamba block (1D conv scan)
# ****************************
class SimpleMambaBlock(nn.Module):
    """Mamba‑inspired block with 1D convolution along sequence (CPU‑friendly)."""
    def __init__(self, dim, expand=2):
        super().__init__()
        d_inner = int(dim * expand)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=5, padding=2, groups=d_inner)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.act(x)
        x = x.transpose(1, 2)

        out = x * z
        out = self.out_proj(out)
        return out + residual

# ****************************
# Edge-Aware Module
# ****************************
class EdgeAwareModule(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sobel_y', torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))
        self.fusion = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_ch, 1)
        )

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x**2 + edge_y**2)
        edge_map = self.fusion(edge)
        return torch.sigmoid(edge_map)

# ****************************
# Multi‑scale fusion module 
# ****************************
class MultiScaleFusion(nn.Module):
    def __init__(self, channels=[48, 96, 192, 384]):
        super().__init__()
        self.up_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        
        
        for i in range(3, 0, -1):
            self.up_convs.append(nn.Conv2d(channels[i], channels[i-1], 1))
            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(channels[i-1] * 2, channels[i-1], 3, padding=1),
                nn.BatchNorm2d(channels[i-1]), 
                nn.ReLU(inplace=True)
            ))
            
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]), 
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        # feats: [e1, e2, e3, e4]
        fused = feats[-1]
        
        for idx, i in enumerate(range(3, 0, -1)):
            up = F.interpolate(fused, size=feats[i-1].shape[2:], mode='bilinear', align_corners=False)
            up = self.up_convs[idx](up)
            
            fused = torch.cat([up, feats[i-1]], dim=1)
            
            fused = self.fuse_convs[idx](fused)
            
        return self.final_conv(fused)

# ****************************
# Main Model 
# ****************************
class LightweightEMamba(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_dim=48):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim), nn.GELU()
        )
        self.mamba1 = SimpleMambaBlock(base_dim)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim*2), nn.GELU()
        )
        self.mamba2 = SimpleMambaBlock(base_dim*2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim*4), nn.GELU()
        )
        self.mamba3 = SimpleMambaBlock(base_dim*4)

        self.enc4 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*8, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim*8), nn.GELU()
        )
        self.mamba4 = SimpleMambaBlock(base_dim*8)

        self.edge = EdgeAwareModule(in_ch, out_ch)
        self.fusion = MultiScaleFusion([base_dim, base_dim*2, base_dim*4, base_dim*8])

        
        self.decoder = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_dim, out_ch, 1)
        )

    def forward(self, x):
        def apply_mamba(block, feat):
            B, C, H, W = feat.shape
            feat_flat = feat.flatten(2).transpose(1, 2)
            feat_flat = block(feat_flat)
            return feat_flat.transpose(1, 2).view(B, C, H, W)

        e1 = self.enc1(x); e1 = apply_mamba(self.mamba1, e1) 
        e2 = self.enc2(e1); e2 = apply_mamba(self.mamba2, e2)
        e3 = self.enc3(e2); e3 = apply_mamba(self.mamba3, e3) 
        e4 = self.enc4(e3); e4 = apply_mamba(self.mamba4, e4) 

        edge_map = self.edge(x) 
        fused = self.fusion([e1, e2, e3, e4])

        out = self.decoder(fused)
        out = out * (1 + edge_map) 

        return torch.sigmoid(out), edge_map

# ****************************
# Metrics and training 
# ****************************

# Soft Dice Loss for Training (Differentiable)
def soft_dice_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * inter + eps) / (union + eps)
    return 1. - dice.mean()

# Hard Dice Score for Evaluation (Non-Differentiable)
def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

def train_epoch(model, loader, optimizer,scheduler, criterion, device):
    model.train()
    total_loss = 0
    for img, mask in tqdm(loader, desc="Training", leave=False):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred, _ = model(img)
        
        # Use SOFT dice loss for gradient flow
        loss = criterion(pred, mask) + soft_dice_loss(pred, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    dice, iou = 0, 0
    with torch.no_grad():
        for img, mask in tqdm(loader, desc="Evaluating", leave=False):
            img, mask = img.to(device), mask.to(device)
            pred, _ = model(img)
            dice += dice_score(pred, mask).item()
            iou += iou_score(pred, mask).item()
    return dice / len(loader), iou / len(loader)

def measure_inference_time(model, loader, device, num=50):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (img, _) in enumerate(loader):
            if i >= num: break
            img = img.to(device)
            start = time.time()
            _ = model(img)
            times.append(time.time() - start)
    return np.mean(times) if len(times) > 0 else 0

# ****************************
# Main
# ****************************
def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders(batch_size=2, img_size=192)
    
    if train_loader is None:
        return 

    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    epochs = 10
    best_dice = 0

    
    model = LightweightEMamba(in_ch=3, out_ch=1, base_dim=48)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=5e-3, 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch
    )
    criterion = nn.BCELoss()

   
    
    print("\n=== Training Lightweight eMMamba (core concepts) ===\n")
    for ep in range(1, epochs+1):
        loss = train_epoch(model, train_loader, optimizer,scheduler, criterion, device)
        d, i = evaluate(model, val_loader, device)
        print(f"Epoch {ep}/{epochs} | Loss: {loss:.4f} | Val Dice: {d:.4f} | Val IoU: {i:.4f}")
        if d > best_dice:
            best_dice = d
            torch.save(model.state_dict(), "best_emmamba_core.pth")

    # Final test
    test_dice, test_iou = evaluate(model, test_loader, device)
    inf_time = measure_inference_time(model, test_loader, device, num=50)
    print("\n" + "="*50)
    print("Final Results on Test Set")
    print("="*50)
    print(f"Test Dice:  {test_dice:.4f}")
    print(f"Test IoU:   {test_iou:.4f}")
    print(f"Inference time per image (CPU): {inf_time:.3f} sec")
    print("="*50)

if __name__ == "__main__":
    main()