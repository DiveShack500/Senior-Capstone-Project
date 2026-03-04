# ================================================================================
# Notes:
#       - Patch-based training with in-memory patching — no separate patch
#         extraction step. Patches are generated on the fly inside __getitem__
#         so nothing is saved to disk.
#
#       - Training patch strategy (proven to reach 90%+ in previous work):
#           * For images WITH anomalies: extract PATCHES_PER_LESION patches
#             centered on a random lesion pixel, with small random offset.
#             This ensures the model always sees anomaly content.
#           * For images WITHOUT anomalies: extract PATCHES_PER_BACKGROUND
#             random patches from anywhere in the image.
#           * This gives a balanced training set focused on hard examples.
#
#       - Inference / reconstruction (solves the previous reconstruction problem):
#           * A fixed sliding window grid is computed from the image dimensions
#             and patch size at inference time — no metadata file needed.
#           * Each patch is run through the model independently.
#           * Predicted patch masks are placed back at their exact pixel
#             coordinates in a blank 840x1615 canvas.
#           * Overlapping regions are averaged to reduce boundary artifacts.
#
#       - Attention U-Net (from v4) is used as the backbone — attention gates
#         on all skip connections focus the model on anomaly regions even
#         within the smaller patch view.
#
#       - Warmup + 2-stage LR + Dice+BCE + gradient clipping all retained
#         from v3/v4.
# ================================================================================
"""
Patch-Based Attention U-Net for Dental Anomaly Segmentation (v5)
=================================================================
Architecture: Attention U-Net with VGG-11 encoder
Patching:     In-memory on the fly — no separate extraction step
Training:     Lesion-centered patches + random background patches
Inference:    Sliding window grid → stitch predicted masks back to 840x1615
Loss:         Combined Dice + Weighted BCE
Training:     Frozen encoder warmup (epochs 1-20), then full fine-tuning

Author: Liam Shack
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
import numpy as np
from PIL import Image
import os
import time
import torch.nn.functional as F
import random
import gc

# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_SPLIT_FOLDER = '/content/drive/MyDrive/Capstone/data'
IMAGE_DIR         = os.path.join(DATA_SPLIT_FOLDER, 'Radiographs')
MASK_DIR          = os.path.join(DATA_SPLIT_FOLDER, 'mask')

MODEL_SAVE_PATH = '/content/drive/MyDrive/Capstone/patch_attention_unet_v6.pth'
OUTPUT_DIR      = '/content/drive/MyDrive/Capstone/test_predictions_v6'

PATCH_SIZE           = 256          # Proven to work in previous experiments
PATCHES_PER_LESION   = 4            # Patches centered on lesion per abnormal image
PATCHES_PER_BG       = 8            # Random patches per normal image
LESION_JITTER        = 32           # Max random offset from lesion center (pixels)

PHYSICAL_BATCH_SIZE  = 8            # Larger — patches are smaller than full images
EFFECTIVE_BATCH_SIZE = 32
ACCUMULATION_STEPS   = EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE

MAX_EPOCHS          = 150
WARMUP_EPOCHS       = 20
LR_DECODER          = 1e-3
LR_ENCODER          = 1e-4
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 50
PREDICT_THRESHOLD   = 0.5
POS_WEIGHT_CAP      = 10.0
DICE_WEIGHT         = 0.6
BCE_WEIGHT          = 0.4
GRAD_CLIP           = 1.0

INFERENCE_STRIDE    = 128           # 50% overlap during inference sliding window

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print("=" * 70)
print("PATCH-BASED ATTENTION U-NET v5 — Dental Anomaly Segmentation")
print("=" * 70)
print(f"Device:              {DEVICE}")
print(f"Patch size:          {PATCH_SIZE}×{PATCH_SIZE}")
print(f"Patches per lesion:  {PATCHES_PER_LESION} (centered on anomaly)")
print(f"Patches per BG:      {PATCHES_PER_BG} (random normal image)")
print(f"Inference stride:    {INFERENCE_STRIDE} (50% overlap → averaged)")
print(f"Physical batch:      {PHYSICAL_BATCH_SIZE}  |  Effective: {EFFECTIVE_BATCH_SIZE}")
print()


# ============================================================
# Patch Extraction Utilities
# ============================================================
def extract_lesion_patch(image_np, mask_np, patch_size, jitter=32):
    """
    Extract a patch centered on a random lesion pixel with small random offset.

    Args:
        image_np: (H, W, 3) numpy array
        mask_np:  (H, W) binary numpy array
        patch_size: side length of square patch
        jitter: max pixel offset from lesion center

    Returns:
        img_patch (H, W, 3), mask_patch (H, W) numpy arrays
        or None if no lesion pixels found
    """
    h, w = mask_np.shape
    lesion_coords = np.argwhere(mask_np > 0)  # (N, 2) array of (row, col)

    if len(lesion_coords) == 0:
        return None, None

    # Pick a random lesion pixel as center
    center = lesion_coords[random.randint(0, len(lesion_coords) - 1)]
    cy, cx = center

    # Add random jitter
    cy += random.randint(-jitter, jitter)
    cx += random.randint(-jitter, jitter)

    # Compute patch boundaries clamped to image
    half = patch_size // 2
    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    y1 = min(y1, h - patch_size)
    x1 = min(x1, w - patch_size)
    y2 = y1 + patch_size
    x2 = x1 + patch_size

    return image_np[y1:y2, x1:x2], mask_np[y1:y2, x1:x2]


def extract_random_patch(image_np, mask_np, patch_size):
    """Extract a patch from a random position in the image."""
    h, w = mask_np.shape
    y1 = random.randint(0, h - patch_size)
    x1 = random.randint(0, w - patch_size)
    return image_np[y1:y1+patch_size, x1:x1+patch_size], \
           mask_np[y1:y1+patch_size, x1:x1+patch_size]


def get_sliding_window_coords(img_h, img_w, patch_size, stride):
    """
    Compute all (y1, x1) top-left coordinates for a sliding window grid.
    Ensures full coverage — last row/col adjusted to stay within bounds.

    Returns: list of (y1, x1) tuples
    """
    coords = []
    y = 0
    while y + patch_size <= img_h:
        x = 0
        while x + patch_size <= img_w:
            coords.append((y, x))
            x += stride
        # Final column — align to right edge
        if x - stride + patch_size < img_w:
            coords.append((y, img_w - patch_size))
        y += stride
    # Final row — align to bottom edge
    if y - stride + patch_size < img_h:
        x = 0
        while x + patch_size <= img_w:
            coords.append((img_h - patch_size, x))
            x += stride
        if x - stride + patch_size < img_w:
            coords.append((img_h - patch_size, img_w - patch_size))

    return list(set(coords))  # deduplicate corner overlaps


# ============================================================
# Dataset — patches generated on the fly
# ============================================================
class PatchDentalDataset(Dataset):
    """
    Generates training patches in memory during __getitem__.

    For abnormal images (mask has white pixels):
        - Extracts PATCHES_PER_LESION patches centered on lesion pixels
        - Each patch is a separate sample in the dataset

    For normal images (all-black mask):
        - Extracts PATCHES_PER_BG random patches

    Augmentation applied per-patch during training.
    """
    def __init__(self, image_dir, mask_dir, image_files,
                 patch_size=256, patches_per_lesion=4, patches_per_bg=2,
                 jitter=32, augment=False):
        self.image_dir        = image_dir
        self.mask_dir         = mask_dir
        self.patch_size       = patch_size
        self.patches_per_lesion = patches_per_lesion
        self.patches_per_bg   = patches_per_bg
        self.jitter           = jitter
        self.augment          = augment
        self.color_jitter     = transforms.ColorJitter(brightness=0.2, contrast=0.2)

        # Build flat list of (filename, patch_index) samples
        # Each image contributes multiple patches
        self.samples = []
        for fname in image_files:
            mask = np.array(
                Image.open(os.path.join(mask_dir, fname)).convert('L')
            )
            has_lesion = (mask > 127).any()
            n_patches  = patches_per_lesion if has_lesion else patches_per_bg
            for i in range(n_patches):
                self.samples.append((fname, has_lesion))

        # Normalisation — ImageNet stats (encoder is VGG-11 pretrained)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def _augment_patch(self, img_pil, mask_pil):
        if random.random() > 0.5:
            img_pil  = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)
        if random.random() > 0.8:
            img_pil  = TF.vflip(img_pil)
            mask_pil = TF.vflip(mask_pil)
        if random.random() > 0.5:
            angle    = random.uniform(-10, 10)
            img_pil  = TF.rotate(img_pil, angle)
            mask_pil = TF.rotate(mask_pil, angle,
                                  interpolation=TF.InterpolationMode.NEAREST)
        if random.random() > 0.5:
            img_pil = self.color_jitter(img_pil)
        return img_pil, mask_pil

    def __getitem__(self, idx):
        fname, has_lesion = self.samples[idx]

        # Load full image and mask
        image_full = np.array(
            Image.open(os.path.join(self.image_dir, fname)).convert('L')
        )
        # Convert grayscale to 3-channel for VGG-11
        image_full = np.stack([image_full] * 3, axis=-1)
        mask_full  = (np.array(
            Image.open(os.path.join(self.mask_dir, fname)).convert('L')
        ) > 127).astype(np.float32)

        # Extract patch
        if has_lesion:
            img_patch, mask_patch = extract_lesion_patch(
                image_full, mask_full, self.patch_size, self.jitter
            )
            # Fallback to random if lesion extraction fails
            if img_patch is None:
                img_patch, mask_patch = extract_random_patch(
                    image_full, mask_full, self.patch_size
                )
        else:
            img_patch, mask_patch = extract_random_patch(
                image_full, mask_full, self.patch_size
            )

        # Convert to PIL for augmentation
        img_pil  = Image.fromarray(img_patch.astype(np.uint8))
        mask_pil = Image.fromarray((mask_patch * 255).astype(np.uint8))

        if self.augment:
            img_pil, mask_pil = self._augment_patch(img_pil, mask_pil)

        # To tensor and normalise
        image_tensor = self.normalize(transforms.ToTensor()(img_pil))
        mask_tensor  = torch.from_numpy(
            (np.array(mask_pil) > 127).astype(np.float32)
        ).unsqueeze(0)

        return image_tensor, mask_tensor, fname


# ============================================================
# Attention Gate
# ============================================================
class AttentionGate(nn.Module):
    """
    Soft attention gate for U-Net skip connections.
    Suppresses background features and highlights anomaly regions.

    g: gating signal from decoder (lower res, higher semantic)
    x: encoder skip features (same res as decoder at this level)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ============================================================
# Attention U-Net
# ============================================================
class AttentionUNet_VGG11(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.3):
        super(AttentionUNet_VGG11, self).__init__()

        vgg      = models.vgg11(weights='IMAGENET1K_V1' if pretrained else None)
        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(*features[:3])
        self.enc2 = nn.Sequential(*features[3:6])
        self.enc3 = nn.Sequential(*features[6:11])
        self.enc4 = nn.Sequential(*features[11:16])
        self.enc5 = nn.Sequential(*features[16:21])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True)
        )

        self.att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)

        self.up5  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = self._dec(1024, 512, dropout_p)
        self.up4  = nn.ConvTranspose2d(512, 512,  kernel_size=2, stride=2)
        self.dec4 = self._dec(1024, 512, dropout_p)
        self.up3  = nn.ConvTranspose2d(512, 256,  kernel_size=2, stride=2)
        self.dec3 = self._dec(512,  256, dropout_p)
        self.up2  = nn.ConvTranspose2d(256, 128,  kernel_size=2, stride=2)
        self.dec2 = self._dec(256,  128, dropout_p)
        self.up1  = nn.ConvTranspose2d(128, 64,   kernel_size=2, stride=2)
        self.dec1 = self._dec(128,  64,  dropout_p)
        self.out  = nn.Conv2d(64, 1, kernel_size=1)

    def _dec(self, in_ch, out_ch, dp):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dp),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def _match(self, enc, dec):
        if enc.shape[2:] != dec.shape[2:]:
            enc = F.interpolate(enc, size=dec.shape[2:],
                                mode='bilinear', align_corners=False)
        return enc

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b  = self.bottleneck(e5)

        up5 = self.up5(b);  d5 = self.dec5(torch.cat([up5, self.att5(up5, self._match(e5, up5))], 1))
        up4 = self.up4(d5); d4 = self.dec4(torch.cat([up4, self.att4(up4, self._match(e4, up4))], 1))
        up3 = self.up3(d4); d3 = self.dec3(torch.cat([up3, self.att3(up3, self._match(e3, up3))], 1))
        up2 = self.up2(d3); d2 = self.dec2(torch.cat([up2, self.att2(up2, self._match(e2, up2))], 1))
        up1 = self.up1(d2); d1 = self.dec1(torch.cat([up1, self.att1(up1, self._match(e1, up1))], 1))

        out = self.out(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:],
                                mode='bilinear', align_corners=False)
        return out

    def freeze_encoder(self):
        for b in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            for p in b.parameters():
                p.requires_grad = False
        print("  Encoder FROZEN — training decoder + attention gates only")

    def unfreeze_encoder(self):
        for b in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            for p in b.parameters():
                p.requires_grad = True
        print("  Encoder UNFROZEN — full model fine-tuning")


# ============================================================
# Loss
# ============================================================
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=None, dice_weight=0.6, bce_weight=0.4, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def dice_loss(self, logits, targets):
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + \
               self.dice_weight * self.dice_loss(logits, targets)


# ============================================================
# Class Weights
# ============================================================
def calculate_positive_weight(mask_dir, mask_files, max_weight=20.0):
    print("Calculating class weights from training patches...")
    total_pos = total_neg = 0
    for fname in mask_files:
        mask        = np.array(Image.open(os.path.join(mask_dir, fname)).convert('L'))
        mask_binary = (mask > 127).astype(np.float32)
        total_pos  += mask_binary.sum()
        total_neg  += (1 - mask_binary).sum()
    raw    = total_neg / max(total_pos, 1)
    capped = min(raw, max_weight)
    print(f"  Background pixels: {total_neg:,.0f}")
    print(f"  Anomaly pixels:    {total_pos:,.0f}")
    print(f"  Raw weight: {raw:.2f}  →  Capped: {capped:.2f}")
    print()
    return torch.tensor([capped], dtype=torch.float32)


# ============================================================
# Metrics
# ============================================================
def calculate_iou(pred, target, threshold=0.5):
    pred         = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).item()

def calculate_dice(pred, target, threshold=0.5):
    pred         = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return ((2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)).item()

def calculate_pixel_accuracy(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    return ((pred == target).float().sum() / target.numel()).item()


# ============================================================
# Train / Validate
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device,
                accumulation_steps, grad_clip):
    model.train()
    total_loss = total_iou = 0.0
    num_batches = len(loader)
    optimizer.zero_grad()

    for batch_idx, (images, masks, _) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss    = criterion(outputs, masks) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or \
           (batch_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_iou  += calculate_iou(outputs, masks)

        del images, masks, outputs, loss
        if batch_idx % 4 == 0:
            torch.cuda.empty_cache()

    return total_loss / num_batches, total_iou / num_batches


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = 0.0
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks  = images.to(device), masks.to(device)
            outputs        = model(images)
            total_loss    += criterion(outputs, masks).item()
            total_iou     += calculate_iou(outputs, masks)
            del images, masks, outputs
    return total_loss / len(loader), total_iou / len(loader)


# ============================================================
# UPDATED FULL-IMAGE INFERENCE
# ============================================================
def predict_full_image(model, image_path, device, patch_size, 
                       stride, threshold, normalize):
    """
    Inference with updated confidence thresholding and 
    no artificial post-processing filters.
    """
    model.eval()
    
    image_np = np.array(Image.open(image_path).convert('L'))
    H, W     = image_np.shape
    image_3ch = np.stack([image_np] * 3, axis=-1)

    # Accumulators for averaging overlapping predictions
    pred_sum   = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)

    coords = get_sliding_window_coords(H, W, patch_size, stride)

    with torch.no_grad():
        for (y1, x1) in coords:
            y2 = y1 + patch_size
            x2 = x1 + patch_size

            patch     = image_3ch[y1:y2, x1:x2].astype(np.uint8)
            patch_pil = Image.fromarray(patch)
            patch_t   = normalize(transforms.ToTensor()(patch_pil))
            patch_t   = patch_t.unsqueeze(0).to(device)

            logit     = model(patch_t)
            prob      = torch.sigmoid(logit).squeeze().cpu().numpy()

            pred_sum  [y1:y2, x1:x2] += prob
            pred_count[y1:y2, x1:x2] += 1.0

    # Final reconstruction via averaging
    pred_avg    = pred_sum / np.maximum(pred_count, 1e-6)
    
    # Strictly thresholding based on model confidence
    pred_binary = (pred_avg > threshold).astype(np.uint8) * 255
    return pred_binary


def evaluate_test_set(model, test_files, image_dir, mask_dir,
                      device, patch_size, stride, threshold, normalize):
    """Evaluate on full images using sliding window inference."""
    model.eval()
    total_iou = total_dice = total_acc = 0.0

    for fname in test_files:
        pred_mask = predict_full_image(
            model, os.path.join(image_dir, fname),
            device, patch_size, stride, threshold, normalize
        )
        gt_mask = (np.array(
            Image.open(os.path.join(mask_dir, fname)).convert('L')
        ) > 127).astype(np.uint8)

        pred_t = torch.from_numpy(pred_mask / 255.0).float().unsqueeze(0)
        gt_t   = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0)

        # Compute metrics directly on numpy arrays
        intersection = (pred_mask // 255 & gt_mask).sum()
        union        = (pred_mask // 255 | gt_mask).sum()
        total_iou   += (intersection + 1e-6) / (union + 1e-6)

        dice_num     = 2 * intersection
        dice_den     = (pred_mask // 255).sum() + gt_mask.sum()
        total_dice  += (dice_num + 1e-6) / (dice_den + 1e-6)

        total_acc   += (pred_mask // 255 == gt_mask).mean()

        torch.cuda.empty_cache()

    n = len(test_files)
    return {'iou': total_iou/n, 'dice': total_dice/n, 'pixel_acc': total_acc/n}


def save_predictions(model, test_files, image_dir, device, output_dir,
                     patch_size, stride, threshold, normalize):
    """Save full reconstructed mask for every test image."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating full-image predictions → {output_dir}")
    for fname in test_files:
        pred_mask = predict_full_image(
            model, os.path.join(image_dir, fname),
            device, patch_size, stride, threshold, normalize
        )
        Image.fromarray(pred_mask, mode='L').save(
            os.path.join(output_dir, fname)
        )
        torch.cuda.empty_cache()
    print(f"✓ Saved {len(test_files)} full masks ({840}×{1615})")


# ============================================================
# Data Split
# ============================================================
print("=" * 70)
print("LOADING AND SPLITTING DATA (70 / 15 / 15)")
print("=" * 70)

all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.upper().endswith('.JPG')])
print(f"Total files: {len(all_files)}")

random.seed(42)
random.shuffle(all_files)

n_total = len(all_files)
n_train = int(TRAIN_RATIO * n_total)
n_val   = int(VAL_RATIO   * n_total)

train_files = all_files[:n_train]
val_files   = all_files[n_train : n_train + n_val]
test_files  = all_files[n_train + n_val:]

print(f"  Training:   {len(train_files):>4}  ({len(train_files)/n_total*100:.1f}%)")
print(f"  Validation: {len(val_files):>4}  ({len(val_files)/n_total*100:.1f}%)")
print(f"  Test:       {len(test_files):>4}  ({len(test_files)/n_total*100:.1f}%)  ← held out")
print()

# ============================================================
# Class Weights
# ============================================================
pos_weight = calculate_positive_weight(
    MASK_DIR, train_files, max_weight=POS_WEIGHT_CAP
).to(DEVICE)

# ============================================================
# Datasets and DataLoaders
# ============================================================
train_dataset = PatchDentalDataset(
    IMAGE_DIR, MASK_DIR, train_files,
    patch_size=PATCH_SIZE,
    patches_per_lesion=PATCHES_PER_LESION,
    patches_per_bg=PATCHES_PER_BG,
    jitter=LESION_JITTER,
    augment=True
)
val_dataset = PatchDentalDataset(
    IMAGE_DIR, MASK_DIR, val_files,
    patch_size=PATCH_SIZE,
    patches_per_lesion=PATCHES_PER_LESION,
    patches_per_bg=PATCHES_PER_BG,
    jitter=LESION_JITTER,
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=PHYSICAL_BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# Normalisation transform reused during inference
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

print(f"Training patches:   {len(train_dataset):,}  "
      f"({PATCHES_PER_LESION} per abnormal + {PATCHES_PER_BG} per normal)")
print(f"Validation patches: {len(val_dataset):,}")
print()

# ============================================================
# Model, Loss, Optimizer
# ============================================================
model     = AttentionUNet_VGG11(pretrained=True, dropout_p=0.3).to(DEVICE)
criterion = DiceBCELoss(pos_weight=pos_weight,
                        dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT)

decoder_params = (
    list(model.bottleneck.parameters()) +
    list(model.att5.parameters()) + list(model.up5.parameters()) + list(model.dec5.parameters()) +
    list(model.att4.parameters()) + list(model.up4.parameters()) + list(model.dec4.parameters()) +
    list(model.att3.parameters()) + list(model.up3.parameters()) + list(model.dec3.parameters()) +
    list(model.att2.parameters()) + list(model.up2.parameters()) + list(model.dec2.parameters()) +
    list(model.att1.parameters()) + list(model.up1.parameters()) + list(model.dec1.parameters()) +
    list(model.out.parameters())
)

optimizer = optim.Adam(decoder_params, lr=LR_DECODER, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
)

total_params = sum(p.numel() for p in model.parameters())

print("=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)
print(f"Architecture:      Attention U-Net + VGG-11 (pretrained)")
print(f"Total parameters:  {total_params:,}")
print(f"Loss:              Dice ({DICE_WEIGHT}) + BCE ({BCE_WEIGHT})")
print(f"Warmup:            Encoder frozen for first {WARMUP_EPOCHS} epochs")
print(f"LR decoder:        {LR_DECODER}  |  LR encoder: {LR_ENCODER}")
print(f"Scheduler:         CosineAnnealingLR")
print(f"Gradient clipping: {GRAD_CLIP}")
print(f"pos_weight:        {pos_weight.item():.2f} (capped at {POS_WEIGHT_CAP})")
print(f"Inference:         Sliding window stride={INFERENCE_STRIDE} → averaged overlap")
print("=" * 70)

# ============================================================
# Training Loop
# ============================================================
print("\n" + "=" * 70)
print(f"TRAINING  (Warmup: decoder + attention for first {WARMUP_EPOCHS} epochs)")
print("=" * 70)
print(f"{'Epoch':>5} | {'Phase':>8} | {'Train Loss':>10} | {'Train IoU':>10} | "
      f"{'Val Loss':>10} | {'Val IoU':>10} | {'Gap':>7} | {'Time':>6}")
print("-" * 90)

model.freeze_encoder()

best_val_iou     = 0.0
patience_counter = 0
encoder_unfrozen = False
start_time       = time.time()

for epoch in range(1, MAX_EPOCHS + 1):

    if epoch == WARMUP_EPOCHS + 1 and not encoder_unfrozen:
        model.unfreeze_encoder()
        encoder_unfrozen = True
        encoder_params = (
            list(model.enc1.parameters()) + list(model.enc2.parameters()) +
            list(model.enc3.parameters()) + list(model.enc4.parameters()) +
            list(model.enc5.parameters())
        )
        optimizer = optim.Adam([
            {'params': encoder_params,  'lr': LR_ENCODER},
            {'params': decoder_params,  'lr': LR_DECODER / 10}
        ], weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
        )
        print(f"\n  → Epoch {epoch}: Encoder unfrozen. "
              f"Encoder LR={LR_ENCODER}, Decoder LR={LR_DECODER/10}\n")

    phase = "warmup" if not encoder_unfrozen else "finetune"

    train_loss, train_iou = train_epoch(
        model, train_loader, criterion, optimizer,
        DEVICE, ACCUMULATION_STEPS, GRAD_CLIP
    )
    val_loss, val_iou = validate_epoch(model, val_loader, criterion, DEVICE)

    scheduler.step()
    torch.cuda.empty_cache()

    elapsed = (time.time() - start_time) / 60
    gap     = train_iou - val_iou
    marker  = "★" if val_iou > best_val_iou else " "

    print(f"{epoch:5d} | {phase:>8} | {train_loss:10.4f} | {train_iou:10.4f} | "
          f"{val_loss:10.4f} | {val_iou:10.4f} | {gap:7.4f} | {elapsed:5.1f}m {marker}")

    if val_iou > best_val_iou:
        best_val_iou     = val_iou
        patience_counter = 0
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou':              val_iou,
            'val_loss':             val_loss,
            'train_files':          train_files,
            'val_files':            val_files,
            'test_files':           test_files,
            'patch_size':           PATCH_SIZE,
            'inference_stride':     INFERENCE_STRIDE,
            'pos_weight':           pos_weight.item(),
            'predict_threshold':    PREDICT_THRESHOLD,
        }, MODEL_SAVE_PATH)
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} — best val IoU: {best_val_iou:.4f}")
        break

print("-" * 90)
print(f"Training complete!  Best validation IoU: {best_val_iou:.4f}")

# ============================================================
# Load Best Model → Full-Image Test Evaluation
# ============================================================
print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded from epoch {checkpoint['epoch']}  |  Val IoU: {checkpoint['val_iou']:.4f}")

print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION — Full 840×1615 Images")
print("(Sliding window inference → reconstructed full masks)")
print("=" * 70)

test_metrics = evaluate_test_set(
    model, test_files, IMAGE_DIR, MASK_DIR,
    DEVICE, PATCH_SIZE, INFERENCE_STRIDE, PREDICT_THRESHOLD, NORMALIZE
)
print(f"  Test IoU:       {test_metrics['iou']:.4f}")
print(f"  Test Dice:      {test_metrics['dice']:.4f}")
print(f"  Test Pixel Acc: {test_metrics['pixel_acc']:.4f}")
print("=" * 70)

save_predictions(
    model, test_files, IMAGE_DIR, DEVICE, OUTPUT_DIR,
    PATCH_SIZE, INFERENCE_STRIDE, PREDICT_THRESHOLD, NORMALIZE
)

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("COMPLETE SUMMARY")
print("=" * 70)
print(f"Architecture:      Patch-Based Attention U-Net + VGG-11")
print(f"Patch size:        {PATCH_SIZE}×{PATCH_SIZE}")
print(f"Training patches:  {len(train_dataset):,}")
print(f"Inference:         Sliding window stride={INFERENCE_STRIDE}, averaged overlap")
print(f"Output masks:      Full 840×1615 (reconstructed from patches)")
print(f"Loss:              Dice ({DICE_WEIGHT}) + BCE ({BCE_WEIGHT})")
print(f"pos_weight:        {checkpoint['pos_weight']:.2f}")
print(f"Split:             {int(TRAIN_RATIO*100)}% / {int(VAL_RATIO*100)}% / {int(TEST_RATIO*100)}%")
print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
print()
print(f"Best Val IoU:   {checkpoint['val_iou']:.4f}")
print(f"Test IoU:       {test_metrics['iou']:.4f}")
print(f"Test Dice:      {test_metrics['dice']:.4f}")
print(f"Test Pixel Acc: {test_metrics['pixel_acc']:.4f}")
print()
print(f"Model:          {MODEL_SAVE_PATH}")
print(f"Predictions:    {OUTPUT_DIR}")
print("=" * 70)
