# Note: This is just me exploring different methods
"""
Baseline U-Net Architecture for Dental Anomaly Segmentation (v3 - Learning Fix)
=================================================================================
Architecture: U-Net with VGG-11 encoder (pretrained on ImageNet)
Activation Function: ReLU (encoder), with Dropout in decoder
Loss: Combined Dice + Weighted BCE (much better for segmentation)
Optimizer: Adam with Weight Decay
Training Strategy: Frozen encoder warmup (epochs 1-20), then full fine-tuning

Changes from v2 to fix poor learning (both IoU < 0.3, noisy predictions):
  1. Loss changed to Dice + BCE combined — BCE alone gives poor gradients
     for small anomaly regions; Dice directly optimises overlap
  2. Encoder frozen for first WARMUP_EPOCHS epochs — lets the decoder learn
     before corrupting pretrained VGG-11 features on small dataset
  3. Lower learning rate for encoder vs decoder after unfreeze (2-stage LR)
  4. Cosine annealing LR scheduler — smoother decay than ReduceLROnPlateau
  5. Gradient clipping added — stabilises training with combined loss

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
import json
import random
import gc

# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_SPLIT_FOLDER = '/content/drive/MyDrive/Capstone/data_split (1)'
IMAGE_DIR = os.path.join(DATA_SPLIT_FOLDER, 'radiographs/')
MASK_DIR  = os.path.join(DATA_SPLIT_FOLDER, 'masks/')

MODEL_SAVE_PATH = '/content/drive/MyDrive/Capstone/baseline_unet_vgg11_v3.pth'
OUTPUT_DIR      = '/content/drive/MyDrive/Capstone/test_predictions'

IMAGE_SIZE           = (840, 1615)
PHYSICAL_BATCH_SIZE  = 2
EFFECTIVE_BATCH_SIZE = 16
ACCUMULATION_STEPS   = EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE

MAX_EPOCHS          = 150
WARMUP_EPOCHS       = 20     # *** NEW: freeze encoder for first N epochs ***
LR_DECODER          = 1e-3   # Learning rate for decoder (always trained)
LR_ENCODER          = 1e-4   # *** NEW: lower LR for encoder after unfreeze ***
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 15     # Increased — cosine schedule needs more patience
PREDICT_THRESHOLD   = 0.5
POS_WEIGHT_CAP      = 20.0
DICE_WEIGHT         = 0.6    # *** NEW: weight of Dice in combined loss ***
BCE_WEIGHT          = 0.4    # *** NEW: weight of BCE in combined loss ***
GRAD_CLIP           = 1.0    # *** NEW: gradient clipping ***

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print("=" * 70)
print("U-NET + VGG-11 v3 (Learning Fix) - Dental Anomaly Segmentation")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Key changes from v2:")
print(f"  ✓ Loss: Dice ({DICE_WEIGHT}) + BCE ({BCE_WEIGHT}) combined")
print(f"  ✓ Encoder frozen for first {WARMUP_EPOCHS} epochs")
print(f"  ✓ 2-stage LR: decoder={LR_DECODER}, encoder={LR_ENCODER} (after unfreeze)")
print(f"  ✓ Cosine annealing LR scheduler")
print(f"  ✓ Gradient clipping (max norm={GRAD_CLIP})")
print()

# ============================================================
# Combined Dice + BCE Loss
# ============================================================
class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Weighted BCE Loss.

    Why this works better than BCE alone:
    - BCE treats each pixel independently; it can score well by predicting
      all background, giving the model no incentive to find anomalies
    - Dice loss directly measures overlap between prediction and ground truth,
      so gradients always push the model toward better segmentation
    - Combining them gives both pixel-level accuracy (BCE) and
      region-level overlap (Dice)
    """
    def __init__(self, pos_weight=None, dice_weight=0.6, bce_weight=0.4, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.smooth      = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ============================================================================
    # Dice loss is defined as its own function. I decided to add dice loss because
    # it is especially good for seeing if images overlap.
    # ============================================================================
    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Flatten spatial dimensions
        probs   = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice   # Loss = 1 - Dice coefficient

    def forward(self, logits, targets):
        bce  = self.bce(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

# ============================================================
# Calculate Class Weights (for BCE component)
# ============================================================
def calculate_positive_weight(mask_dir, mask_files, max_weight=20.0):
    print("Calculating class weights...")
    total_pos = total_neg = 0

    for img_name in mask_files:
        mask = np.array(Image.open(os.path.join(mask_dir, img_name)).convert('L'))
        mask_binary = (mask > 127).astype(np.float32)
        total_pos += np.sum(mask_binary == 1.0)
        total_neg += np.sum(mask_binary == 0.0)

    raw_weight = (total_neg / total_pos) if total_pos > 0 else 1.0
    pos_weight = min(raw_weight, max_weight)

    print(f"  Background pixels: {total_neg:,}")
    print(f"  Anomaly pixels:    {total_pos:,}")
    print(f"  Raw weight: {raw_weight:.2f}  →  Capped: {pos_weight:.2f}")
    print()
    return torch.tensor([pos_weight], dtype=torch.float32)

# ============================================================
# Dataset with Augmentation
# ============================================================
class DentalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, img_size=(840, 1615), augment=False):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.image_files = image_files
        self.img_size    = img_size
        self.augment     = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.image_files)

    def _augment(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        if random.random() > 0.8:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            # *** NEAREST interpolation prevents gray values in mask ***
            mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        if random.random() > 0.5:
            image = self.color_jitter(image)

        return image, mask

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        image = Image.open(os.path.join(self.image_dir, img_name)).convert('L')
        image = Image.merge('RGB', (image, image, image))
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)

        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('L')
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)

        mask = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, img_name

# ============================================================
# U-Net with VGG-11 Encoder + Dropout
# ============================================================
class UNet_VGG11(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.3):
        super(UNet_VGG11, self).__init__()

        vgg = models.vgg11(weights='IMAGENET1K_V1' if pretrained else None)
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

        self.up5  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = self._decoder_block(1024, 512, dropout_p)
        self.up4  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(1024, 512, dropout_p)
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(512, 256, dropout_p)
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(256, 128, dropout_p)
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64, dropout_p)
        self.out  = nn.Conv2d(64, 1, kernel_size=1)

    def _decoder_block(self, in_ch, out_ch, dropout_p):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def _match_sizes(self, enc, dec):
        if enc.shape[2:] != dec.shape[2:]:
            enc = F.interpolate(enc, size=dec.shape[2:], mode='bilinear', align_corners=False)
        return enc

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b  = self.bottleneck(e5)

        d5 = self.dec5(torch.cat([self.up5(b),  self._match_sizes(e5, self.up5(b))],  dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), self._match_sizes(e4, self.up4(d5))], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self._match_sizes(e3, self.up3(d4))], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self._match_sizes(e2, self.up2(d3))], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self._match_sizes(e1, self.up1(d2))], dim=1))

        out = self.out(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

    def freeze_encoder(self):
        """Freeze all VGG-11 encoder weights (used during warmup phase)."""
        for block in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            for param in block.parameters():
                param.requires_grad = False
        print("  Encoder FROZEN — training decoder only")

    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning after warmup."""
        for block in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            for param in block.parameters():
                param.requires_grad = True
        print("  Encoder UNFROZEN — full model fine-tuning")

# ============================================================
# Metrics
# ============================================================
def calculate_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).item()

def calculate_dice(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return ((2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)).item()

def calculate_pixel_accuracy(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    return ((pred == target).float().sum() / target.numel()).item()

# ============================================================
# Train / Validate / Test
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps, grad_clip):
    model.train()
    total_loss = total_iou = 0
    optimizer.zero_grad()

    for batch_idx, (images, masks, _) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # *** NEW: gradient clipping before optimizer step ***
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_iou  += calculate_iou(outputs, masks)

        del images, masks, outputs, loss
        if batch_idx % 2 == 0:
            torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    return total_loss / len(loader), total_iou / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()
            total_iou  += calculate_iou(outputs, masks)
            del images, masks, outputs
    return total_loss / len(loader), total_iou / len(loader)

def evaluate_test_set(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = total_iou = total_dice = total_acc = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()
            total_iou  += calculate_iou(outputs, masks, threshold)
            total_dice += calculate_dice(outputs, masks, threshold)
            total_acc  += calculate_pixel_accuracy(outputs, masks, threshold)
            del images, masks, outputs
    n = len(loader)
    return {'loss': total_loss/n, 'iou': total_iou/n, 'dice': total_dice/n, 'pixel_acc': total_acc/n}

def save_predictions(model, loader, device, output_dir, threshold=0.5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating predictions → {output_dir}  (threshold={threshold})")
    with torch.no_grad():
        for images, _, filenames in loader:
            images = images.to(device)
            preds  = torch.sigmoid(model(images)).cpu().numpy()
            for i, fname in enumerate(filenames):
                pred_binary = (preds[i, 0] > threshold).astype(np.uint8) * 255
                Image.fromarray(pred_binary, mode='L').save(os.path.join(output_dir, fname))
            del images, preds
            torch.cuda.empty_cache()
    print(f"✓ Saved {len(loader.dataset)} masks")

# ============================================================
# Data Split
# ============================================================
print("=" * 70)
print("LOADING AND SPLITTING DATA (70 / 15 / 15)")
print("=" * 70)

with open(os.path.join(DATA_SPLIT_FOLDER, 'dataset_info.json')) as f:
    dataset_info = json.load(f)

all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.JPG')])
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
# Class Weights, Datasets, DataLoaders
# ============================================================
pos_weight = calculate_positive_weight(MASK_DIR, train_files, max_weight=POS_WEIGHT_CAP)
pos_weight = pos_weight.to(DEVICE)

train_dataset = DentalDataset(IMAGE_DIR, MASK_DIR, train_files, IMAGE_SIZE, augment=True)
val_dataset   = DentalDataset(IMAGE_DIR, MASK_DIR, val_files,   IMAGE_SIZE, augment=False)
test_dataset  = DentalDataset(IMAGE_DIR, MASK_DIR, test_files,  IMAGE_SIZE, augment=False)

train_loader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# Model, Loss, Optimizer
# ============================================================
model     = UNet_VGG11(pretrained=True, dropout_p=0.3).to(DEVICE)
criterion = DiceBCELoss(pos_weight=pos_weight, dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT)

# Start with only decoder parameters (encoder is frozen)
decoder_params = (
    list(model.bottleneck.parameters()) +
    list(model.up5.parameters())  + list(model.dec5.parameters()) +
    list(model.up4.parameters())  + list(model.dec4.parameters()) +
    list(model.up3.parameters())  + list(model.dec3.parameters()) +
    list(model.up2.parameters())  + list(model.dec2.parameters()) +
    list(model.up1.parameters())  + list(model.dec1.parameters()) +
    list(model.out.parameters())
)

optimizer = optim.Adam(decoder_params, lr=LR_DECODER, weight_decay=WEIGHT_DECAY)

# Cosine annealing over full training run
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

print("=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)
print(f"Loss:              Dice ({DICE_WEIGHT}) + BCE ({BCE_WEIGHT}) combined")
print(f"Warmup strategy:   Encoder frozen for first {WARMUP_EPOCHS} epochs")
print(f"LR (decoder):      {LR_DECODER}  |  LR (encoder after unfreeze): {LR_ENCODER}")
print(f"Scheduler:         CosineAnnealingLR (T_max={MAX_EPOCHS})")
print(f"Gradient clipping: max_norm={GRAD_CLIP}")
print(f"Dropout:           p=0.3 in bottleneck + all decoder blocks")
print(f"Weight decay:      {WEIGHT_DECAY}")
print(f"pos_weight cap:    {POS_WEIGHT_CAP}  (actual: {pos_weight.item():.2f})")
print("=" * 70)

# ============================================================
# Training Loop with Encoder Warmup
# ============================================================
print("\n" + "=" * 70)
print("TRAINING  (Warmup phase: decoder only for first {} epochs)".format(WARMUP_EPOCHS))
print("=" * 70)
print(f"{'Epoch':>5} | {'Phase':>8} | {'Train Loss':>10} | {'Train IoU':>10} | {'Val Loss':>10} | {'Val IoU':>10} | {'Gap':>7} | {'Time':>6}")
print("-" * 85)

# Freeze encoder at start
model.freeze_encoder()

best_val_iou     = 0.0
patience_counter = 0
encoder_unfrozen = False
start_time       = time.time()

for epoch in range(1, MAX_EPOCHS + 1):

    # *** NEW: Unfreeze encoder after warmup and rebuild optimizer with 2-stage LR ***
    if epoch == WARMUP_EPOCHS + 1 and not encoder_unfrozen:
        model.unfreeze_encoder()
        encoder_unfrozen = True

        # Rebuild optimizer with separate LRs for encoder vs decoder
        encoder_params = (
            list(model.enc1.parameters()) + list(model.enc2.parameters()) +
            list(model.enc3.parameters()) + list(model.enc4.parameters()) +
            list(model.enc5.parameters())
        )
        optimizer = optim.Adam([
            {'params': encoder_params,  'lr': LR_ENCODER},
            {'params': decoder_params,  'lr': LR_DECODER / 10}  # Decay decoder LR too
        ], weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
        )
        print(f"\n  → Epoch {epoch}: Encoder unfrozen. Encoder LR={LR_ENCODER}, Decoder LR={LR_DECODER/10}\n")

    phase = "warmup" if not encoder_unfrozen else "finetune"
    train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS, GRAD_CLIP)
    val_loss,   val_iou   = validate_epoch(model, val_loader, criterion, DEVICE)

    scheduler.step()
    torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 60
    gap        = train_iou - val_iou
    marker     = "★" if val_iou > best_val_iou else " "

    print(f"{epoch:5d} | {phase:>8} | {train_loss:10.4f} | {train_iou:10.4f} | {val_loss:10.4f} | {val_iou:10.4f} | {gap:7.4f} | {total_time:5.1f}m {marker}")

    if val_iou > best_val_iou:
        best_val_iou     = val_iou
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou':   val_iou,
            'val_loss':  val_loss,
            'train_files': train_files,
            'val_files':   val_files,
            'test_files':  test_files,
            'effective_batch_size': EFFECTIVE_BATCH_SIZE,
            'pos_weight': pos_weight.item(),
            'predict_threshold': PREDICT_THRESHOLD,
        }, MODEL_SAVE_PATH)
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} — best val IoU: {best_val_iou:.4f}")
        break

print("-" * 85)
print(f"Training complete!  Best validation IoU: {best_val_iou:.4f}")

# ============================================================
# Load Best Model → Test Evaluation
# ============================================================
print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded from epoch {checkpoint['epoch']}  |  Val IoU: {checkpoint['val_iou']:.4f}")

print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION  (never seen during training)")
print("=" * 70)

test_metrics = evaluate_test_set(model, test_loader, criterion, DEVICE, PREDICT_THRESHOLD)
print(f"  Test Loss:      {test_metrics['loss']:.4f}")
print(f"  Test IoU:       {test_metrics['iou']:.4f}")
print(f"  Test Dice:      {test_metrics['dice']:.4f}")
print(f"  Test Pixel Acc: {test_metrics['pixel_acc']:.4f}")
print(f"\n  Val IoU (reference): {checkpoint['val_iou']:.4f}")
print("=" * 70)

save_predictions(model, test_loader, DEVICE, OUTPUT_DIR, PREDICT_THRESHOLD)

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("COMPLETE SUMMARY")
print("=" * 70)
print(f"Architecture:   U-Net + VGG-11 (ImageNet pretrained)")
print(f"Loss:           Dice ({DICE_WEIGHT}) + BCE ({BCE_WEIGHT})")
print(f"Warmup:         Encoder frozen for first {WARMUP_EPOCHS} epochs")
print(f"Split:          {int(TRAIN_RATIO*100)}% Train / {int(VAL_RATIO*100)}% Val / {int(TEST_RATIO*100)}% Test")
print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
print(f"pos_weight:     {checkpoint['pos_weight']:.2f}")
print()
print(f"Best Val IoU:   {checkpoint['val_iou']:.4f}")
print(f"Test IoU:       {test_metrics['iou']:.4f}")
print(f"Test Dice:      {test_metrics['dice']:.4f}")
print(f"Test Pixel Acc: {test_metrics['pixel_acc']:.4f}")
print()
print(f"Model:        {MODEL_SAVE_PATH}")
print(f"Predictions:  {OUTPUT_DIR}")
print("=" * 70)
