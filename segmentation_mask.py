from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)

# ================================================================================
# Patch-Based Attention U-Net for Dental Anomaly Segmentation (v13)
# ================================================================================
# Changes vs v12 (fixed) — all motivated by white-pixel over-prediction:
#
#   [THRESHOLD]  PREDICT_THRESHOLD 0.15 → 0.45
#   [SAMPLING]   LESION_PATCH_RATIO 0.90 → 0.60
#   [LOSS]       POS_WEIGHT_CAP 50.0 → 8.0
#   [SCHEDULE]   WARMUP_EPOCHS 20 → 30
#   [LR]         LR_ENCODER 1e-4 → 5e-5
#
# ================================================================================
# DICE COEFFICIENT — WHERE AND HOW IT IS CALCULATED
# ================================================================================
#
# Dice is calculated in TWO separate places in this script, for two different
# purposes. They use different inputs (soft vs hard) and different smoothing.
#
# ── LOCATION 1: DiceBCELoss.dice_loss  (training loss, soft Dice) ────────────
#
#   Called during: every training and validation batch
#   Input:         raw model logits (before sigmoid)
#   Formula:
#
#       p     = sigmoid(logits)          # soft probabilities, values in [0, 1]
#       inter = sum(p * t)               # soft intersection (per sample in batch)
#       denom = sum(p) + sum(t)          # sum of predicted + ground truth pixels
#       dice_loss = 1 - (2 * inter + smooth) / (denom + smooth)
#
#   Key properties:
#   - Uses SOFT probabilities, not binary predictions — gradient can flow through
#   - smooth = 1e-6 is added to numerator and denominator to avoid division by
#     zero on all-negative patches (no lesion pixels)
#   - Computed PER SAMPLE in the batch, then averaged across the batch
#   - Returns a LOSS (1 - Dice), so minimising it maximises Dice
#   - Combined with BCE: total_loss = 0.2 * BCE + 0.8 * dice_loss
#
# ── LOCATION 2: full_image_metrics  (evaluation metric, hard Dice) ───────────
#
#   Called during: every FULL_VAL_EVERY epochs on the val set, and once on test
#   Input:         binary predicted mask (already thresholded by model.predict)
#                  and binary ground truth mask
#   Formula:
#
#       pred_bin = predicted mask pixels > 127   # hard 0/1
#       gt_bin   = ground truth pixels   > 127   # hard 0/1
#       inter    = sum(pred_bin AND gt_bin)       # true positives (TP)
#       denom    = sum(pred_bin) + sum(gt_bin)    # TP+FP + TP+FN = 2TP+FP+FN
#       dice     = (2 * inter) / denom            # = 2TP / (2TP + FP + FN)
#
#   Key properties:
#   - Uses HARD binary predictions (already thresholded at PREDICT_THRESHOLD)
#   - NO smoothing term — if both pred and gt are all-zero (normal image with
#     no lesion and perfect prediction), denom == 0 and dice = 1.0 explicitly
#   - Computed PER IMAGE, then averaged across all images in the split
#     (macro average — each image contributes equally regardless of lesion size)
#   - This is the metric used for early stopping and final test reporting
#
# ── KEY DIFFERENCES BETWEEN THE TWO ─────────────────────────────────────────
#
#   Property          | Loss (Location 1)     | Metric (Location 2)
#   ------------------|-----------------------|------------------------
#   Input             | Soft probabilities    | Hard binary mask
#   Smoothing         | Yes (1e-6)            | No (explicit 0/0 = 1)
#   Granularity       | Per patch in batch    | Per full image
#   Average           | Batch mean            | Macro (per-image) mean
#   Purpose           | Drive weight updates  | Evaluate generalisation
#
# ================================================================================

import gc
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models


# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEG_ROOT        = "/content/drive/MyDrive/Capstone/segmentation_data/SEGMENTATION MODEL"

TRAIN_RAD       = os.path.join(SEG_ROOT, "Train", "Train", "Radiographs")
TRAIN_RAD_FLIP  = os.path.join(SEG_ROOT, "Train", "Train", "Radiographs_flipped")
TRAIN_MASK      = os.path.join(SEG_ROOT, "Train", "Train", "mask")
TRAIN_MASK_FLIP = os.path.join(SEG_ROOT, "Train", "Train", "mask_flipped")

TEST_RAD        = os.path.join(SEG_ROOT, "Test", "Radiographs")
TEST_MASK       = os.path.join(SEG_ROOT, "Test", "mask")

MODEL_SAVE_PATH = "/content/drive/MyDrive/Capstone/seg_attention_unet_v13_1.pth"
OUTPUT_DIR      = "/content/drive/MyDrive/Capstone/seg_predictions_v13_1"

PATCH_SIZE           = 256
PATCHES_PER_IMAGE    = 12
LESION_PATCH_RATIO   = 0.30 #Changed this from 0.60. Need to revisit later for tuning
LESION_JITTER        = 64

PHYSICAL_BATCH_SIZE  = 8
EFFECTIVE_BATCH_SIZE = 32
ACCUMULATION_STEPS   = EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE
MAX_EPOCHS           = 128
WARMUP_EPOCHS        = 30
LR_DECODER           = 1e-3
LR_ENCODER           = 5e-5
WEIGHT_DECAY         = 5e-3
EARLY_STOP_PATIENCE  = 20
GRAD_CLIP            = 1.0
FULL_VAL_EVERY       = 5

INFERENCE_STRIDE  = 128
PREDICT_THRESHOLD = 0.45

POS_WEIGHT_CAP = 4.0 #Changed this from 8
DICE_WEIGHT    = 0.8
BCE_WEIGHT     = 0.2

DROPOUT_P    = 0.55
VAL_FRACTION = 0.2
SEED         = 42

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print("=" * 70)
print("ATTENTION U-NET SEGMENTATION v13")
print("=" * 70)
print(f"Device          : {DEVICE}")
print(f"Patch size      : {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Inference stride: {INFERENCE_STRIDE}  (reflect-pad, uniform overlap)")
print(f"Val fraction    : {VAL_FRACTION}  (originals only)")
print(f"Full-val every  : {FULL_VAL_EVERY} epochs")
print(f"Dropout         : {DROPOUT_P}")
print(f"Weight decay    : {WEIGHT_DECAY}")
print(f"Patches/image   : {PATCHES_PER_IMAGE}")
print(f"Lesion ratio    : {LESION_PATCH_RATIO:.0%} lesion / {1-LESION_PATCH_RATIO:.0%} background  (train only)")
print(f"Val lesion ratio: 0%  (unbiased uniform random)")
print(f"Lesion jitter   : {LESION_JITTER}px")
print(f"Threshold       : {PREDICT_THRESHOLD}  (swept on val after training)")
print(f"pos_weight cap  : {POS_WEIGHT_CAP}  (lowered to reduce false-positive bias)")
print(f"Dice/BCE weights: {DICE_WEIGHT}/{BCE_WEIGHT}  (Dice dominant at extreme imbalance)")
print(f"Augmentation    : vflip + color jitter  (no rotation, hflip on disk)")
print()


# ============================================================
# Normalisation (ImageNet stats)
# ============================================================
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225],
)


# ============================================================
# Patch utilities
# ============================================================

def extract_lesion_patch(image_np, mask_np, patch_size, jitter=32):
    h, w          = mask_np.shape
    lesion_coords = np.argwhere(mask_np > 0)
    if len(lesion_coords) == 0:
        return extract_random_patch(image_np, mask_np, patch_size)
    center = lesion_coords[random.randint(0, len(lesion_coords) - 1)]
    cy     = int(center[0]) + random.randint(-jitter, jitter)
    cx     = int(center[1]) + random.randint(-jitter, jitter)
    half   = patch_size // 2
    y1     = max(0, min(cy - half, h - patch_size))
    x1     = max(0, min(cx - half, w - patch_size))
    return (image_np[y1:y1+patch_size, x1:x1+patch_size],
            mask_np [y1:y1+patch_size, x1:x1+patch_size])


def extract_random_patch(image_np, mask_np, patch_size):
    h, w = mask_np.shape
    y1   = random.randint(0, h - patch_size)
    x1   = random.randint(0, w - patch_size)
    return (image_np[y1:y1+patch_size, x1:x1+patch_size],
            mask_np [y1:y1+patch_size, x1:x1+patch_size])


def get_sliding_window_coords(h, w, patch_size, stride):
    coords = []
    y = 0
    while y + patch_size <= h:
        x = 0
        while x + patch_size <= w:
            coords.append((y, x))
            x += stride
        y += stride
    return coords


# ============================================================
# Dataset
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, samples, patch_size=256, patches_per_image=12,
                 jitter=32, augment=False, lesion_ratio=0.7):
        self.unique_pairs    = samples
        self.patch_size      = patch_size
        self.patches_per_img = patches_per_image
        self.jitter          = jitter
        self.augment         = augment
        self.lesion_ratio    = lesion_ratio
        self.color_jitter    = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.unique_pairs) * self.patches_per_img

    def _augment(self, img_pil, mask_pil):
        if random.random() > 0.8:
            img_pil  = TF.vflip(img_pil)
            mask_pil = TF.vflip(mask_pil)
        if random.random() > 0.5:
            img_pil = self.color_jitter(img_pil)
        return img_pil, mask_pil

    def __getitem__(self, idx):
        img_path, mask_path = self.unique_pairs[idx % len(self.unique_pairs)]

        img_np  = np.array(Image.open(img_path).convert("L"))
        img_np  = np.stack([img_np] * 3, axis=-1)
        mask_np = (np.array(Image.open(mask_path).convert("L")) > 127
                   ).astype(np.float32)

        if random.random() < self.lesion_ratio:
            img_p, mask_p = extract_lesion_patch(
                img_np, mask_np, self.patch_size, self.jitter)
        else:
            img_p, mask_p = extract_random_patch(
                img_np, mask_np, self.patch_size)

        img_pil  = Image.fromarray(img_p.astype(np.uint8))
        mask_pil = Image.fromarray((mask_p * 255).astype(np.uint8))

        if self.augment:
            img_pil, mask_pil = self._augment(img_pil, mask_pil)

        img_tensor  = NORMALIZE(transforms.ToTensor()(img_pil))
        mask_tensor = torch.from_numpy(
            (np.array(mask_pil) > 127).astype(np.float32)
        ).unsqueeze(0)

        return img_tensor, mask_tensor, os.path.basename(img_path)


# ============================================================
# Attention Gate
# ============================================================

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True),
                                   nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True),
                                   nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True),
                                   nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode="bilinear", align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


# ============================================================
# Attention U-Net backbone
# ============================================================

class _AttentionUNetBackbone(nn.Module):

    def __init__(self, pretrained=True, dropout_p=0.55):
        super().__init__()
        vgg   = models.vgg11(weights="IMAGENET1K_V1" if pretrained else None)
        feats = list(vgg.features.children())

        self.enc1 = nn.Sequential(*feats[:3])
        self.enc2 = nn.Sequential(*feats[3:6])
        self.enc3 = nn.Sequential(*feats[6:11])
        self.enc4 = nn.Sequential(*feats[11:16])
        self.enc5 = nn.Sequential(*feats[16:21])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_p),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.att5 = AttentionGate(512, 512, 256)
        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64,  64,  32)

        self.up5  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = self._blk(1024, 512, dropout_p)
        self.up4  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = self._blk(1024, 512, dropout_p)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._blk(512,  256, dropout_p)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._blk(256,  128, dropout_p)
        self.up1  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec1 = self._blk(128,  64,  dropout_p)
        self.out  = nn.Conv2d(64, 1, 1)

    def _blk(self, ic, oc, dp):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True), nn.Dropout2d(p=dp),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )

    def _match(self, enc, dec):
        if enc.shape[2:] != dec.shape[2:]:
            enc = F.interpolate(enc, size=dec.shape[2:],
                                mode="bilinear", align_corners=False)
        return enc

    def forward(self, x):
        e1 = self.enc1(x);  e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        b  = self.bottleneck(e5)

        up5_b  = self.up5(b)
        d5     = self.dec5(torch.cat([up5_b,  self.att5(up5_b,  self._match(e5, up5_b))],  1))

        up4_d5 = self.up4(d5)
        d4     = self.dec4(torch.cat([up4_d5, self.att4(up4_d5, self._match(e4, up4_d5))], 1))

        up3_d4 = self.up3(d4)
        d3     = self.dec3(torch.cat([up3_d4, self.att3(up3_d4, self._match(e3, up3_d4))], 1))

        up2_d3 = self.up2(d3)
        d2     = self.dec2(torch.cat([up2_d3, self.att2(up2_d3, self._match(e2, up2_d3))], 1))

        up1_d2 = self.up1(d2)
        d1     = self.dec1(torch.cat([up1_d2, self.att1(up1_d2, self._match(e1, up1_d2))], 1))

        out = self.out(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:],
                                mode="bilinear", align_corners=False)
        return out

    def freeze_encoder(self):
        for blk in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            for p in blk.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        for blk in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            for p in blk.parameters():
                p.requires_grad = True


# ============================================================
# SegmentationModel — public-facing wrapper
# ============================================================

class SegmentationModel(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.55,
                 patch_size=256, stride=128, threshold=0.45):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.threshold  = threshold
        self.net        = _AttentionUNetBackbone(pretrained=pretrained,
                                                 dropout_p=dropout_p)

    def forward(self, x):
        return self.net(x)

    def freeze_encoder(self):
        self.net.freeze_encoder()
        print("  Encoder FROZEN -- training decoder + attention gates only")

    def unfreeze_encoder(self):
        self.net.unfreeze_encoder()
        print("  Encoder UNFROZEN -- full model fine-tuning")

    def _load_image(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("L")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("L")
        elif isinstance(image, Image.Image):
            return image.convert("L")
        raise TypeError(f"Expected str, PIL Image, or ndarray -- got {type(image)}")

    def predict(self, image, tta=True):
        self.eval()
        device  = next(self.parameters()).device
        img_pil = self._load_image(image)

        with torch.no_grad():
            pred_orig = self._sliding_window(np.array(img_pil), device)
            if tta:
                flipped   = np.array(TF.hflip(img_pil))
                pred_flip = self._sliding_window(flipped, device)
                avg_map   = (pred_orig + np.fliplr(pred_flip).copy()) / 2
            else:
                avg_map = pred_orig

        binary = (avg_map > self.threshold).astype(np.uint8) * 255
        return Image.fromarray(binary, mode="L")

    def _sliding_window(self, img_np, device):
        H, W = img_np.shape

        pad_h = (self.stride - H % self.stride) % self.stride
        pad_w = (self.stride - W % self.stride) % self.stride
        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(img_np, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            img_padded = img_np

        pH, pW     = img_padded.shape
        img_3ch    = np.stack([img_padded] * 3, axis=-1)
        pred_sum   = np.zeros((pH, pW), dtype=np.float32)
        pred_count = np.zeros((pH, pW), dtype=np.float32)

        coords = get_sliding_window_coords(pH, pW, self.patch_size, self.stride)

        for y1, x1 in coords:
            y2, x2 = y1 + self.patch_size, x1 + self.patch_size
            patch  = img_3ch[y1:y2, x1:x2].astype(np.uint8)
            t      = NORMALIZE(transforms.ToTensor()(Image.fromarray(patch)))
            logit  = self.net(t.unsqueeze(0).to(device))
            prob   = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_sum  [y1:y2, x1:x2] += prob
            pred_count[y1:y2, x1:x2] += 1.0

        avg = pred_sum / np.maximum(pred_count, 1e-6)
        return np.ascontiguousarray(avg[:H, :W])


# ============================================================
# Loss
# ============================================================

class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss.

    DICE CALCULATION — LOCATION 1 (training loss, soft Dice):
    ---------------------------------------------------------
    Called on every training and validation batch via the `dice_loss` method.

    Inputs:
      logits  — raw model output BEFORE sigmoid, shape (B, 1, H, W)
      targets — binary ground truth mask,        shape (B, 1, H, W), values 0 or 1

    Steps:
      1. p = sigmoid(logits)
            Convert raw logits to soft probabilities in [0, 1].
            Using soft probabilities (not hard 0/1) keeps the loss
            differentiable so gradients can flow back through the network.

      2. Flatten both p and targets to shape (B, N) where N = H * W,
            so each sample in the batch is treated independently.

      3. inter = sum(p * t, dim=1)
            Element-wise product, then sum over all pixels.
            For a hard binary prediction this would equal TP (true positives).
            With soft probabilities it is a weighted count of positive pixels.

      4. denom = sum(p, dim=1) + sum(t, dim=1)
            Sum of predicted positives + sum of ground truth positives.
            For hard binary: denom = (TP + FP) + (TP + FN) = 2TP + FP + FN.

      5. per_sample_dice_loss = 1 - (2 * inter + smooth) / (denom + smooth)
            smooth = 1e-6 prevents division by zero on all-negative patches
            (patches with no lesion pixels at all).
            This is the standard soft Dice loss formula.

      6. Return mean over the batch dimension.

    Combined loss = (BCE_WEIGHT * BCE) + (DICE_WEIGHT * dice_loss)
                  = (0.2 * BCE)        + (0.8 * dice_loss)

    Note: Dice dominates (0.8 weight) because BCE alone struggles at
    extreme class imbalance (<1% foreground pixels per patch).
    """
    def __init__(self, pos_weight=None, dice_weight=0.6, bce_weight=0.4, smooth=1e-6):
        super().__init__()
        self.dice_w = dice_weight
        self.bce_w  = bce_weight
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def dice_loss(self, logits, targets):
        B = logits.size(0)

        # Step 1+2: soft probabilities, flattened to (B, N)
        p = torch.sigmoid(logits).view(B, -1)
        t = targets.view(B, -1)

        # Step 3: soft intersection per sample
        inter = (p * t).sum(dim=1)

        # Step 4: denominator per sample
        denom = p.sum(dim=1) + t.sum(dim=1)

        # Step 5: soft Dice loss per sample, averaged over batch
        per_sample = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        return per_sample.mean()

    def forward(self, logits, targets):
        return (self.bce_w  * self.bce(logits, targets) +
                self.dice_w * self.dice_loss(logits, targets))


# ============================================================
# Metrics
# ============================================================

def iou_score(pred_logits, targets, threshold=PREDICT_THRESHOLD):
    has_lesion = targets.view(targets.size(0), -1).sum(dim=1) > 0
    if has_lesion.sum() == 0:
        return float('nan')
    pred    = (torch.sigmoid(pred_logits[has_lesion]) > threshold).float()
    tgt     = targets[has_lesion]
    i       = (pred * tgt).sum()
    u       = pred.sum() + tgt.sum() - i
    return ((i + 1e-6) / (u + 1e-6)).item()


def full_image_metrics(model, fnames, rad_dir, mask_dir, threshold=None):
    """
    Evaluate full-image IoU, Dice, and pixel accuracy.

    DICE CALCULATION — LOCATION 2 (evaluation metric, hard binary Dice):
    --------------------------------------------------------------------
    Called every FULL_VAL_EVERY epochs on the val set, and once on the
    held-out test set after training is complete.

    Unlike the training loss (Location 1), this uses HARD binary predictions
    from the already-thresholded output mask, not soft probabilities.

    For each image:
      1. pred_bin = predicted mask pixels > 127
            The model's predict() method already applies PREDICT_THRESHOLD
            to the sliding-window probability map and returns a binary PIL
            image (0 or 255). Dividing by 255 and comparing > 127 gives a
            boolean array: True where the model predicts lesion.

      2. gt_bin = ground truth mask pixels > 127
            Same binarisation applied to the ground truth mask file.

      3. inter = sum(pred_bin AND gt_bin)
            Pixel-wise AND counts pixels that are lesion in BOTH prediction
            and ground truth — these are the True Positives (TP).

      4. denom = sum(pred_bin) + sum(gt_bin)
            = (TP + FP) + (TP + FN)
            = 2*TP + FP + FN

      5. if denom == 0:
              dice = 1.0   (both prediction and ground truth are all-zero,
                            meaning a normal image was correctly predicted
                            as normal — perfect score by convention)
         else:
              dice = (2 * inter) / denom
                   = 2*TP / (2*TP + FP + FN)

      6. Accumulate per-image dice, then divide by number of images at the end.
            This is a MACRO average: every image contributes equally,
            regardless of how many lesion pixels it contains.
            A small lesion and a large lesion are weighted the same.

    Key differences from the training loss (Location 1):
      - Hard binary inputs (not soft probabilities)
      - No smoothing term (explicit 0/0 = 1.0 for normal images)
      - Per-image average (not per-patch-in-batch)
      - Not differentiable — used only for monitoring, not for weight updates

    Pass threshold=<float> to temporarily override model.threshold for the
    threshold sweep without permanently changing the model's state.
    """
    model.eval()
    orig_thresh = model.threshold
    if threshold is not None:
        model.threshold = threshold

    total_iou = total_dice = total_acc = 0.0
    for fname in fnames:
        pred_pil = model.predict(os.path.join(rad_dir, fname), tta=False)

        # Step 1: hard binary prediction
        pred_bin = (np.array(pred_pil) > 127).astype(np.uint8)

        # Step 2: hard binary ground truth
        gt_bin   = (np.array(Image.open(
            os.path.join(mask_dir, fname)).convert("L")) > 127).astype(np.uint8)

        # IoU (Jaccard index): TP / (TP + FP + FN)
        inter = (pred_bin & gt_bin).sum()
        union = (pred_bin | gt_bin).sum()
        total_iou += (inter + 1e-6) / (union + 1e-6)

        # Steps 3–5: hard binary Dice
        denom = int(pred_bin.sum()) + int(gt_bin.sum())
        if denom == 0:
            # Both pred and gt are all-zero: normal image, perfectly predicted
            total_dice += 1.0
        else:
            total_dice += (2 * inter) / denom   # 2*TP / (2*TP + FP + FN)

        # Pixel accuracy: fraction of pixels correctly classified
        total_acc += (pred_bin == gt_bin).mean()
        torch.cuda.empty_cache()

    model.threshold = orig_thresh   # restore
    n = len(fnames)
    return {"iou": total_iou/n, "dice": total_dice/n, "pixel_acc": total_acc/n}


# ============================================================
# Threshold sweep — run on val set after training
# ============================================================

def sweep_threshold(model, val_fnames, rad_dir, mask_dir,
                    candidates=None):
    """
    Evaluate IoU at each candidate threshold on the val set and return the
    best threshold. Runs with tta=False for speed. Does NOT mutate model.threshold.
    The Dice calculated here uses the hard binary formula from Location 2 above,
    evaluated at each candidate threshold via full_image_metrics.
    """
    if candidates is None:
        candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("\n  --- Threshold sweep on val set ---")
    print(f"  {'Threshold':>10}  {'Val IoU':>10}  {'Val Dice':>10}")
    best_thresh, best_iou = candidates[0], -1.0
    for t in candidates:
        fm = full_image_metrics(model, val_fnames, rad_dir, mask_dir, threshold=t)
        marker = " ←" if fm["iou"] > best_iou else ""
        print(f"  {t:>10.2f}  {fm['iou']:>10.4f}  {fm['dice']:>10.4f}{marker}")
        if fm["iou"] > best_iou:
            best_iou    = fm["iou"]
            best_thresh = t
    print(f"  Best threshold: {best_thresh}  (val IoU={best_iou:.4f})")
    return best_thresh, best_iou


# ============================================================
# Train / Validate
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device,
                accum_steps, grad_clip):
    model.train()
    total_loss = total_iou = 0.0
    iou_count  = 0
    optimizer.zero_grad()
    for i, (imgs, masks, _) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        out  = model(imgs)
        loss = criterion(out, masks) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
        score = iou_score(out.detach(), masks)
        if not (score != score):
            total_iou += score
            iou_count += 1
        del imgs, masks, out, loss
        if i % 4 == 0:
            torch.cuda.empty_cache()
    avg_iou = total_iou / iou_count if iou_count > 0 else float('nan')
    return total_loss / len(loader), avg_iou


def patch_validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = 0.0
    iou_count  = 0
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out          = model(imgs)
            total_loss  += criterion(out, masks).item()
            score        = iou_score(out, masks)
            if not (score != score):
                total_iou += score
                iou_count += 1
            del imgs, masks, out
    avg_iou = total_iou / iou_count if iou_count > 0 else float('nan')
    return total_loss / len(loader), avg_iou


def compute_pos_weight(train_pairs, n_samples=500, patch_size=256,
                       lesion_ratio=0.6, cap=8.0):
    total_pos = total_neg = 0
    dummy_img = np.zeros((1, 1), dtype=np.float32)

    for _ in range(n_samples):
        _, mask_path = random.choice(train_pairs)
        mask_np = (np.array(Image.open(mask_path).convert("L")) > 127
                   ).astype(np.float32)
        h, w = mask_np.shape
        if h < patch_size or w < patch_size:
            continue
        if random.random() < lesion_ratio:
            _, patch = extract_lesion_patch(dummy_img, mask_np, patch_size)
        else:
            _, patch = extract_random_patch(dummy_img, mask_np, patch_size)
        total_pos += patch.sum()
        total_neg += (1.0 - patch).sum()

    raw = total_neg / max(total_pos, 1)
    return raw, min(raw, cap)


# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("LOADING SEGMENTATION DATA")
print("=" * 70)

def _numeric_key(fname):
    stem   = os.path.splitext(fname)[0]
    digits = ''.join(filter(str.isdigit, stem))
    return int(digits) if digits else stem

orig_fnames = sorted(
    (f for f in os.listdir(TRAIN_RAD)
     if f.upper().endswith(".JPG")
     and f in set(os.listdir(TRAIN_MASK))),
    key=_numeric_key
)

test_fnames = sorted(
    (f for f in os.listdir(TEST_RAD)
     if f.upper().endswith(".JPG")
     and f in set(os.listdir(TEST_MASK))),
    key=_numeric_key
)

train_orig, val_fnames = train_test_split(
    orig_fnames, test_size=VAL_FRACTION, random_state=SEED)

train_flip_fnames = []
for f in train_orig:
    stem, ext = os.path.splitext(f)
    flipped   = f"{stem}_flip{ext}"
    if (os.path.exists(os.path.join(TRAIN_RAD_FLIP,  flipped)) and
            os.path.exists(os.path.join(TRAIN_MASK_FLIP, flipped))):
        train_flip_fnames.append(flipped)

train_pairs = (
    [(os.path.join(TRAIN_RAD,      f), os.path.join(TRAIN_MASK,      f)) for f in train_orig]
  + [(os.path.join(TRAIN_RAD_FLIP, f), os.path.join(TRAIN_MASK_FLIP, f)) for f in train_flip_fnames]
)
val_pairs = [
    (os.path.join(TRAIN_RAD, f), os.path.join(TRAIN_MASK, f)) for f in val_fnames
]

print(f"  Original train  : {len(train_orig)} images")
print(f"  Flipped train   : {len(train_flip_fnames)} images")
print(f"  Total train     : {len(train_pairs)} images")
print(f"  Val             : {len(val_pairs)} images  (originals only)")
print(f"  Test            : {len(test_fnames)} images  (held out)")
print()


# ============================================================
# Training
# ============================================================
print("=" * 70)
print("ATTENTION U-NET v13  --  single run")
print("=" * 70)
print(f"Patch size      : {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Patches/image   : {PATCHES_PER_IMAGE}")
print(f"Lesion ratio    : {LESION_PATCH_RATIO:.0%} lesion / {1-LESION_PATCH_RATIO:.0%} background")
print(f"Lesion jitter   : {LESION_JITTER}px")
print(f"Loss            : DiceBCE per-sample  (dice={DICE_WEIGHT}, bce={BCE_WEIGHT})")
print(f"Threshold       : {PREDICT_THRESHOLD}  (swept on val after training)")
print(f"pos_weight cap  : {POS_WEIGHT_CAP}")
print(f"LR              : decoder={LR_DECODER}  encoder={LR_ENCODER}")
print(f"Weight decay    : {WEIGHT_DECAY}")
print(f"Dropout         : {DROPOUT_P}")
print(f"Warmup          : encoder frozen for {WARMUP_EPOCHS} epochs")
print(f"Max epochs      : {MAX_EPOCHS}")
print(f"Early stop      : {EARLY_STOP_PATIENCE} full-val checks")
print(f"Test set        : {len(test_fnames)} images  <- held out until the end")
print(f"Inference       : reflect-padded sliding window -> full mask")
print("=" * 70)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

train_ds = PatchDataset(train_pairs, PATCH_SIZE, PATCHES_PER_IMAGE,
                        LESION_JITTER, augment=True,  lesion_ratio=LESION_PATCH_RATIO)
val_ds   = PatchDataset(val_pairs,   PATCH_SIZE, PATCHES_PER_IMAGE,
                        LESION_JITTER, augment=False, lesion_ratio=0.0)

train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=PHYSICAL_BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

raw_pw, pw_val = compute_pos_weight(train_pairs, n_samples=500,
                                    patch_size=PATCH_SIZE,
                                    lesion_ratio=LESION_PATCH_RATIO,
                                    cap=POS_WEIGHT_CAP)
pw_tensor = torch.tensor([pw_val], dtype=torch.float32).to(DEVICE)
print(f"  pos_weight = {pw_val:.2f}  (raw={raw_pw:.1f}, cap={POS_WEIGHT_CAP})")
criterion = DiceBCELoss(pw_tensor, DICE_WEIGHT, BCE_WEIGHT)

model = SegmentationModel(
    pretrained=True, dropout_p=DROPOUT_P,
    patch_size=PATCH_SIZE, stride=INFERENCE_STRIDE,
    threshold=PREDICT_THRESHOLD,
).to(DEVICE)

model.freeze_encoder()

decoder_params = (
    list(model.net.bottleneck.parameters())
    + list(model.net.att5.parameters()) + list(model.net.up5.parameters()) + list(model.net.dec5.parameters())
    + list(model.net.att4.parameters()) + list(model.net.up4.parameters()) + list(model.net.dec4.parameters())
    + list(model.net.att3.parameters()) + list(model.net.up3.parameters()) + list(model.net.dec3.parameters())
    + list(model.net.att2.parameters()) + list(model.net.up2.parameters()) + list(model.net.dec2.parameters())
    + list(model.net.att1.parameters()) + list(model.net.up1.parameters()) + list(model.net.dec1.parameters())
    + list(model.net.out.parameters())
)
optimizer = optim.Adam(decoder_params, lr=LR_DECODER, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

best_full_val_iou = 0.0
patience_ctr      = 0
encoder_unfrozen  = False
start_time        = time.time()

_hdr = (f"{'Ep':>5} | {'Phase':>8} | {'TLoss':>8} | {'TIoU':>7} | "
        f"{'VLoss':>8} | {'VIoU':>7} | {'Time':>6}")
print(_hdr)
print("-" * len(_hdr))

for epoch in range(1, MAX_EPOCHS + 1):

    if epoch == WARMUP_EPOCHS + 1 and not encoder_unfrozen:
        model.unfreeze_encoder()
        encoder_unfrozen  = True
        patience_ctr      = 0
        best_full_val_iou = 0.0

        enc_params = (
            list(model.net.enc1.parameters()) + list(model.net.enc2.parameters())
            + list(model.net.enc3.parameters()) + list(model.net.enc4.parameters())
            + list(model.net.enc5.parameters())
        )
        optimizer.add_param_group({
            "params": enc_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY
        })
        optimizer.param_groups[0]["lr"] = LR_DECODER / 10
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    phase = "warmup" if not encoder_unfrozen else "finetune"

    tr_loss, tr_iou = train_epoch(model, train_loader, criterion,
                                  optimizer, DEVICE, ACCUMULATION_STEPS, GRAD_CLIP)
    va_loss, va_iou = patch_validate(model, val_loader, criterion, DEVICE)
    scheduler.step()
    torch.cuda.empty_cache()

    elapsed     = (time.time() - start_time) / 60
    tr_iou_str  = f"{tr_iou:7.4f}" if tr_iou == tr_iou else "    nan"
    va_iou_str  = f"{va_iou:7.4f}" if va_iou == va_iou else "    nan"
    print(f"{epoch:5d} | {phase:>8} | {tr_loss:8.4f} | {tr_iou_str} | "
          f"{va_loss:8.4f} | {va_iou_str} | {elapsed:5.1f}m")

    if epoch % FULL_VAL_EVERY == 0 or epoch == 1:
        fm   = full_image_metrics(model, val_fnames, TRAIN_RAD, TRAIN_MASK)
        mark = "★" if fm["iou"] > best_full_val_iou else " "
        print(f"        [Full] IoU={fm['iou']:.4f}  "
              f"Dice={fm['dice']:.4f}  PixAcc={fm['pixel_acc']:.4f}  {mark}")

        if fm["iou"] > best_full_val_iou:
            best_full_val_iou = fm["iou"]
            patience_ctr      = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "full_val_iou":     fm["iou"],
                "full_val_dice":    fm["dice"],
                "patch_size":       PATCH_SIZE,
                "stride":           INFERENCE_STRIDE,
                "threshold":        PREDICT_THRESHOLD,
                "dropout_p":        DROPOUT_P,
            }, MODEL_SAVE_PATH)
        else:
            patience_ctr += 1

        if patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping (best full-val IoU={best_full_val_iou:.4f})")
            break

print("-" * len(_hdr))


# ============================================================
# Threshold sweep on val set
# ============================================================
ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

best_thresh, _ = sweep_threshold(
    model, val_fnames, TRAIN_RAD, TRAIN_MASK,
    candidates=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
)
model.threshold = best_thresh

# Persist best threshold back into checkpoint
ckpt["threshold"] = best_thresh
torch.save(ckpt, MODEL_SAVE_PATH)
print(f"  Checkpoint updated: threshold set to {best_thresh}")


# ============================================================
# Final evaluation on held-out test set
# ============================================================
model.eval()
test_m = full_image_metrics(model, test_fnames, TEST_RAD, TEST_MASK)
print(f"\nTEST RESULTS (best val epoch={ckpt['epoch']}, threshold={best_thresh})")
print(f"  IoU      : {test_m['iou']:.4f}")
print(f"  Dice     : {test_m['dice']:.4f}")   # hard binary Dice — see Location 2 above
print(f"  Pix Acc  : {test_m['pixel_acc']:.4f}")
print(f"  Val IoU  : {best_full_val_iou:.4f}")
print(f"  Saved to : {MODEL_SAVE_PATH}")


# ============================================================
# Save test predictions
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nGenerating test predictions -> {OUTPUT_DIR}")
for fname in test_fnames:
    pred = model.predict(os.path.join(TEST_RAD, fname), tta=True)
    pred.save(os.path.join(OUTPUT_DIR, fname))
    torch.cuda.empty_cache()
print(f"  Saved {len(test_fnames)} full-resolution masks")


# ============================================================
# Standalone Inference
# ============================================================
# ckpt  = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
# model = SegmentationModel(
#     pretrained=False, dropout_p=ckpt["dropout_p"],
#     patch_size=ckpt["patch_size"], stride=ckpt["stride"],
#     threshold=ckpt["threshold"],   # ← best threshold from val sweep
# ).to(DEVICE)
# model.load_state_dict(ckpt["model_state_dict"])
# mask = model.predict("/path/to/radiograph.jpg", tta=True)
# mask.save("/path/to/output_mask.jpg")
print("\nDone.")
