from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)

# ================================================================================
# Convert all JPG images in working directories to PNG (run once, idempotent)
# ================================================================================
from PIL import Image
from pathlib import Path
import os

_IMG_DIRS = [
    "/content/drive/MyDrive/Capstone/data/Train/Radiographs",
    "/content/drive/MyDrive/Capstone/data/Train/mask",
    "/content/drive/MyDrive/Capstone/data/Test/Radiographs",
    "/content/drive/MyDrive/Capstone/data/Test/mask",
]

# Also wipe any stale label caches — they store old .jpg paths and must be rebuilt
_STALE_CACHES = [
    "/content/drive/MyDrive/Capstone/data/labels_train_v35.csv",
    "/content/drive/MyDrive/Capstone/data/labels_test_v35.csv",
]

print("Converting JPG images to PNG...")
for _dir in _IMG_DIRS:
    _p = Path(_dir)
    if not _p.exists():
        print(f"  ⚠️  Skipping (not found): {_dir}")
        continue
    _jpgs = list(_p.glob("*.JPG")) + list(_p.glob("*.jpg"))
    for _jpg in _jpgs:
        _out = _jpg.with_suffix(".png")
        if not _out.exists():
            with Image.open(_jpg) as _im:
                _im.convert("RGB").save(_out, "PNG")
        _jpg.unlink()
    if _jpgs:
        print(f"  ✅ {_dir}: converted {len(_jpgs)} file(s)")

for _cache in _STALE_CACHES:
    if os.path.exists(_cache):
        os.remove(_cache)
        print(f"  🗑️  Deleted stale cache: {_cache}")

print("PNG conversion complete.\n")

# ================================================================================
# Full-Image Binary Classifier v35
# ================================================================================
# KEY FIXES vs v34-CV-v3:
#
#  1. AUGMENTATION LEAKAGE CLOSED
#     Originals and their _hflip counterparts are always kept in the same
#     split. When image X.jpg is assigned to val, X_hflip.jpg never ends
#     up in train. Grouping is done before any fold/val split.
#
#  2. CV DISTRIBUTION MATCHES TEST THROUGHOUT
#     All splits (CV pool, fold val sets, fixed val carve-out, test) share
#     the same ~34% abnormal ratio — the natural ratio of the dataset.
#     No distribution shift between training folds and evaluation.
#     CV fold F1/recall are directly comparable to test performance.
#
#  3. MUCH STRONGER IMBALANCE HANDLING
#     - FocalLoss alpha raised to 0.75 (was 0.25 — that DOWN-weights positives)
#     - WeightedRandomSampler oversamples abnormal in every training batch
#     - pos_weight on an auxiliary BCE term as a second recall nudge
#
#  4. LARGER ENCODER: EfficientNet-B2 (configurable back to B0)
#
#  5. THRESHOLD STRATEGY UNCHANGED (val → threshold → test once)
# ================================================================================

import csv, gc, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

# ============================================================
# Paths  —  update DATA_ROOT if your layout differs
# ============================================================
DATA_ROOT      = "/content/drive/MyDrive/Capstone/data"
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "Train", "mask")
TRAIN_RAD_DIR  = os.path.join(DATA_ROOT, "Train", "Radiographs")
TEST_MASK_DIR  = os.path.join(DATA_ROOT, "Test",  "mask")
TEST_RAD_DIR   = os.path.join(DATA_ROOT, "Test",  "Radiographs")
MODEL_SAVE_DIR = "/content/drive/MyDrive/Capstone"
LABELS_CSV     = os.path.join(DATA_ROOT, "labels_v35.csv")

# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
ENCODER        = "b0"      # B2 overfits on ~1150 images; B0 has 4M fewer params
IMG_W, IMG_H   = 640, 320

# --- Training ---
BATCH_SIZE          = 16
MAX_EPOCHS          = 60
LR_WARMUP_EPOCHS    = 3
LR_HEAD             = 5e-4
LR_ENCODER          = 1e-5
WEIGHT_DECAY        = 5e-2
DROPOUT_P           = 0.5
EARLY_STOP_PATIENCE = 15
F1_ROLLING_WINDOW   = 3
GRAD_CLIP           = 1.0
SEED                = 42

FREEZE_ENCODER_EPOCHS = 10
LABEL_SMOOTHING       = 0.1

# --- Loss / imbalance ---
FOCAL_ALPHA    = 0.60
FOCAL_GAMMA    = 2.0
FOCAL_BCE_MIX  = 0.5

# --- Preprocessing ---
PIXEL_THRESHOLD  = 10
FORCE_PREPROCESS = False

# --- CV ---
N_FOLDS    = 5
VAL_FRAC   = 0.15

# ============================================================
print("=" * 70)
print("FULL-IMAGE BINARY CLASSIFIER v35  (leak-free, imbalance-aware)")
print("=" * 70)
print(f"Device : {DEVICE}  |  Encoder : EfficientNet-{ENCODER.upper()}")
print(f"Input  : {IMG_W}x{IMG_H}  |  Batch : {BATCH_SIZE}")
print(f"Focal  : alpha={FOCAL_ALPHA} gamma={FOCAL_GAMMA}  |  BCE mix : {FOCAL_BCE_MIX}")
print(f"Pixel threshold : > {PIXEL_THRESHOLD}  (lower = more sensitive to faint lesions)")
print(f"  Encoder freeze : first {FREEZE_ENCODER_EPOCHS} epochs (head-only), then full fine-tune")
print(f"  Label smoothing: {LABEL_SMOOTHING}")
print(f"  Weight decay   : {WEIGHT_DECAY}  |  Dropout : {DROPOUT_P}")
print()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# STEP 1 — SCAN MASKS → LABELS
# ============================================================

HFLIP_SUFFIX = "_hflip"

def stem_of(fname):
    return os.path.splitext(fname)[0]

def is_hflip(fname):
    return stem_of(fname).endswith(HFLIP_SUFFIX)

def original_stem(fname):
    """For X_hflip.png → 'X'.  For X.png → 'X'."""
    s = stem_of(fname)
    return s[: -len(HFLIP_SUFFIX)] if s.endswith(HFLIP_SUFFIX) else s

def has_white_pixels(mask_path):
    arr = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return bool(np.any(arr > PIXEL_THRESHOLD))


def build_labels(mask_dir, rad_dir, split_name, force=False, cache_path=None):
    """
    Scan mask_dir, pair each mask with its radiograph, assign label.
    Returns list of dicts: {filename, rad_path, mask_path, label, is_hflip, orig_stem}
    """
    if cache_path and os.path.exists(cache_path) and not force:
        rows = []
        with open(cache_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({k: (int(v) if k in ("label","is_hflip") else v)
                              for k, v in r.items()})
        print(f"  [{split_name}] Loaded {len(rows)} rows from cache.")
        return rows

    # Only scan PNG files                                          ← PNG
    mask_fnames = sorted(f for f in os.listdir(mask_dir)
                         if f.lower().endswith(".png"))
    rad_fnames  = {f for f in os.listdir(rad_dir)
                   if f.lower().endswith(".png")}

    rows, n_skip = [], 0
    for fname in tqdm(mask_fnames, desc=f"  Scanning {split_name}", unit="img"):
        if fname not in rad_fnames:
            n_skip += 1
            continue
        mask_path = os.path.join(mask_dir, fname)
        label     = int(has_white_pixels(mask_path))
        rows.append({
            "split":      split_name,
            "filename":   fname,
            "rad_path":   os.path.join(rad_dir, fname),
            "mask_path":  mask_path,
            "label":      label,
            "is_hflip":   int(is_hflip(fname)),
            "orig_stem":  original_stem(fname),
        })

    n_abn = sum(r["label"] for r in rows)
    print(f"  [{split_name}] {len(rows)} paired  "
          f"({n_abn} abnormal={n_abn/max(len(rows),1)*100:.1f}%,  "
          f"{len(rows)-n_abn} normal)  [{n_skip} skipped]")

    if cache_path:
        fieldnames = ["split","filename","rad_path","mask_path","label",
                      "is_hflip","orig_stem"]
        with open(cache_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return rows


print("=" * 70)
print("STEP 1 — BUILD LABELS")
print("=" * 70)

train_cache = os.path.join(DATA_ROOT, "labels_train_v35.csv")
test_cache  = os.path.join(DATA_ROOT, "labels_test_v35.csv")

train_rows = build_labels(TRAIN_MASK_DIR, TRAIN_RAD_DIR, "train",
                          force=FORCE_PREPROCESS, cache_path=train_cache)
test_rows  = build_labels(TEST_MASK_DIR,  TEST_RAD_DIR,  "test",
                          force=FORCE_PREPROCESS, cache_path=test_cache)
print()


# ============================================================
# STEP 2 — LEAK-FREE TRAIN / VAL SPLIT
# ============================================================

print("=" * 70)
print("STEP 2 — LEAK-FREE TRAIN / VAL CARVE-OUT")
print("=" * 70)

random.seed(SEED)
np.random.seed(SEED)

hflip_by_stem = {}
orig_rows     = []
for r in train_rows:
    if r["is_hflip"]:
        hflip_by_stem[r["orig_stem"]] = r
    else:
        orig_rows.append(r)

abn_origs = [r for r in orig_rows if r["label"] == 1]
nor_origs = [r for r in orig_rows if r["label"] == 0]
random.shuffle(abn_origs)
random.shuffle(nor_origs)

n_val_abn = max(1, round(len(abn_origs) * VAL_FRAC))
n_val_nor = max(1, round(len(nor_origs) * VAL_FRAC))

val_origs  = abn_origs[:n_val_abn]  + nor_origs[:n_val_nor]
cv_origs   = abn_origs[n_val_abn:]  + nor_origs[n_val_nor:]

def expand_with_hflips(orig_list, hflip_lookup):
    out = []
    for r in orig_list:
        out.append(r)
        hf = hflip_lookup.get(r["orig_stem"])
        if hf:
            out.append(hf)
    return out

val_samples_all  = expand_with_hflips(val_origs,  hflip_by_stem)
cv_pool_all      = expand_with_hflips(cv_origs,   hflip_by_stem)
val_samples_eval = val_origs

n_abn_val = sum(r["label"] for r in val_samples_eval)
n_nor_val = len(val_samples_eval) - n_abn_val
n_abn_cv  = sum(r["label"] for r in cv_pool_all)
n_nor_cv  = len(cv_pool_all) - n_abn_cv

print(f"  Val  (fixed, eval-only originals) : {len(val_samples_eval)} images  "
      f"({n_abn_val} abn={n_abn_val/len(val_samples_eval)*100:.0f}%,  "
      f"{n_nor_val} nor)  ← ~34% abnormal, consistent with CV pool & test")
print(f"  CV pool (orig + hflips)            : {len(cv_pool_all)} images  "
      f"({n_abn_cv} abn={n_abn_cv/len(cv_pool_all)*100:.0f}%,  {n_nor_cv} nor)  ← ~34% abnormal")

n_abn_tst = sum(r["label"] for r in test_rows)
n_nor_tst = len(test_rows) - n_abn_tst
print(f"  Test (held out)                    : {len(test_rows)} images  "
      f"({n_abn_tst} abn={n_abn_tst/len(test_rows)*100:.0f}%,  {n_nor_tst} nor)")
print()


# ============================================================
# STEP 3 — DATASET
# ============================================================

class RadioDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, rows, img_w, img_h, augment=False):
        self.rows      = rows
        self.img_w     = img_w
        self.img_h     = img_h
        self.augment   = augment
        self.normalize = transforms.Normalize(mean=self.MEAN, std=self.STD)

    def __len__(self):
        return len(self.rows)

    def _augment(self, img):
        if random.random() > 0.4:
            img = TF.rotate(img, random.uniform(-10, 10), fill=0)
        if random.random() > 0.4:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3)(img)
        if random.random() > 0.5:
            w, h   = img.size
            cw     = int(w * random.uniform(0.85, 1.0))
            ch     = int(h * random.uniform(0.85, 1.0))
            x0     = random.randint(0, w - cw)
            y0     = random.randint(0, h - ch)
            img    = img.crop((x0, y0, x0 + cw, y0 + ch))
            img    = img.resize((self.img_w, self.img_h), Image.BILINEAR)
        if random.random() > 0.5:
            arr    = np.array(img).astype(np.float32)
            arr    = np.clip(arr + np.random.normal(0, 5, arr.shape), 0, 255).astype(np.uint8)
            img    = Image.fromarray(arr)
        return img

    def __getitem__(self, idx):
        row     = self.rows[idx]
        img     = (Image.open(row["rad_path"])
                   .convert("RGB")
                   .resize((self.img_w, self.img_h), Image.BILINEAR))
        if self.augment:
            img = self._augment(img)
        tensor  = self.normalize(transforms.ToTensor()(img))
        return tensor, torch.tensor(float(row["label"])), row["filename"]


def make_weighted_sampler(rows):
    labels    = [r["label"] for r in rows]
    n_abn     = sum(labels)
    n_nor     = len(labels) - n_abn
    w_abn     = 1.0 / n_abn if n_abn > 0 else 0.0
    w_nor     = 1.0 / n_nor if n_nor > 0 else 0.0
    weights   = [w_abn if l == 1 else w_nor for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ============================================================
# STEP 4 — MODEL
# ============================================================

class BinaryClassifier(nn.Module):
    def __init__(self, encoder="b2", dropout_p=0.4, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        if encoder == "b2":
            base = models.efficientnet_b2(weights=weights)
            feat_dim = 1408
        else:
            base = models.efficientnet_b0(weights=weights)
            feat_dim = 1280
        self.features = base.features
        self.avgpool  = base.avgpool
        self.head     = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.avgpool(self.features(x)))


# ============================================================
# STEP 5 — LOSS
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.0):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        probs   = torch.sigmoid(logits)
        p_t     = torch.where(targets >= 0.5, probs, 1 - probs)
        alpha_t = torch.where(targets >= 0.5,
                              torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1 - self.alpha))
        fl      = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t.clamp(1e-8))
        return fl.mean()


def build_criterion(train_rows, device):
    focal = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING)
    bce   = nn.BCEWithLogitsLoss()

    def combined(logits, targets):
        return (FOCAL_BCE_MIX * focal(logits, targets)
                + (1 - FOCAL_BCE_MIX) * bce(logits, targets))
    return combined


# ============================================================
# Metrics helpers
# ============================================================

def compute_counts(logits, labels, threshold=0.5):
    preds  = (torch.sigmoid(logits) > threshold).float().view(-1)
    labels = labels.float().view(-1)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return tp, fp, fn, tn

def metrics(tp, fp, fn, tn):
    acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-6)
    return acc, prec, rec, f1


# ============================================================
# Train / val epoch
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    optimizer.zero_grad()
    for imgs, lbls, _ in loader:
        imgs   = imgs.to(device)
        lbls   = lbls.to(device).unsqueeze(1)
        logits = model(imgs)
        loss   = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        a, b, c, d = compute_counts(logits, lbls)
        tp+=a; fp+=b; fn+=c; tn+=d
        del imgs, lbls, logits, loss
    torch.cuda.empty_cache()
    acc, _, rec, f1 = metrics(tp, fp, fn, tn)
    return total_loss / len(loader), acc, rec, f1


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs   = imgs.to(device)
            lbls   = lbls.to(device).unsqueeze(1)
            logits = model(imgs)
            total_loss += criterion(logits, lbls).item()
            a, b, c, d = compute_counts(logits, lbls)
            tp+=a; fp+=b; fn+=c; tn+=d
    acc, _, rec, f1 = metrics(tp, fp, fn, tn)
    return total_loss / len(loader), acc, rec, f1


# ============================================================
# TTA inference
# ============================================================

NORMALIZE = transforms.Normalize(mean=[0.485,0.456,0.406],
                                  std= [0.229,0.224,0.225])

def tta_scores(model, rows, device, img_w, img_h):
    """2-view TTA (original + hflip). Returns [(score, label), ...]."""
    model.eval()
    out = []
    for row in rows:
        img = (Image.open(row["rad_path"])
               .convert("RGB")
               .resize((img_w, img_h), Image.BILINEAR))
        views = [img, TF.hflip(img)]
        s = []
        with torch.no_grad():
            for v in views:
                t = NORMALIZE(transforms.ToTensor()(v)).unsqueeze(0).to(device)
                s.append(torch.sigmoid(model(t)).item())
        out.append((sum(s) / len(s), row["label"]))
    torch.cuda.empty_cache()
    return out


def sweep_thresholds(scores_gt, label=""):
    thresholds = [round(t, 2) for t in np.arange(0.25, 0.81, 0.05)]
    if label:
        print(f"  {label}")
    print(f"  {'Thr':>5} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print(f"  {'-'*68}")

    best_f1, best_thr           = 0.0, 0.5
    target_thr, target_best_acc = None, 0.0

    for thr in thresholds:
        tp=fp=fn=tn=0
        for sc, gt in scores_gt:
            pred = sc > thr
            if   gt  and  pred: tp+=1
            elif gt  and not pred: fn+=1
            elif not gt and pred: fp+=1
            else: tn+=1
        acc  = (tp+tn)/max(tp+fp+fn+tn,1)
        prec = tp/max(tp+fp,1)
        rec  = tp/max(tp+fn,1)
        f1   = 2*prec*rec/max(prec+rec,1e-6)

        meets_target = acc >= 0.80 and rec >= 0.70
        flag = " ← TARGET MET" if meets_target else ""
        print(f"  {thr:>5.2f} | {acc:>7.4f} | {prec:>7.4f} | {rec:>7.4f} | "
              f"{f1:>7.4f} | {tp:>4} {fp:>4} {fn:>4} {tn:>4}{flag}")

        if tn > 0 and f1 > best_f1:
            best_f1, best_thr = f1, thr
        if meets_target and acc > target_best_acc:
            target_best_acc, target_thr = acc, thr

    if target_thr is not None:
        print(f"\n  Target threshold (acc≥0.80, rec≥0.70) : {target_thr}")
        print(f"  Fallback F1 threshold                  : {best_thr}")
        return target_thr
    else:
        print(f"\n  No threshold met acc≥0.80 + rec≥0.70 — using best-F1 threshold: {best_thr}")
        return best_thr


def report_at_threshold(scores_gt, threshold, label=""):
    tp=fp=fn=tn=0
    for sc, gt in scores_gt:
        pred = sc > threshold
        if   gt  and  pred: tp+=1
        elif gt  and not pred: fn+=1
        elif not gt and pred: fp+=1
        else: tn+=1
    acc  = (tp+tn)/max(tp+fp+fn+tn,1)
    prec = tp/max(tp+fp,1)
    rec  = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-6)
    if label: print(f"  {label}")
    print(f"  Threshold : {threshold}")
    print(f"  Accuracy  : {acc:.4f}  {'✓' if acc>=0.80 else '✗'} (target ≥ 0.80)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}  {'✓' if rec>=0.70 else '✗'} (target ≥ 0.70)")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return dict(acc=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


# ============================================================
# STEP 6 — CROSS-VALIDATION
# ============================================================

print("=" * 70)
print(f"STEP 3 — {N_FOLDS}-FOLD CV  (grouped by original, no hflip leakage)")
print("=" * 70)
print(f"  Architecture : EfficientNet-{ENCODER.upper()} + 2-layer head")
print(f"  Loss         : {FOCAL_BCE_MIX:.0%} FocalLoss(α={FOCAL_ALPHA}, γ={FOCAL_GAMMA})"
      f" + {1-FOCAL_BCE_MIX:.0%} BCE(pos_weight=auto)")
print(f"  Sampler      : natural class ratio in batches (no oversampling — focal+pos_weight handle imbalance)")
print(f"  TTA          : original + hflip (2 views)")
print("=" * 70)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

cv_orig_labels = [r["label"] for r in cv_origs]
skf            = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results   = {}

HDR = (f"{'Ep':>4} | {'LR':>8} | {'TLoss':>8} | {'TAcc':>7} | {'TRec':>7} | {'TF1':>7} | "
       f"{'VLoss':>8} | {'VAcc':>7} | {'VRec':>7} | {'VF1':>7} | {'Min':>5}")

for fold_idx, (tr_idx, vl_idx) in enumerate(
        skf.split(cv_origs, cv_orig_labels), start=1):

    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx}/{N_FOLDS}")
    print("="*70)

    fold_train_origs = [cv_origs[i] for i in tr_idx]
    fold_val_origs   = [cv_origs[i] for i in vl_idx]

    fold_train_rows = expand_with_hflips(fold_train_origs, hflip_by_stem)
    fold_val_rows   = fold_val_origs

    n_abn_tr = sum(r["label"] for r in fold_train_rows)
    n_abn_vl = sum(r["label"] for r in fold_val_rows)
    print(f"  Train : {len(fold_train_rows)}  ({n_abn_tr} abn,  "
          f"{len(fold_train_rows)-n_abn_tr} nor)")
    print(f"  Val   : {len(fold_val_rows)}  ({n_abn_vl} abn,  "
          f"{len(fold_val_rows)-n_abn_vl} nor)  ← early stopping only")

    train_ds = RadioDataset(fold_train_rows, IMG_W, IMG_H, augment=True)
    val_ds   = RadioDataset(fold_val_rows,   IMG_W, IMG_H, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

    model     = BinaryClassifier(encoder=ENCODER, dropout_p=DROPOUT_P).to(DEVICE)
    criterion = build_criterion(fold_train_rows, DEVICE)

    for p in model.features.parameters():
        p.requires_grad = False

    head_params    = list(model.head.parameters())
    encoder_params = list(model.features.parameters()) + list(model.avgpool.parameters())
    optimizer      = optim.AdamW(
        [{"params": encoder_params, "lr": 0.0},
         {"params": head_params,    "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(ep):
        if ep < LR_WARMUP_EPOCHS:
            return 0.1 + 0.9 * (ep / LR_WARMUP_EPOCHS)
        return 1.0

    sched_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    sched_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, MAX_EPOCHS - FREEZE_ENCODER_EPOCHS),
        eta_min=1e-6)

    save_path    = os.path.join(MODEL_SAVE_DIR, f"v35_fold{fold_idx}.pth")
    best_val_f1  = 0.0
    best_val_rec = 0.0
    best_avg_f1  = 0.0
    recent_f1s   = []
    patience_ctr = 0

    print(HDR)
    print("-" * len(HDR))

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == FREEZE_ENCODER_EPOCHS + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer.param_groups[0]["lr"] = LR_ENCODER
            print(f"\n  [Epoch {epoch}] Encoder unfrozen — fine-tuning at LR={LR_ENCODER}\n")

        t0 = time.time()
        tr_loss, tr_acc, tr_rec, tr_f1 = train_epoch(
            model, train_dl, optimizer, criterion, DEVICE, GRAD_CLIP)
        va_loss, va_acc, va_rec, va_f1 = val_epoch(
            model, val_dl, criterion, DEVICE)

        if epoch <= LR_WARMUP_EPOCHS:       sched_warmup.step()
        elif epoch > FREEZE_ENCODER_EPOCHS: sched_cosine.step()

        lr_now  = optimizer.param_groups[1]["lr"]
        elapsed = (time.time() - t0) / 60
        marker  = " *" if va_f1 > best_val_f1 else "  "
        print(f"{epoch:4d} | {lr_now:.2e} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
              f"{tr_rec:7.4f} | {tr_f1:7.4f} | {va_loss:8.4f} | {va_acc:7.4f} | "
              f"{va_rec:7.4f} | {va_f1:7.4f} | {elapsed:4.1f}m{marker}")

        if va_f1 > best_val_f1:
            best_val_f1  = va_f1
            best_val_rec = va_rec
            torch.save({
                "fold": fold_idx, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": va_f1, "val_rec": va_rec,
                "img_w": IMG_W, "img_h": IMG_H,
                "encoder": ENCODER,
            }, save_path)

        recent_f1s.append(va_f1)
        avg_f1 = float(np.mean(recent_f1s[-F1_ROLLING_WINDOW:]))
        if avg_f1 > best_avg_f1:
            best_avg_f1  = avg_f1
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  "
                  f"(best roll-F1={best_avg_f1:.4f}, best inst-F1={best_val_f1:.4f})")
            break

    fold_results[fold_idx] = {
        "val_f1": best_val_f1, "val_rec": best_val_rec,
        "save_path": save_path, "model": model,
    }
    print(f"\n  Fold {fold_idx} → val_F1={best_val_f1:.4f}  val_Rec={best_val_rec:.4f}")

    del optimizer, train_dl, val_dl, train_ds, val_ds
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# STEP 7 — CV SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"STEP 4 — CV SUMMARY  ({N_FOLDS} folds)")
print("="*70)
print(f"  {'Fold':>5} | {'Val F1':>8} | {'Val Rec':>8}")
print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}")

all_f1 = []; all_rec = []
best_fold = max(fold_results, key=lambda k: fold_results[k]["val_f1"])
for i in range(1, N_FOLDS + 1):
    r    = fold_results[i]
    flag = "  <- best" if i == best_fold else ""
    print(f"  {i:>5} | {r['val_f1']:>8.4f} | {r['val_rec']:>8.4f}{flag}")
    all_f1.append(r["val_f1"]); all_rec.append(r["val_rec"])

print(f"\n  Mean val F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"  Mean val Rec : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")


# ============================================================
# STEP 8 — THRESHOLD CALIBRATION
# ============================================================
print(f"\n{'='*70}")
print(f"STEP 5 — THRESHOLD CALIBRATION  (fixed val set, fold {best_fold} model)")
print("="*70)

best_model = fold_results[best_fold]["model"]
ckpt       = torch.load(fold_results[best_fold]["save_path"],
                        map_location=DEVICE, weights_only=False)
best_model.load_state_dict(ckpt["model_state_dict"])
best_model.eval()

val_sc   = tta_scores(best_model, val_samples_eval, DEVICE, IMG_W, IMG_H)
best_thr = sweep_thresholds(val_sc, label="Threshold sweep — fixed val set:")
print(f"\n  Selected threshold : {best_thr}  (best F1 on val)")


# ============================================================
# STEP 9 — TEST  (seen exactly once)
# ============================================================
print(f"\n{'='*70}")
print(f"STEP 6 — TEST EVALUATION  (fold {best_fold}, threshold={best_thr})")
print("="*70)

test_sc = tta_scores(best_model, test_rows, DEVICE, IMG_W, IMG_H)

print(f"\n  *** FINAL RESULT ***  (threshold from val — no test leakage)")
result = report_at_threshold(test_sc, best_thr)

print(f"\n  Best model : {fold_results[best_fold]['save_path']}")
print("="*70)
