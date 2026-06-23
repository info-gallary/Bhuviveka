"""
BhuViveka — Stage 1 (v3): Lake Segmentation
=============================================
Trains Prithvi-EO-2.0 + lightweight UperNet-style decoder for
binary lake segmentation. Predicts which pixels are glacial lake.

Why this works where risk-classification failed:
  - Labels are real (Hi-MAG polygon rasters), not heuristic
  - Task is pixel-visible (lakes ARE in the pixels)
  - Hundreds of positive examples per epoch (not 66)
  - Segmentation IoU is a stable metric (vs 10-sample HIGH F1)

Architecture:
  Input (B, 6, 128, 128)
    -> SpectralAdapter (learnable 6->6 residual CNN)
    -> upsample to 448x448
    -> Prithvi encoder (frozen + LoRA on QKV)
    -> reshape tokens to feature map (28x28 grid)
    -> lightweight decoder (2x bilinear upsample + conv)
    -> sigmoid -> (B, 1, 128, 128) lake probability map

Loss: BCE-Dice combo (handles class imbalance; lakes are ~5-15% of pixels)

Outputs:
  checkpoints/seg_300m/best.pt
  figures/seg_training/
    training_curves.png      loss + IoU per epoch
    sample_predictions.png   input | GT mask | predicted mask

Risk derivation AFTER segmentation:
  - Measure lake_area_px per patch (sum of predicted mask)
  - Multi-temporal: compare pre-event vs event window lake area
  - Large + growing = HIGH; stable = MEDIUM; small/shrinking = LOW
  -> This is your PINN stage (physics-grounded, not ML-guessed)

Usage:
  uv run train_seg.py --dry-run
  uv run train_seg.py --epochs 30
  uv run train_seg.py --backbone 600M
  uv run train_seg.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

GREEN, YELLOW, RED, CYAN = "\033[92m", "\033[93m", "\033[91m", "\033[96m"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def banner(t, p=None):
    bar = "=" * 68
    label = f"PHASE {p}" if p else "INFO"
    print(f"\n{CYAN}{bar}\n  {BOLD}{label}{RESET}{CYAN}  -  {t}{RESET}\n{CYAN}{bar}{RESET}")

def ok(m="OK"): print(f"  {GREEN}OK {m}{RESET}")
def info(m): print(f"  {DIM}- {m}{RESET}")
def warn(m): print(f"  {YELLOW}! {m}{RESET}")
def fatal(m): print(f"\n{RED}{BOLD}STOP: {m}{RESET}"); sys.exit(1)


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path.cwd()
SEG_INDEX    = PROJECT_ROOT / "data" / "interim" / "training_index_seg.csv"
MASK_DIR     = PROJECT_ROOT / "data" / "interim" / "masks_v1"

BACKBONE_PATHS = {
    "300M": PROJECT_ROOT / "models" / "prithvi-300m-tl",
    "600M": PROJECT_ROOT / "models" / "prithvi-600m-tl",
}

PATCH_SIZE_INPUT = 128
PATCH_SIZE_MODEL = 448
BATCH_SIZE = 2
GRAD_ACCUM = 4              # effective batch 8
LR_LORA    = 3e-5
LR_DECODER = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_PCT   = 0.05
EPOCHS       = 30
PATIENCE     = 7
LORA_RANK    = 16
LORA_ALPHA   = 32
RANDOM_SEED  = 42

TRAIN_EVENTS = ["sikkim_south_lhonak_2016"]
TEST_EVENTS  = ["sikkim_south_lhonak_2023",
                "uttarakhand_chamoli_2021",
                "uttarakhand_chorabari_2013",
                "uttarakhand_kedarnath_2013",
                "himachal_gepang_gath"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ============================================================
# DATASET
# ============================================================

def load_seg_index():
    if not SEG_INDEX.exists():
        fatal(f"{SEG_INDEX} not found — run make_seg_masks.py first")
    with open(SEG_INDEX, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class LakeSegDataset(torch.utils.data.Dataset):
    def __init__(self, rows, training=True):
        self.training = training
        # Keep rows that have a valid mask
        self.rows = [r for r in rows if r.get("mask_path") and
                     (PROJECT_ROOT / r["mask_path"]).exists()]
        info(f"  dataset: {len(self.rows)}/{len(rows)} rows have masks")

    def __len__(self): return len(self.rows)

    def _to_six_band(self, arr, sensor):
        H, W = arr.shape[1], arr.shape[2]
        out = np.zeros((6, H, W), dtype=np.float32)
        if sensor == "LISS3":
            out[1]=arr[0]; out[2]=arr[1]; out[3]=arr[2]; out[4]=arr[3]
        elif sensor == "LISS4":
            out[1]=arr[0]; out[2]=arr[1]; out[3]=arr[2]
        else:
            n=min(6,arr.shape[0]); out[:n]=arr[:n]
        return out

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img=img[:,:,::-1].copy(); mask=mask[:,::-1].copy()
        if random.random() < 0.5:
            img=img[:,::-1,:].copy(); mask=mask[::-1,:].copy()
        k = random.randint(0,3)
        if k:
            img=np.rot90(img,k,axes=(1,2)).copy()
            mask=np.rot90(mask,k).copy()
        return img, mask

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = np.load(PROJECT_ROOT / row["npy_path"])
        img = self._to_six_band(img, row["sensor"])
        mask = np.load(PROJECT_ROOT / row["mask_path"]).astype(np.float32)
        if self.training:
            img, mask = self._augment(img, mask)
        return {
            "image": torch.from_numpy(img).float(),
            "mask":  torch.from_numpy(mask).float().unsqueeze(0),
            "has_lake": float(row.get("has_lake", 0)),
        }


def make_loaders(all_rows, batch_size):
    train_rows = [r for r in all_rows if r.get("event") in TRAIN_EVENTS]
    import random as _rnd
    if len(train_rows) > 4000:
        train_rows = _rnd.Random(RANDOM_SEED).sample(train_rows, 4000)
    val_rows   = [r for r in all_rows if r.get("event") in TRAIN_EVENTS]
    # Use 15% for val
    random.seed(RANDOM_SEED); random.shuffle(train_rows)
    n_val = int(len(train_rows) * 0.15)
    val_rows   = train_rows[:n_val]
    train_rows = train_rows[n_val:]

    train_ds = LakeSegDataset(train_rows, training=True)
    val_ds   = LakeSegDataset(val_rows,   training=False)

    # Oversample patches that have lakes
    has_lake = [float(r.get("has_lake", 0)) for r in train_ds.rows]
    n_lake = max(sum(1 for h in has_lake if h > 0), 1)
    n_nolake = max(sum(1 for h in has_lake if h == 0), 1)
    weights = [3.0/n_lake if h>0 else 1.0/n_nolake for h in has_lake]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_rows, val_rows

# train_rows = [r for r in all_rows if r.get("event") in TRAIN_EVENTS]
# Cap to 4000 patches — Sikkim 2016 has 16K, redundancy hurts more than it helps

# ============================================================
# MODEL — SPECTRAL ADAPTER
# ============================================================

class SpectralAdapter(nn.Module):
    def __init__(self, n_bands=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bands, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, n_bands, 1),
        )
        self.bn = nn.BatchNorm2d(n_bands)

    def forward(self, x):
        return self.bn(x + self.net(x))


# ============================================================
# MODEL — LIGHTWEIGHT DECODER
# ============================================================

class LakeDecoder(nn.Module):
    """Lightweight decoder: reshape tokens -> 28x28 -> upsample -> 128x128.

    Prithvi outputs (B, 785, 1024) — 784 patch tokens + 1 CLS.
    We drop CLS, reshape to (B, 1024, 28, 28) feature map,
    then upsample with transposed convs to (B, 1, 128, 128).
    """

    def __init__(self, embed_dim=1024, grid_size=28, out_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.out_size  = out_size
        self.decode = nn.Sequential(
            # (B, D, 28, 28) -> (B, 256, 28, 28)
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.GELU(),
            # -> (B, 128, 56, 56)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.GELU(),
            # -> (B, 64, 112, 112)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.GELU(),
            # -> (B, 1, 112, 112)
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, tokens):
        # tokens: (B, 785, D) — drop CLS token
        patch_tokens = tokens[:, 1:, :]          # (B, 784, D)
        B, N, D = patch_tokens.shape
        # reshape to 2D feature map
        feat = patch_tokens.permute(0, 2, 1)      # (B, D, 784)
        feat = feat.reshape(B, D, self.grid_size, self.grid_size)
        out  = self.decode(feat)                   # (B, 1, ~112, ~112)
        # bilinear resize to exact output size
        out  = F.interpolate(out, size=(self.out_size, self.out_size),
                             mode="bilinear", align_corners=False)
        return out                                 # (B, 1, 128, 128)


# ============================================================
# MODEL — LORA
# ============================================================

def apply_lora(backbone):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        fatal("uv pip install peft")
    target_modules = [name for name, m in backbone.named_modules()
                      if isinstance(m, nn.Linear) and name.endswith(".qkv")]
    if not target_modules:
        target_modules = [name for name, m in backbone.named_modules()
                          if isinstance(m, nn.Linear) and "attn" in name.lower()]
    info(f"LoRA targets: {len(target_modules)} QKV modules")
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                     target_modules=target_modules, lora_dropout=0.05,
                     bias="none", modules_to_save=["patch_embed"])
    return get_peft_model(backbone, cfg)


# ============================================================
# MODEL — FULL BHUVIVEKA-SEG
# ============================================================

class BhuVivekaSeg(nn.Module):
    def __init__(self, backbone, embed_dim):
        super().__init__()
        self.adapter = SpectralAdapter(6)
        self.backbone = backbone
        self.decoder  = LakeDecoder(embed_dim=embed_dim,
                                     grid_size=PATCH_SIZE_MODEL // 16,
                                     out_size=PATCH_SIZE_INPUT)

    def forward(self, x):
        x = self.adapter(x)
        x = F.interpolate(x, size=(PATCH_SIZE_MODEL, PATCH_SIZE_MODEL),
                          mode="bilinear", align_corners=False)
        tokens = self.backbone(x)           # (B, 785, D)
        return self.decoder(tokens)          # (B, 1, 128, 128)


def build_model(model_size, device):
    banner("Build BhuVivekaSeg model", 3)
    from prithvi_model import load_prithvi_backbone as _load
    step_print("Loading Prithvi")
    backbone, _ = _load(model_size, BACKBONE_PATHS[model_size])
    embed_dim = backbone.embed_dim
    ok(f"backbone embed_dim={embed_dim}")
    for p in backbone.parameters(): p.requires_grad = False
    step_print("Injecting LoRA")
    backbone = apply_lora(backbone)
    ok("LoRA injected")
    model = BhuVivekaSeg(backbone, embed_dim).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ok(f"trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")
    return model

def step_print(t): print(f"  > {t}", end=" ", flush=True)


# ============================================================
# LOSS — BCE + DICE COMBO
# ============================================================

def bce_dice_loss(pred, target, bce_weight=0.5):
    """pred: (B,1,H,W) logits;  target: (B,1,H,W) float 0/1"""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    p = torch.sigmoid(pred)
    smooth = 1.0
    inter = (p * target).sum(dim=(1,2,3))
    dice  = 1 - (2 * inter + smooth) / (p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + smooth)
    return bce_weight * bce + (1 - bce_weight) * dice.mean()


# ============================================================
# METRICS
# ============================================================

def iou_score(pred_logits, target, threshold=0.5):
    p = (torch.sigmoid(pred_logits) > threshold).float()
    t = target.float()
    inter = (p * t).sum()
    union = (p + t).clamp(0,1).sum()
    return float(inter / (union + 1e-6))


def compute_seg_metrics(pred_logits, targets):
    ious = [iou_score(pred_logits[i:i+1], targets[i:i+1]) for i in range(len(targets))]
    p = (torch.sigmoid(pred_logits) > 0.5).float()
    tp = (p * targets).sum().item()
    fp = (p * (1 - targets)).sum().item()
    fn = ((1 - p) * targets).sum().item()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return {"iou": float(np.mean(ious)), "f1": float(f1),
            "precision": float(prec), "recall": float(rec)}


# ============================================================
# TRAIN + EVAL
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, device,
                log_path, epoch, grad_accum, steps_so_far):
    model.train()
    total_loss = 0; n = 0; all_iou = []
    n_batches = len(loader); step_in = 0
    optimizer.zero_grad()

    for bi, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)
            loss   = bce_dice_loss(logits, y) / grad_accum

        scaler.scale(loss).backward()
        if (bi+1) % grad_accum == 0 or (bi+1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step(); optimizer.zero_grad(); step_in += 1

        with torch.no_grad():
            all_iou.append(iou_score(logits.detach(), y))
            total_loss += loss.item() * grad_accum * x.size(0)
            n += x.size(0)

        if bi % 10 == 0:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "batch": bi,
                    "global_step": steps_so_far + step_in,
                    "loss": float(loss.item() * grad_accum),
                    "lr": float(scheduler.get_last_lr()[0]),
                }) + "\n")

        if (bi+1) % max(1, n_batches//5) == 0:
            print(f"    ep{epoch} {bi+1}/{n_batches}  "
                  f"loss={total_loss/n:.4f}  IoU={np.mean(all_iou):.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return {"loss": total_loss/max(n,1), "iou": float(np.mean(all_iou))}, \
           steps_so_far + step_in


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0; n = 0
    all_preds, all_targets = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)
            loss   = bce_dice_loss(logits.float(), y)
        all_preds.append(logits.float().cpu())
        all_targets.append(y.float().cpu())
        total_loss += loss.item() * x.size(0); n += x.size(0)
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    m = compute_seg_metrics(preds, targets)
    m["loss"] = total_loss / max(n, 1)
    return m


# ============================================================
# TRAINING CURVES
# ============================================================

def render_curves(log_path, val_history, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError: return
    if not log_path.exists() or not val_history: return
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                for k,v in r.items(): train_data[k].append(v)
            except: pass

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})

    if train_data.get("global_step"):
        axes[0].plot(train_data["global_step"], train_data["loss"],
                     color="#3498db", alpha=0.7, label="train")
    val_steps = [v.get("global_step", v["epoch"]) for v in val_history]
    val_epochs = [v["epoch"] for v in val_history]
    axes[0].plot(val_steps, [v["loss"] for v in val_history], "o-",
                 color="#c0392b", label="val", linewidth=2)
    axes[0].set_title("BCE+Dice Loss", fontweight="bold")
    axes[0].set_xlabel("Global step"); axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25, linestyle=":")

    ious = [v["iou"] for v in val_history]
    axes[1].plot(val_epochs, ious, "o-", color="#27ae60", linewidth=2, markersize=7)
    best = int(np.argmax(ious))
    axes[1].scatter([val_epochs[best]], [ious[best]], color="gold", s=200,
                     zorder=5, edgecolor="black",
                     label=f"best IoU {ious[best]:.3f} @ ep{val_epochs[best]}")
    axes[1].set_title("Val IoU (lake segmentation)", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25, linestyle=":")

    for k, c in [("f1","#e74c3c"), ("precision","#9b59b6"), ("recall","#f39c12")]:
        axes[2].plot(val_epochs, [v[k] for v in val_history], "o-",
                     color=c, label=k, linewidth=2, markersize=5)
    axes[2].set_title("Val F1 / Precision / Recall", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.25, linestyle=":")

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# CHECKPOINT
# ============================================================

def save_ckpt(model, optimizer, scheduler, scaler, epoch, metrics, path, label="ckpt"):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "metrics": metrics}, path)
    ok(f"saved {label} -> {path.name}")

def load_ckpt(path, model, optimizer, scheduler, scaler):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    scheduler.load_state_dict(ck["scheduler_state_dict"])
    scaler.load_state_dict(ck["scaler_state_dict"])
    return ck["epoch"], ck["metrics"]


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["300M","600M"], default="300M")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"\n{BOLD}BhuViveka - Lake Segmentation Training{RESET}")
    print(f"{DIM}Backbone: {args.backbone}-TL  |  Task: binary lake segmentation{RESET}")
    set_seed(RANDOM_SEED)

    ckpt_dir = PROJECT_ROOT / "checkpoints" / f"seg_{args.backbone.lower()}"
    log_dir  = PROJECT_ROOT / "logs"
    fig_dir  = PROJECT_ROOT / "figures" / "seg_training"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seg_{args.backbone.lower()}.jsonl"

    banner("Load segmentation index", 1)
    all_rows = load_seg_index()
    ok(f"{len(all_rows)} total patches")
    train_loader, val_loader, tr, va = make_loaders(all_rows, BATCH_SIZE)
    info(f"train {len(tr)}  val {len(va)}")
    info(f"train batches {len(train_loader)}  val batches {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu": fatal("CUDA required")
    info(f"device: {torch.cuda.get_device_name(0)}")

    model = build_model(args.backbone, device)

    banner("Optimizer", 4)
    lora_p, decoder_p, other_p = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        n = name.lower()
        if "lora" in n: lora_p.append(p)
        elif "decoder" in n or "adapter" in n: decoder_p.append(p)
        else: other_p.append(p)
    info(f"LoRA {sum(p.numel() for p in lora_p)/1e6:.2f}M  "
         f"Decoder {sum(p.numel() for p in decoder_p)/1e6:.2f}M  "
         f"Other {sum(p.numel() for p in other_p)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [{"params": lora_p,    "lr": LR_LORA},
         {"params": decoder_p + other_p, "lr": LR_DECODER}],
        weight_decay=WEIGHT_DECAY)

    total_steps  = (len(train_loader) // GRAD_ACCUM) * args.epochs
    warmup_steps = max(1, int(WARMUP_PCT * total_steps))
    def lr_lambda(s):
        if s < warmup_steps: return s / warmup_steps
        p = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0; best_iou = 0.0; val_history = []
    if args.resume:
        last = ckpt_dir / "last.pt"
        if last.exists():
            start_epoch, m = load_ckpt(last, model, optimizer, scheduler, scaler)
            best_iou = m.get("iou", 0.0)
            ok(f"resumed from epoch {start_epoch}, best IoU {best_iou:.3f}")
    if not args.resume and log_path.exists():
        log_path.unlink()

    banner("Train", 6)
    if args.dry_run:
        warn("--dry-run: 1 epoch only"); args.epochs = start_epoch + 1

    patience_counter = 0; steps_so_far = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\n  {BOLD}Epoch {epoch+1}/{args.epochs}{RESET}")
        tr_m, steps_so_far = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            log_path, epoch, GRAD_ACCUM, steps_so_far)
        elapsed = time.time() - t0
        info(f"  TRAIN  loss={tr_m['loss']:.4f}  IoU={tr_m['iou']:.3f}  ({elapsed:.0f}s)")

        val_m = eval_epoch(model, val_loader, device)
        val_m["epoch"] = epoch + 1; val_m["global_step"] = steps_so_far
        val_history.append(val_m)
        info(f"  VAL    loss={val_m['loss']:.4f}  IoU={val_m['iou']:.3f}  "
             f"F1={val_m['f1']:.3f}  prec={val_m['precision']:.3f}  "
             f"rec={val_m['recall']:.3f}")

        save_ckpt(model, optimizer, scheduler, scaler,
                   epoch+1, val_m, ckpt_dir/"last.pt", "last")
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_ckpt(model, optimizer, scheduler, scaler,
                       epoch+1, val_m, ckpt_dir/"best.pt", "best")
            patience_counter = 0
        else:
            patience_counter += 1
        if (epoch+1) % 5 == 0:
            save_ckpt(model, optimizer, scheduler, scaler, epoch+1, val_m,
                       ckpt_dir/f"epoch_{epoch+1}.pt", f"ep{epoch+1}")

        render_curves(log_path, val_history, fig_dir)
        if patience_counter >= PATIENCE:
            warn(f"early stop — no IoU improvement for {PATIENCE} epochs")
            break

    print(f"\n{GREEN}{BOLD}Segmentation training complete.{RESET}")
    print(f"  best val IoU: {best_iou:.4f}")
    print(f"  Next: uv run risk_from_seg.py\n")


if __name__ == "__main__":
    main()
