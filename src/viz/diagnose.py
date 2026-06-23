"""
BhuViveka — Signal Diagnostic
===============================
Two recipes (v1, v2) both failed to learn. Before any more training,
this answers ONE question: do the labels carry learnable signal?

Runs three tests on train_final -> val_final:
  1. Majority-class baseline      (the floor; what "no learning" scores)
  2. Handcrafted-stats probe      (NDWI + per-band mean/std -> logistic)
                                   does ANYTHING separate the classes?
  3. Frozen-Prithvi-CLS probe     (frozen backbone features -> logistic)
                                   does the foundation model see signal?

Interpretation:
  - All three near ~0.33  -> labels have no signal. STOP tuning models;
                             fix the labels. (relabel / verify / reframe)
  - Stats probe >> Prithvi probe -> signal exists but Prithvi washes it
                             out (domain gap / normalization). Fix input
                             normalization to Prithvi's expected stats.
  - Prithvi probe 0.5+    -> signal IS reachable; our training loop has a
                             bug or instability, not a data problem.

Also: --dump-patches saves RGB PNGs of HIGH vs LOW patches for eyeballing.

Usage:
  uv run diagnose_signal.py
  uv run diagnose_signal.py --dump-patches
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

GREEN, YELLOW, RED, CYAN = "\033[92m", "\033[93m", "\033[91m", "\033[96m"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"

PROJECT_ROOT = Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BACKBONE_DIR = PROJECT_ROOT / "models" / "prithvi-300m-tl"
CACHE_DIR = PROJECT_ROOT / "data" / "interim" / "probe_cache"
FIG_DIR = PROJECT_ROOT / "figures" / "diagnostic"

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
N_CLASSES = 3
MODEL_IMG = 448


def banner(t):
    bar = "=" * 68
    print(f"\n{CYAN}{bar}\n  {BOLD}{t}{RESET}\n{CYAN}{bar}{RESET}")

def ok(m): print(f"  {GREEN}OK {m}{RESET}")
def info(m): print(f"  {DIM}- {m}{RESET}")
def warn(m): print(f"  {YELLOW}! {m}{RESET}")


def load_split(name):
    p = PROCESSED_DIR / f"{name}.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_six_band(arr, sensor):
    H, W = arr.shape[1], arr.shape[2]
    out = np.zeros((6, H, W), dtype=np.float32)
    if sensor == "LISS3":
        out[1] = arr[0]; out[2] = arr[1]; out[3] = arr[2]; out[4] = arr[3]
    elif sensor == "LISS4":
        out[1] = arr[0]; out[2] = arr[1]; out[3] = arr[2]
    else:
        n = min(6, arr.shape[0]); out[:n] = arr[:n]
    return out


# ============================================================
# METRICS
# ============================================================

def f1_report(y_true, y_pred, tag):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in range(N_CLASSES):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f1s.append(f1)
    macro = float(np.mean(f1s))
    acc = float((y_true == y_pred).mean())
    print(f"  {BOLD}{tag}{RESET}")
    print(f"    acc={acc:.3f}  F1-macro={GREEN if macro>0.45 else YELLOW if macro>0.38 else RED}{macro:.3f}{RESET}"
          f"   per-class: LOW={f1s[0]:.3f} MED={f1s[1]:.3f} HIGH={f1s[2]:.3f}")
    return macro


# ============================================================
# HANDCRAFTED FEATURES
# ============================================================

def handcrafted_features(arr6):
    """Per-band mean/std + NDWI mean/std -> 14-dim vector."""
    feats = []
    for b in range(6):
        band = arr6[b]
        valid = band[band > 0]
        feats.append(valid.mean() if len(valid) else 0.0)
        feats.append(valid.std() if len(valid) else 0.0)
    g, nir = arr6[1], arr6[3]
    denom = g + nir
    ndwi = np.where(denom > 0, (g - nir) / (denom + 1e-6), 0)
    feats.append(float(ndwi.mean()))
    feats.append(float(ndwi.std()))
    return np.array(feats, dtype=np.float32)


# ============================================================
# TORCH LINEAR PROBE (dependency-free, class-balanced)
# ============================================================

def linear_probe(Xtr, ytr, Xva, yva, n_steps=400, lr=0.05):
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # standardize
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva = torch.tensor(Xva, dtype=torch.float32, device=device)

    counts = Counter(ytr.cpu().numpy().tolist())
    w = torch.tensor([1.0 / max(counts.get(c, 1), 1) for c in range(N_CLASSES)],
                     dtype=torch.float32, device=device)
    w = w / w.sum() * N_CLASSES

    clf = nn.Linear(Xtr.shape[1], N_CLASSES).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=w)
    clf.train()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = lossf(clf(Xtr), ytr)
        loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Xva).argmax(1).cpu().numpy()
    return pred


# ============================================================
# PRITHVI FEATURE EXTRACTION (frozen, cached)
# ============================================================

def extract_prithvi_features(rows, tag):
    import torch.nn.functional as F
    cache = CACHE_DIR / f"prithvi_cls_{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        info(f"loaded cached {tag} features {d['X'].shape}")
        return d["X"], d["y"]

    from prithvi_model import PrithviViTEncoder, load_prithvi_backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"loading frozen Prithvi for {tag} feature extraction")
    model, _ = load_prithvi_backbone("300M", BACKBONE_DIR)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    feats, labels = [], []
    B = 4
    buf_img, buf_lab = [], []

    def flush():
        if not buf_img:
            return
        x = torch.from_numpy(np.stack(buf_img)).float().to(device)
        x = F.interpolate(x, size=(MODEL_IMG, MODEL_IMG),
                          mode="bilinear", align_corners=False)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            tok = model(x)
            cls = tok[:, 0].float().cpu().numpy()
        feats.extend(list(cls)); labels.extend(buf_lab)
        buf_img.clear(); buf_lab.clear()

    for i, r in enumerate(rows):
        arr = np.load(PROJECT_ROOT / r["npy_path"])
        arr = to_six_band(arr, r["sensor"])
        buf_img.append(arr); buf_lab.append(int(r["label_id"]))
        if len(buf_img) >= B:
            flush()
        if (i + 1) % 200 == 0:
            print(f"    {tag}: {i+1}/{len(rows)}")
    flush()

    X = np.array(feats, dtype=np.float32); y = np.array(labels, dtype=np.int64)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y)
    ok(f"{tag} features {X.shape} cached")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return X, y


# ============================================================
# PATCH DUMP (eyeball test)
# ============================================================

def dump_patches(train_rows, n=12):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warn("matplotlib missing, skip dump")
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    by_cls = {0: [], 1: [], 2: []}
    for r in train_rows:
        by_cls[int(r["label_id"])].append(r)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.4, 4.5))
    for ci, cls in enumerate([2, 1, 0]):   # HIGH, MED, LOW rows
        rows = by_cls[cls][:n]
        for j in range(n):
            ax = axes[ci, j]; ax.axis("off")
            if j >= len(rows):
                continue
            arr = np.load(PROJECT_ROOT / rows[j]["npy_path"])
            arr6 = to_six_band(arr, rows[j]["sensor"])
            rgb = np.stack([arr6[2], arr6[1], arr6[3]], axis=-1)  # R,G,NIR
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb)
            if j == 0:
                ax.set_title(CLASS_NAMES[cls], loc="left", fontsize=11,
                             fontweight="bold")
    plt.suptitle("HIGH (top) vs MEDIUM (mid) vs LOW (bottom) — can you tell them apart?",
                 fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "patch_eyeball_test.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"saved {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-patches", action="store_true")
    args = ap.parse_args()

    print(f"\n{BOLD}BhuViveka - Signal Diagnostic{RESET}")
    train_rows = load_split("train_final")
    val_rows = load_split("val_final")
    info(f"train {len(train_rows)}  val {len(val_rows)}")
    info(f"train class counts: {Counter(int(r['label_id']) for r in train_rows)}")
    info(f"val   class counts: {Counter(int(r['label_id']) for r in val_rows)}")

    yva = np.array([int(r["label_id"]) for r in val_rows])
    ytr = np.array([int(r["label_id"]) for r in train_rows])

    # ---- Test 1: majority baseline ----
    banner("Test 1 - Majority-class baseline (the floor)")
    maj = Counter(ytr).most_common(1)[0][0]
    f1_report(yva, np.full_like(yva, maj), f"always predict {CLASS_NAMES[maj]}")

    # ---- Test 2: handcrafted-stats probe ----
    banner("Test 2 - Handcrafted pixel-stats probe (NDWI + band stats)")
    info("computing handcrafted features")
    Xtr_h = np.stack([handcrafted_features(to_six_band(np.load(PROJECT_ROOT / r["npy_path"]), r["sensor"]))
                      for r in train_rows])
    Xva_h = np.stack([handcrafted_features(to_six_band(np.load(PROJECT_ROOT / r["npy_path"]), r["sensor"]))
                      for r in val_rows])
    pred_h = linear_probe(Xtr_h, ytr, Xva_h, yva)
    f1_h = f1_report(yva, pred_h, "stats -> logistic")

    # ---- Test 3: frozen Prithvi CLS probe ----
    banner("Test 3 - Frozen Prithvi CLS-feature probe")
    Xtr_p, ytr_p = extract_prithvi_features(train_rows, "train")
    Xva_p, yva_p = extract_prithvi_features(val_rows, "val")
    pred_p = linear_probe(Xtr_p, ytr_p, Xva_p, yva_p)
    f1_p = f1_report(yva_p, pred_p, "Prithvi CLS -> logistic")

    if args.dump_patches:
        banner("Eyeball test - dumping sample patches")
        dump_patches(train_rows)

    # ---- Verdict ----
    banner("VERDICT")
    print(f"  stats-probe   F1-macro: {f1_h:.3f}")
    print(f"  Prithvi-probe F1-macro: {f1_p:.3f}")
    print()
    if max(f1_h, f1_p) < 0.40:
        print(f"  {RED}{BOLD}-> No learnable signal in the labels.{RESET}")
        print(f"  {RED}   Both probes are near the majority floor. More model")
        print(f"     tuning (CBAM, bigger LoRA, etc.) will NOT help.")
        print(f"     The tier x window labels are the bottleneck.{RESET}")
        print(f"  Next: relabel / verify a HIGH sample by eye, or reframe the task.")
    elif f1_h - f1_p > 0.10:
        print(f"  {YELLOW}{BOLD}-> Signal exists in pixels but Prithvi washes it out.{RESET}")
        print(f"  {YELLOW}   Fix input normalization to Prithvi's expected band stats,")
        print(f"     then retrain.{RESET}")
    elif f1_p >= 0.50:
        print(f"  {GREEN}{BOLD}-> Signal IS reachable by Prithvi features.{RESET}")
        print(f"  {GREEN}   The training loop has a bug/instability, not a data")
        print(f"     problem. A frozen-features + trainable-head model should")
        print(f"     hit ~{f1_p:.2f}. Debug the end-to-end loop.{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}-> Weak but non-zero signal ({max(f1_h,f1_p):.2f}).{RESET}")
        print(f"  {YELLOW}   Marginal. Likely needs better labels to be publishable.{RESET}")
    print()


if __name__ == "__main__":
    main()