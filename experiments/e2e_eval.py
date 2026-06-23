"""
BhuViveka — End-to-End Evaluation v3 (fast, progress bar)
==========================================================
Same as v2 but:
  - Batched GPU inference (8 patches at a time → ~8× faster)
  - Live progress bar so you know it's running
  - --max default 200 per event (1000 patches total) for a quick first run
  - --max 9999 for full eval

Usage:
    uv run experiment_e2e_eval_v3.py              # quick (1000 patches, ~2 min)
    uv run experiment_e2e_eval_v3.py --max 9999   # full (all patches, ~15 min)
    uv run experiment_e2e_eval_v3.py --split temporal
"""
from __future__ import annotations
import argparse, csv, json, sys, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

ROOT     = Path.cwd()
_fixed = ROOT/"data"/"processed"/"risk_from_seg"/"risk_labeled_with_centroids.csv"
_orig  = ROOT/"data"/"processed"/"risk_from_seg"/"risk_labeled_all.csv"
RISK_CSV = _fixed if _fixed.exists() else _orig

SEG_CKPT = ROOT/"checkpoints"/"seg_300m"/"best.pt"
RL_CKPT  = ROOT/"checkpoints"/"stage2_grpo"/"best.pt"
OUT_DIR  = ROOT/"data"/"results"/"e2e_eval"
CLASSES  = ["LOW","MEDIUM","HIGH"]
PIXEL_M2 = {"LISS3": 23.5**2, "LISS4": 5.8**2}

TEST_TEMPORAL = ["sikkim_south_lhonak_2023"]
TEST_SPATIAL  = ["uttarakhand_chamoli_2021","uttarakhand_chorabari_2013",
                 "uttarakhand_kedarnath_2013","himachal_gepang_gath"]

BATCH = 8   # patches per GPU batch — adjust down to 4 if OOM


def safe_load(p):
    f = ROOT/p
    if not f.exists(): return None
    try:
        a = np.load(f, allow_pickle=False)
        return np.nan_to_num(a,0,0,0).astype(np.float32) if a.ndim==3 else None
    except: return None


def to6(arr, sensor):
    out = np.zeros((6,arr.shape[1],arr.shape[2]),np.float32)
    if sensor=="LISS3" and arr.shape[0]>=4: out[1:5]=arr[:4]
    elif sensor=="LISS4" and arr.shape[0]>=3: out[1:4]=arr[:3]
    else: n=min(6,arr.shape[0]); out[:n]=arr[:n]
    return out


def pred_area(mask_np, sensor):
    return float((mask_np>0.5).sum() * PIXEL_M2.get(sensor,23.5**2) / 1e6)


def load_models(device):
    """Load with embed_dim fix: int(base.embed_dim), not the tuple return value."""
    from train_seg import BhuVivekaSeg
    from train_stage2_grpo import ReasoningHead
    from prithvi_model import load_prithvi_backbone as _load
    from peft import get_peft_model, LoraConfig

    base, _ = _load("300M", ROOT/"models"/"prithvi-300m-tl")
    for p in base.parameters(): p.requires_grad = False
    dim = int(base.embed_dim)   # ← THE FIX

    tgt = [n for n,m in base.named_modules()
           if isinstance(m,nn.Linear) and n.endswith(".qkv")]
    if not tgt:
        tgt = [n for n,m in base.named_modules()
               if isinstance(m,nn.Linear) and "attn" in n.lower()]

    cfg = LoraConfig(r=16,lora_alpha=32,target_modules=tgt,
                     lora_dropout=0.05,bias="none",modules_to_save=["patch_embed"])
    seg = BhuVivekaSeg(get_peft_model(base,cfg), dim).to(device).eval()
    seg.load_state_dict(torch.load(SEG_CKPT,map_location="cpu",weights_only=False)
                        ["model_state_dict"])

    rh = ReasoningHead(visual_dim=dim, physics_dim=3).to(device).eval()
    rh.load_state_dict(torch.load(RL_CKPT,map_location="cpu",weights_only=False)
                       ["reasoning_state_dict"])
    return seg, rh, dim


def bar(done, total, width=30):
    pct = done/max(total,1)
    filled = int(pct*width)
    return f"[{'█'*filled}{'░'*(width-filled)}] {done}/{total} ({pct:.0%})"


def run_batched(seg, rh, rows, device, use_gt, label):
    """Batched inference. Returns (preds, labels)."""
    preds, labels = [], []
    skipped = 0
    n = len(rows)
    t0 = time.time()

    i = 0
    while i < n:
        batch_rows = rows[i:i+BATCH]
        imgs, phy_gt, phy_pred, true_labs = [], [], [], []

        for r in batch_rows:
            arr = safe_load(r.get("npy_path",""))
            if arr is None: skipped+=1; continue
            imgs.append(to6(arr, r["sensor"]))
            area   = float(r.get("seg_area_km2",0) or 0)
            change = float(r.get("change_pct",0) or 0)
            dist   = float(r.get("dist_km",0) or 0)
            phy_gt.append([area, change, dist])
            phy_pred.append([0.0, 0.0, dist])  # area+change from prediction below
            true_labs.append(r.get("risk_label","LOW"))

        if not imgs:
            i += BATCH; continue

        try:
            x = torch.from_numpy(np.stack(imgs)).float().to(device)
            with torch.no_grad():
                x_up = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
                seg_logits = seg(x_up)
                seg_probs  = torch.sigmoid(seg_logits)[:,0].cpu().numpy()  # (B,H,W)

                # Get CLS tokens
                x5 = x_up.unsqueeze(2)
                tokens = seg.backbone(x5)
                cls = tokens[:,0,:]  # (B,D)

                # Build physics tensors
                if use_gt:
                    phy_t = torch.tensor(phy_gt, dtype=torch.float32).to(device)
                else:
                    # Replace area with predicted area; change stays 0
                    phy_arr = [
                        [pred_area(seg_probs[j], batch_rows[j]["sensor"]),
                         0.0,
                         float(batch_rows[j].get("dist_km",0) or 0)]
                        for j in range(len(imgs))
                    ]
                    phy_t = torch.tensor(phy_arr, dtype=torch.float32).to(device)

                logits, _ = rh(cls, phy_t)
                pred_ids = logits.argmax(-1).cpu().numpy()

            for j in range(len(imgs)):
                preds.append(CLASSES[int(pred_ids[j])])
                labels.append(true_labs[j])

        except Exception as e:
            skipped += len(imgs)

        i += BATCH

        # Progress bar
        done = len(preds)+skipped
        elapsed = time.time()-t0
        eta = elapsed/max(done,1)*(n-done)
        sys.stdout.write(f"\r  {label}: {bar(done,n)}  skip={skipped}  "
                         f"ETA {eta:.0f}s  ")
        sys.stdout.flush()

    sys.stdout.write("\n")
    return preds, labels, skipped


def report(preds, labels, tag):
    from sklearn.metrics import (f1_score, balanced_accuracy_score,
                                  precision_recall_fscore_support, confusion_matrix)
    if not preds:
        print(f"  {tag}: no predictions"); return {}
    mf1 = f1_score(labels,preds,average="macro",labels=CLASSES,zero_division=0)
    ba  = balanced_accuracy_score(labels,preds)
    p,r,f,_ = precision_recall_fscore_support(labels,preds,labels=CLASSES,zero_division=0)
    cm = confusion_matrix(labels,preds,labels=CLASSES)
    print(f"\n{'='*58}")
    print(f"  {tag}  (n={len(preds)})")
    print(f"  Macro-F1:          {mf1:.4f}")
    print(f"  Balanced accuracy: {ba:.4f}")
    print(f"  {'Class':<10} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    for i,c in enumerate(CLASSES):
        print(f"  {c:<10} {p[i]:>6.3f} {r[i]:>6.3f} {f[i]:>6.3f}")
    print(f"\n  Confusion (rows=GT, cols=pred):")
    print("  " + " ".join(f"{c:>8}" for c in [""]+CLASSES))
    for i,row in enumerate(cm):
        print(f"  {CLASSES[i]:<8} " + " ".join(f"{v:>8d}" for v in row))
    return {"tag":tag,"n":len(preds),"macro_f1":round(mf1,4),"bal_acc":round(ba,4),
            "per_class":{c:{"p":float(p[i]),"r":float(r[i]),"f1":float(f[i])}
                         for i,c in enumerate(CLASSES)}}


def main():
    global BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["temporal","spatial","all"], default="all")
    ap.add_argument("--max", type=int, default=200,
                    help="Max patches per event (default 200 = ~1000 total, fast)")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    BATCH = args.batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Batch size: {BATCH}")
    print(f"CSV: {RISK_CSV.name}")

    print("Loading models...")
    seg, rh, dim = load_models(device)
    print(f"  embed_dim = {dim}  ✓")

    all_rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    events = (TEST_TEMPORAL if args.split=="temporal"
              else TEST_SPATIAL if args.split=="spatial"
              else TEST_TEMPORAL+TEST_SPATIAL)

    # Sample up to max_n per event
    count = defaultdict(int)
    test_rows = []
    for r in all_rows:
        ev = r.get("event","")
        if ev in events and count[ev] < args.max:
            test_rows.append(r); count[ev]+=1

    print(f"\nTest set: {len(test_rows)} patches "
          f"(≤{args.max}/event × {len(events)} events)")
    per = {ev: count[ev] for ev in events}
    for ev, n in per.items(): print(f"  {ev}: {n}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Oracle
    print(f"\n--- Oracle: GT descriptors (upper bound, target leakage) ---")
    p_gt, l_gt, sk = run_batched(seg, rh, test_rows, device, use_gt=True, label="oracle")
    print(f"  Skipped: {sk}")
    results["oracle_gt"] = report(p_gt, l_gt, "Oracle GT-descriptor (upper bound)")
    if results["oracle_gt"]:
        results["oracle_gt"]["WARNING"] = (
            "Target leakage: area/change/dist overlap with label-deriving rule. "
            "This is the UPPER BOUND. Not a validated end-to-end result.")

    # True E2E
    print(f"\n--- True E2E: predicted area, change_rate=0 (lower bound) ---")
    p_e2e, l_e2e, sk2 = run_batched(seg, rh, test_rows, device, use_gt=False, label="e2e")
    print(f"  Skipped: {sk2}")
    results["e2e_predicted"] = report(p_e2e, l_e2e, "True E2E (pred area, change=0)")
    if results["e2e_predicted"]:
        results["e2e_predicted"]["NOTES"] = [
            "area = predicted from Stage 1 mask  (true E2E)",
            "change_rate = 0.0  (temporal pairing not yet implemented → lower bound)",
            "dist_km = from GT CSV  (distance bug not yet fixed)",
        ]

    out = OUT_DIR / f"e2e_results_{args.split}_v3.json"
    json.dump(results, open(out,"w"), indent=2)

    print(f"\n{'='*58}")
    print(f"  SUMMARY")
    print(f"{'='*58}")
    for key, res in results.items():
        if res: print(f"  {key:<30} macro-F1 = {res.get('macro_f1','N/A')}")
    print(f"\nSaved: {out}")
    print("\nInterpretation:")
    oracle = results.get("oracle_gt",{}).get("macro_f1")
    e2e    = results.get("e2e_predicted",{}).get("macro_f1")
    if oracle and e2e:
        gap = oracle - e2e
        print(f"  Oracle F1 = {oracle:.4f}  (GT descriptors, upper bound)")
        print(f"  E2E F1    = {e2e:.4f}  (predicted area, change=0)")
        print(f"  Gap       = {gap:.4f}  (includes change_rate=0 artefact)")
        print(f"  When change_rate is properly computed from temporal pairs,")
        print(f"  true E2E F1 will lie between {e2e:.3f} and {oracle:.3f}.")


if __name__ == "__main__": main()