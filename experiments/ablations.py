r"""
BhuViveka — Ablations + Calibration + XAI Quantitative + Multi-Seed (Points 5, 9, 10, 11)
===========================================================================================

Run as separate modes:
    uv run experiment_ablations.py --mode ablations   # Point 5
    uv run experiment_ablations.py --mode calibration # Point 10
    uv run experiment_ablations.py --mode xai         # Point 9
    uv run experiment_ablations.py --mode multiseed   # Point 11
"""
from __future__ import annotations
import argparse, csv, json, warnings, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT     = Path.cwd()
RISK_CSV = ROOT / "data" / "processed" / "risk_from_seg" / "risk_labeled_all.csv"
OUT_DIR  = ROOT / "data" / "results" / "ablations"
CLASSES  = ["LOW", "MEDIUM", "HIGH"]
TEST_EVENTS = ["sikkim_south_lhonak_2023",
               "uttarakhand_chamoli_2021", "uttarakhand_chorabari_2013",
               "uttarakhand_kedarnath_2013", "himachal_gepang_gath"]


def safe_load(path):
    p = ROOT / path
    if not p.exists(): return None
    try:
        a = np.load(p, allow_pickle=False)
        return np.nan_to_num(a, 0, 0, 0).astype(np.float32) if a.ndim == 3 else None
    except: return None


def to6(arr, sensor):
    out = np.zeros((6, arr.shape[1], arr.shape[2]), np.float32)
    if sensor == "LISS3" and arr.shape[0] >= 4: out[1:5] = arr[:4]
    elif sensor == "LISS4" and arr.shape[0] >= 3: out[1:4] = arr[:3]
    else: n = min(6, arr.shape[0]); out[:n] = arr[:n]
    return out


def load_base_models(device, use_lora=True, use_adapter=True):
    from train_seg import BhuVivekaSeg
    from train_stage2_grpo import ReasoningHead
    from prithvi_model import load_prithvi_backbone as _load
    from peft import get_peft_model, LoraConfig

    base, dim = _load("300M", ROOT/"models"/"prithvi-300m-tl")
    for p in base.parameters(): p.requires_grad = False

    if use_lora:
        tgt = [n for n,m in base.named_modules() if isinstance(m,nn.Linear) and n.endswith(".qkv")]
        cfg = LoraConfig(r=16, lora_alpha=32, target_modules=tgt or ["qkv"],
                         lora_dropout=0.05, bias="none", modules_to_save=["patch_embed"])
        backbone = get_peft_model(base, cfg)
    else:
        backbone = base  # frozen, no LoRA

    seg = BhuVivekaSeg(backbone, dim, use_spectral_adapter=use_adapter).to(device)
    seg.load_state_dict(torch.load(ROOT/"checkpoints"/"seg_300m"/"best.pt",
                        map_location="cpu",weights_only=False)["model_state_dict"], strict=False)
    seg.eval()

    reason = ReasoningHead(visual_dim=dim, physics_dim=3).to(device)
    reason.load_state_dict(torch.load(ROOT/"checkpoints"/"stage2_grpo"/"best.pt",
                           map_location="cpu",weights_only=False)["reasoning_state_dict"])
    reason.eval()
    return seg, reason, dim


def run_inference(seg, reason, rows, device, zero_physics_idx=None, max_n=300):
    """Run Stage 2 inference; zero_physics_idx omits that physics variable."""
    preds, labels, confs, logits_all = [], [], [], []
    from collections import defaultdict; count = defaultdict(int)
    for r in rows:
        ev = r.get("event","")
        if count[ev] >= max_n: continue
        arr = safe_load(r.get("npy_path",""))
        if arr is None: continue
        try:
            x = torch.from_numpy(to6(arr,r["sensor"])).float().unsqueeze(0).to(device)
            phy = [float(r.get("seg_area_km2",0)),
                   float(r.get("change_pct",0)),
                   float(r.get("dist_km",0))]
            if zero_physics_idx is not None:
                phy[zero_physics_idx] = 0.0
            phy_t = torch.tensor([phy]).float().to(device)
            with torch.no_grad():
                x_up = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False).unsqueeze(2)
                cls = seg.backbone(x_up)[:,0,:]
                logits, _ = reason(cls, phy_t)
                probs = F.softmax(logits,-1)[0].cpu().numpy()
            pred = CLASSES[int(logits.argmax(-1).item())]
            preds.append(pred); labels.append(r.get("risk_label","LOW"))
            confs.append(float(probs.max())); logits_all.append(probs)
            count[ev] += 1
        except: continue
    return preds, labels, np.array(confs), np.array(logits_all)


def m(preds, labels):
    if not preds: return {"macro_f1": 0, "bal_acc": 0}
    return {"macro_f1": round(f1_score(labels,preds,average="macro",labels=CLASSES,zero_division=0),4),
            "bal_acc": round(balanced_accuracy_score(labels,preds),4)}


# ─────────────────────────────────────────────────────────────────────────────
# POINT 5: ABLATIONS
# ─────────────────────────────────────────────────────────────────────────────
def run_ablations(device, rows):
    results = {}

    print("Full model (baseline)...")
    seg, reason, _ = load_base_models(device)
    p,l,_,_ = run_inference(seg, reason, rows, device)
    results["full_model"] = m(p,l); print(f"  {results['full_model']}")

    print("No LoRA (frozen Prithvi, no adapters)...")
    seg_nl, reason_nl, _ = load_base_models(device, use_lora=False)
    p,l,_,_ = run_inference(seg_nl, reason_nl, rows, device)
    results["no_lora"] = m(p,l); print(f"  {results['no_lora']}")

    print("No spectral adapter...")
    seg_na, reason_na, _ = load_base_models(device, use_adapter=False)
    p,l,_,_ = run_inference(seg_na, reason_na, rows, device)
    results["no_spectral_adapter"] = m(p,l); print(f"  {results['no_spectral_adapter']}")

    seg, reason, _ = load_base_models(device)
    for idx, name in [(0,"no_area"), (1,"no_change"), (2,"no_dist")]:
        print(f"Physics ablation: {name}...")
        p,l,_,_ = run_inference(seg, reason, rows, device, zero_physics_idx=idx)
        results[name] = m(p,l); print(f"  {results[name]}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# POINT 10: CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def reliability_diagram_data(confs, correct, n_bins=10):
    """Returns bin centres, mean_conf, mean_acc, count for each bin."""
    bins = np.linspace(0, 1, n_bins+1)
    centres, accs, mean_confs, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0: continue
        centres.append((lo+hi)/2)
        accs.append(correct[mask].mean())
        mean_confs.append(confs[mask].mean())
        counts.append(int(mask.sum()))
    return centres, mean_confs, accs, counts


def run_calibration(device, rows_temporal, rows_spatial):
    from scipy.special import log_softmax

    seg, reason, _ = load_base_models(device)
    results = {}

    for split_name, rows in [("temporal", rows_temporal), ("spatial", rows_spatial)]:
        if not rows: continue
        p, l, confs, probs = run_inference(seg, reason, rows, device)
        if not p: continue

        correct = np.array([int(pi==li) for pi,li in zip(p,l)], dtype=float)
        n = len(p)
        label_ids = np.array([CLASSES.index(li) if li in CLASSES else 0 for li in l])

        # ECE
        n_bins = 10
        edges = np.linspace(0, 1, n_bins+1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0: continue
            ece += mask.sum()/n * abs(confs[mask].mean() - correct[mask].mean())

        # Brier score (multi-class one-vs-rest)
        one_hot = np.zeros((n, 3))
        for i, li in enumerate(label_ids): one_hot[i, li] = 1.0
        brier = float(((probs - one_hot)**2).mean())

        # NLL
        nll = float(-np.log(probs[np.arange(n), label_ids] + 1e-8).mean())

        # Reliability data
        centres, mean_confs, mean_accs, counts = reliability_diagram_data(confs, correct)

        results[split_name] = {
            "n": n, "macro_f1": m(p,l)["macro_f1"],
            "ece": round(ece,4), "brier": round(brier,4), "nll": round(nll,4),
            "reliability_diagram": {
                "bin_centres": [round(c,3) for c in centres],
                "mean_confidence": [round(c,3) for c in mean_confs],
                "mean_accuracy": [round(a,3) for a in mean_accs],
                "counts": counts
            }
        }
        print(f"\n  {split_name}: ECE={ece:.4f}, Brier={brier:.4f}, NLL={nll:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# POINT 9: XAI QUANTITATIVE (Grad-CAM mass fraction in lake mask)
# ─────────────────────────────────────────────────────────────────────────────
def run_xai_quantitative(device, rows):
    """
    Computes fraction of Grad-CAM saliency mass that falls inside the Hi-MAG
    lake mask vs outside, and vs a random baseline.
    Metric: lake_fraction = sum(cam[mask==1]) / sum(cam)
    """
    seg, reason, _ = load_base_models(device)
    lake_fracs, random_fracs = [], []

    for r in rows:
        arr = safe_load(r.get("npy_path",""))
        mask_arr = safe_load(r.get("mask_path","")) if r.get("mask_path","") else None
        if arr is None or mask_arr is None: continue

        mask = (mask_arr[0] if mask_arr.ndim==3 else mask_arr) > 0.5
        if mask.sum() == 0: continue  # skip patches with no lake

        try:
            x = torch.from_numpy(to6(arr,r["sensor"])).float().unsqueeze(0).to(device)
            acts, grads = {}, {}
            def fwd(m,i,o): acts["v"] = o.detach()
            def bwd(m,gi,go): grads["v"] = go[0].detach()
            h1 = seg.seg.last_conv.register_forward_hook(fwd)
            h2 = seg.seg.last_conv.register_full_backward_hook(bwd)
            x_up = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False)
            out = seg(x_up); score = torch.sigmoid(out).mean()
            seg.zero_grad(); score.backward()
            h1.remove(); h2.remove()
            A, G = acts["v"], grads["v"]
            w = G.mean((2,3), keepdim=True)
            cam = F.relu((w*A).sum(1,keepdim=True))
            cam = F.interpolate(cam, size=(128,128), mode="bilinear", align_corners=False)
            cam = cam[0,0].detach().cpu().numpy()
            cam_total = cam.sum() + 1e-8
            # resize mask to 128x128 if needed
            if mask.shape != (128,128):
                mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask_t, size=(128,128)).squeeze().numpy() > 0.5

            lake_frac = float((cam * mask).sum() / cam_total)
            random_frac = float(mask.mean())  # expected fraction under uniform random
            lake_fracs.append(lake_frac)
            random_fracs.append(random_frac)
        except: continue

    if not lake_fracs:
        return {"error": "no valid patches with lake masks"}

    result = {
        "n_patches": len(lake_fracs),
        "mean_cam_lake_fraction": round(float(np.mean(lake_fracs)),4),
        "mean_random_baseline": round(float(np.mean(random_fracs)),4),
        "lift_over_random": round(float(np.mean(lake_fracs)/max(np.mean(random_fracs),1e-6)),4),
        "interpretation": (
            "lift_over_random > 1.0 means Grad-CAM concentrates on lake pixels "
            "more than expected by chance. Values near 1.0 indicate poor spatial grounding."
        )
    }
    print(f"\n  XAI: mean CAM lake fraction = {result['mean_cam_lake_fraction']:.3f}")
    print(f"       random baseline          = {result['mean_random_baseline']:.3f}")
    print(f"       lift over random         = {result['lift_over_random']:.3f}×")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POINT 11: MULTI-SEED
# ─────────────────────────────────────────────────────────────────────────────
def run_multiseed(device, rows, seeds=(42, 7, 13, 99, 2025)):
    """Run Stage 2 inference across multiple seeds; report mean ± std.
    Note: inference is deterministic; seeds affect Stage 2 training only.
    To fully implement this, retrain Stage 2 from each seed and load each ckpt.
    Here we demonstrate the evaluation framework."""
    print(f"\nMulti-seed evaluation framework (Point 11)")
    print(f"Seeds: {seeds}")
    print("NOTE: True multi-seed requires retraining Stage 2 from each seed.")
    print("      Loading the existing checkpoint for all seeds as a placeholder.")
    print("      Retrain with: uv run train_stage2_grpo.py --seed <N>")
    print("                    for seeds:", seeds)

    all_f1s, all_ba = [], []
    seg, _, _ = load_base_models(device)

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        from train_stage2_grpo import ReasoningHead
        reason = ReasoningHead(visual_dim=1024, physics_dim=3).to(device)
        ckpt_path = ROOT / "checkpoints" / "stage2_grpo" / f"seed_{seed}" / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = ROOT / "checkpoints" / "stage2_grpo" / "best.pt"
            print(f"  Seed {seed}: no dedicated checkpoint, using main ckpt (placeholder)")
        reason.load_state_dict(torch.load(ckpt_path,map_location="cpu",
                               weights_only=False)["reasoning_state_dict"])
        reason.eval()
        p,l,_,_ = run_inference(seg, reason, rows, device)
        r = m(p,l); all_f1s.append(r["macro_f1"]); all_ba.append(r["bal_acc"])
        print(f"  Seed {seed}: macro_f1={r['macro_f1']:.4f}, bal_acc={r['bal_acc']:.4f}")

    result = {
        "seeds": list(seeds),
        "macro_f1_mean": round(float(np.mean(all_f1s)),4),
        "macro_f1_std":  round(float(np.std(all_f1s)),4),
        "bal_acc_mean":  round(float(np.mean(all_ba)),4),
        "bal_acc_std":   round(float(np.std(all_ba)),4),
        "note": "std may be 0 if single checkpoint used; retrain per seed for true variance"
    }
    print(f"\n  macro-F1: {result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}")
    print(f"  bal-acc:  {result['bal_acc_mean']:.4f} ± {result['bal_acc_std']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablations","calibration","xai","multiseed","all"],
                    default="all")
    ap.add_argument("--max", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    from collections import defaultdict; count = defaultdict(int)
    test_rows = []
    for r in rows:
        ev = r.get("event","")
        if ev in TEST_EVENTS and count[ev] < args.max:
            test_rows.append(r); count[ev] += 1

    temporal = [r for r in test_rows if "lhonak_2023" in r.get("event","")]
    spatial  = [r for r in test_rows if "lhonak_2023" not in r.get("event","")]

    results = {}

    if args.mode in ("ablations","all"):
        print("\n=== ABLATIONS (Point 5) ===")
        results["ablations"] = run_ablations(device, test_rows)
        json.dump(results, open(OUT_DIR/"ablations.json","w"), indent=2)

    if args.mode in ("calibration","all"):
        print("\n=== CALIBRATION (Point 10) ===")
        results["calibration"] = run_calibration(device, temporal, spatial)
        json.dump(results, open(OUT_DIR/"calibration.json","w"), indent=2)

    if args.mode in ("xai","all"):
        print("\n=== XAI QUANTITATIVE (Point 9) ===")
        lake_rows = [r for r in test_rows if r.get("mask_path","")][:100]
        results["xai"] = run_xai_quantitative(device, lake_rows)
        json.dump(results, open(OUT_DIR/"xai_quantitative.json","w"), indent=2)

    if args.mode in ("multiseed","all"):
        print("\n=== MULTI-SEED (Point 11) ===")
        results["multiseed"] = run_multiseed(device, test_rows)
        json.dump(results, open(OUT_DIR/"multiseed.json","w"), indent=2)

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__": main()
