"""
BhuViveka — Temporal Sequence Figure (PRE → ON → POST)
========================================================
Publication Figure 8: Complete temporal progression showing:
  - PRE-event (2016): Baseline lake state
  - ON-event (2021): Lake growth/activity
  - POST-event (2023): Post-GLOF state

For each frame:
  Top: Input satellite image (RGB)
  Middle: Lake segmentation mask (blue overlay)
  Bottom: Physics reasoning + risk prediction

Generates publication-ready 3-panel figure.

Usage:
  uv run temporal_sequence_figure.py
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

GREEN, YELLOW, RED, CYAN = "\033[92m", "\033[93m", "\033[91m", "\033[96m"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"

def ok(m="OK"): print(f"  {GREEN}✓ {m}{RESET}")
def info(m): print(f"  {DIM}  {m}{RESET}")
def warn(m): print(f"  {YELLOW}⚠ {m}{RESET}")
def fatal(m): print(f"\n{RED}{BOLD}✗ STOP: {m}{RESET}"); sys.exit(1)

PROJECT_ROOT = Path.cwd()
FIG_DIR = PROJECT_ROOT / "figures" / "publication_ready"


def to_six_band(arr, sensor):
    H, W = arr.shape[1], arr.shape[2]
    out = np.zeros((6, H, W), dtype=np.float32)
    if sensor == "LISS3":
        out[1]=arr[0]; out[2]=arr[1]; out[3]=arr[2]; out[4]=arr[3]
    elif sensor == "LISS4":
        out[1]=arr[0]; out[2]=arr[1]; out[3]=arr[2]
    else:
        n=min(6,arr.shape[0]); out[:n]=arr[:n]
    return out


def arr_to_rgb(arr6):
    """Convert 6-band to RGB using R, G, NIR."""
    rgb = np.stack([arr6[2], arr6[1], arr6[3]], axis=-1)
    for c in range(3):
        ch = rgb[:, :, c]
        if (ch > 0).any():
            pmin, pmax = np.percentile(ch[ch > 0], [2, 98])
            rgb[:, :, c] = np.clip((ch - pmin) / max(pmax - pmin, 1e-6), 0, 1)
    return rgb


def load_models(device):
    from train_seg import BhuVivekaSeg
    from train_stage2_grpo import ReasoningHead
    from prithvi_model import load_prithvi_backbone as _load
    from peft import get_peft_model, LoraConfig
    import torch.nn as nn

    base, _ = _load("300M", PROJECT_ROOT / "models" / "prithvi-300m-tl")
    for p in base.parameters(): p.requires_grad = False

    tgt = [n for n,m in base.named_modules()
           if isinstance(m, nn.Linear) and n.endswith(".qkv")]
    if not tgt:
        tgt = [n for n,m in base.named_modules()
               if isinstance(m, nn.Linear) and "attn" in n.lower()]
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=tgt,
                     lora_dropout=0.05, bias="none",
                     modules_to_save=["patch_embed"])
    base_lora = get_peft_model(base, cfg)

    seg_model = BhuVivekaSeg(base_lora, base.embed_dim)
    seg_ckpt = torch.load(
        PROJECT_ROOT / "checkpoints" / "seg_300m" / "best.pt",
        map_location="cpu", weights_only=False)
    seg_model.load_state_dict(seg_ckpt["model_state_dict"])
    seg_model = seg_model.to(device).eval()

    reasoning = ReasoningHead(visual_dim=1024, physics_dim=3)
    reason_ckpt = torch.load(
        PROJECT_ROOT / "checkpoints" / "stage2_grpo" / "best.pt",
        map_location="cpu", weights_only=False)
    reasoning.load_state_dict(reason_ckpt["reasoning_state_dict"])
    reasoning = reasoning.to(device).eval()
    
    return seg_model, reasoning


def infer_patch(seg_model, reasoning_model, arr, sensor, area_km2, change_pct, dist_km, device):
    """Single patch inference."""
    arr6 = to_six_band(arr, sensor)
    x = torch.from_numpy(arr6).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        x_up = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        seg_logits = seg_model(x_up)
        seg_mask = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
        
        if x_up.dim() == 4:
            x_up = x_up.unsqueeze(2)
        tokens = seg_model.backbone(x_up)
        cls_feat = tokens[:, 0, :]
        
        phy = torch.tensor([[area_km2, change_pct, dist_km]], dtype=torch.float32).to(device)
        logits, _ = reasoning_model(cls_feat, phy)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        risk_pred_id = logits.argmax(dim=-1).item()
        risk_pred = ["LOW", "MEDIUM", "HIGH"][risk_pred_id]
        confidence = float(probs[risk_pred_id])
    
    return {
        "rgb": arr_to_rgb(arr6),
        "seg_mask": seg_mask,
        "risk_pred": risk_pred,
        "confidence": confidence,
        "probs": probs,
    }


def generate_temporal_figure(frames_data):
    """Generate temporal sequence figure (2-3 panels)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        warn("matplotlib not available")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    n_frames = len(frames_data)
    fig = plt.figure(figsize=(6*n_frames, 14))
    
    stage_labels = {"pre": "EARLY STAGE", "on": "ACTIVE PERIOD", "post": "RECENT STATE"}
    colors_stage = {"pre": "#3498db", "on": "#e67e22", "post": "#c0392b"}
    
    col_idx = 0
    for stage in ["pre", "on", "post"]:
        if stage not in frames_data:
            continue
        
        frame = frames_data[stage]
        color = colors_stage[stage]
        
        # Row 1: Input RGB image
        ax1 = plt.subplot(3, n_frames, col_idx + 1)
        ax1.imshow(frame["rgb"])
        ax1.axis("off")
        ax1.set_title(stage_labels[stage], fontweight="bold", fontsize=12, color=color)
        if col_idx == 0:
            ax1.set_ylabel("Input Image\n(R, G, NIR)", fontweight="bold", fontsize=11)
        
        # Row 2: Segmentation mask
        ax2 = plt.subplot(3, n_frames, col_idx + n_frames + 1)
        ax2.imshow(frame["rgb"], alpha=0.5)
        mask_im = ax2.imshow(frame["seg_mask"], cmap="Blues", alpha=0.8, vmin=0, vmax=1)
        ax2.axis("off")
        if col_idx == 0:
            ax2.set_ylabel("Lake Segmentation\n(Blue = detected)", fontweight="bold", fontsize=11)
        
        # Row 3: Physics + reasoning
        ax3 = plt.subplot(3, n_frames, col_idx + 2*n_frames + 1)
        ax3.axis("off")
        
        # Format reasoning text
        reasoning_text = f"""
{frame['risk_pred'].upper()} RISK
{'━' * 20}

Physics:
  Area: {frame['area_km2']:.3f} km²
  Change: {frame['change_pct']:+.1f}%
  Dist: {frame['dist_km']:.1f} km

Confidence:
  {frame['risk_pred']}: {frame['probs'][['LOW','MEDIUM','HIGH'].index(frame['risk_pred'])]:.1%}

Status: {'✓' if frame['correct'] else '✗'}
"""
        
        risk_color_map = {"LOW": "#2980b9", "MEDIUM": "#e67e22", "HIGH": "#c0392b"}
        bg_color = risk_color_map[frame['risk_pred']]
        
        ax3.text(0.05, 0.95, reasoning_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor=bg_color, alpha=0.15, 
                         edgecolor=bg_color, linewidth=2))
        
        if col_idx == 0:
            ax3.set_ylabel("Risk Assessment", fontweight="bold", fontsize=11)
        
        col_idx += 1
    
    title = f"BhuViveka Temporal Progression: Sikkim Lhonak Glacial Lake\n" \
            f"Stage 1 (Segmentation: IoU=0.293) + Stage 2 (Deep RL: F1=0.862)"
    
    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    out_path = FIG_DIR / "figure_8_temporal_sequence.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    ok(f"saved {out_path.name}")


def main():
    print(f"\n{BOLD}BhuViveka — Temporal Sequence Figure{RESET}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        fatal("CUDA required")
    
    print(f"{BOLD}Loading models{RESET}")
    seg_model, reasoning_model = load_models(device)
    ok("models loaded")
    
    print(f"\n{BOLD}Load temporal sequence patches{RESET}")
    risk_csv = PROJECT_ROOT / "data" / "processed" / "risk_from_seg" / "risk_labeled_all.csv"
    
    with open(risk_csv, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    
    # Group by temporal stages for Sikkim Lhonak (any year with area > 0)
    by_year = defaultdict(list)
    for row in all_rows:
        event = row.get("event", "")
        if "lhonak" in event.lower() and float(row.get("seg_area_km2", 0)) > 0.01:
            if "2016" in event:
                by_year["pre"].append(row)
            elif "2021" in event:
                by_year["on"].append(row)
            elif "2023" in event:
                by_year["post"].append(row)
    
    # If no 2021, use other years as middle frame
    if not by_year.get("on"):
        for row in all_rows:
            event = row.get("event", "")
            if "lhonak" in event.lower() and float(row.get("seg_area_km2", 0)) > 0.01:
                if "2019" in event or "2020" in event or "2022" in event:
                    by_year["on"].append(row)
    
    # If still missing stages, use any lhonak patches
    if not by_year.get("pre") or not by_year.get("post"):
        all_lhonak = [r for r in all_rows 
                      if "lhonak" in r.get("event", "").lower()
                      and float(r.get("seg_area_km2", 0)) > 0.01]
        if not by_year.get("pre"):
            by_year["pre"] = all_lhonak[:len(all_lhonak)//3]
        if not by_year.get("on"):
            by_year["on"] = all_lhonak[len(all_lhonak)//3:2*len(all_lhonak)//3]
        if not by_year.get("post"):
            by_year["post"] = all_lhonak[2*len(all_lhonak)//3:]
    
    # Select best examples (prefer HIGH risk, but accept any with real area)
    frames_data = {}
    stage_names = {"pre": "EARLY", "on": "ACTIVE", "post": "RECENT"}
    
    for stage, stage_label in stage_names.items():
        candidates = by_year.get(stage, [])
        
        if not candidates:
            warn(f"No {stage} candidates found")
            continue
        
        # Sort by area (largest lakes most visible)
        candidates_sorted = sorted(
            candidates,
            key=lambda r: float(r.get("seg_area_km2", 0)),
            reverse=True
        )
        
        selected = candidates_sorted[0]
        
        info(f"{stage_label}: {selected['patch_id'][:40]} (area={float(selected.get('seg_area_km2', 0)):.3f} km²)")
        
        arr = np.load(PROJECT_ROOT / selected["npy_path"])
        area = float(selected.get("seg_area_km2", 0))
        change = float(selected.get("change_pct", 0))
        dist = float(selected.get("dist_km", 0))
        
        result = infer_patch(seg_model, reasoning_model, arr,
                            selected["sensor"], area, change, dist, device)
        
        result.update({
            "patch_id": selected["patch_id"],
            "area_km2": area,
            "change_pct": change,
            "dist_km": dist,
            "true_label": selected.get("risk_label", "?"),
            "correct": result["risk_pred"] == selected.get("risk_label", "?"),
        })
        
        frames_data[stage] = result
    
    if len(frames_data) < 2:
        fatal("Could not find enough temporal stages (need at least 2)")
    
    # Print frame info
    print(f"\n{BOLD}Selected Temporal Frames:{RESET}\n")
    for stage in ["pre", "on", "post"]:
        if stage in frames_data:
            f = frames_data[stage]
            print(f"{stage.upper()}:")
            print(f"  Patch:       {f['patch_id']}")
            print(f"  Area:        {f['area_km2']:.3f} km²")
            print(f"  Change:      {f['change_pct']:+.1f}%")
            print(f"  Prediction:  {f['risk_pred']} (conf: {f['confidence']:.2f})")
            print(f"  Ground truth: {f['true_label']}")
            print(f"  Status:      {'✓ CORRECT' if f['correct'] else '✗ WRONG'}\n")
    
    # Generate figure
    print(f"{BOLD}Generating temporal sequence figure{RESET}")
    generate_temporal_figure(frames_data)
    
    print(f"\n{GREEN}{BOLD}Done!{RESET}")
    print(f"  Figure 8 saved to: {FIG_DIR}/figure_8_temporal_sequence.png\n")


if __name__ == "__main__":
    main()