"""
BhuViveka-VLM — Self-contained inference
=========================================
No dependency on the training code. Loads two checkpoints from ./checkpoints
and runs the full pipeline on a single satellite patch (.npy).

  from inference import load_models, predict
  models = load_models()
  result = predict(models, "samples/xxx.npy", sensor="LISS3",
                   change_pct=12.0, dist_km=3.4)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import BhuVivekaSeg, ReasoningHead, build_lora_backbone

HERE = Path(__file__).resolve().parent
SEG_CKPT  = HERE / "checkpoints" / "seg_300m_best.pt"
GRPO_CKPT = HERE / "checkpoints" / "grpo_best.pt"

CLASSES = ["LOW", "MEDIUM", "HIGH"]
PIXEL_M2 = {"LISS3": 23.5 ** 2, "LISS4": 5.8 ** 2}

REASONING_TEMPLATE = (
    "Lake area: {area:.3f} km2. Annual change: {change:+.1f}%. "
    "Distance to known GLOF: {dist:.1f} km. Risk assessment: {risk}."
)


# ─────────────────────────────────────────────
# Band / image helpers
# ─────────────────────────────────────────────
def to_six_band(arr: np.ndarray, sensor: str) -> np.ndarray:
    """Pad LISS-3/4 arrays to Prithvi's 6-band HLS layout."""
    out = np.zeros((6, arr.shape[1], arr.shape[2]), dtype=np.float32)
    if sensor == "LISS3" and arr.shape[0] >= 4:
        out[1:5] = arr[:4]
    elif sensor == "LISS4" and arr.shape[0] >= 3:
        out[1:4] = arr[:3]
    else:
        n = min(6, arr.shape[0])
        out[:n] = arr[:n]
    return out


def arr_to_rgb(arr6: np.ndarray) -> np.ndarray:
    """6-band -> display RGB (R=ch2, G=ch1, B=ch3/NIR for lake contrast)."""
    rgb = np.stack([arr6[2], arr6[1], arr6[3]], axis=-1)
    for c in range(3):
        ch = rgb[:, :, c]
        if (ch > 0).any():
            lo, hi = np.percentile(ch[ch > 0], [2, 98])
            rgb[:, :, c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_models(device: str | None = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    backbone_lora, embed_dim = build_lora_backbone()

    # Stage 1 — segmentation (checkpoint embeds the full backbone weights)
    seg = BhuVivekaSeg(backbone_lora, embed_dim).to(dev).eval()
    seg_state = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    seg.load_state_dict(seg_state["model_state_dict"])

    # Stage 2 — reasoning head
    rh = ReasoningHead(visual_dim=embed_dim, physics_dim=3).to(dev).eval()
    rh_state = torch.load(GRPO_CKPT, map_location="cpu", weights_only=False)
    rh.load_state_dict(rh_state["reasoning_state_dict"])

    return {"seg": seg, "rh": rh, "embed_dim": embed_dim, "device": dev}


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(models: dict, npy_path, sensor: str,
            change_pct: float = 0.0, dist_km: float = 99.0) -> dict:
    """Run the full pipeline on one patch. Lake area is computed from the
    predicted mask (not taken from any label)."""
    dev = models["device"]
    seg, rh = models["seg"], models["rh"]

    arr = np.load(npy_path, allow_pickle=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    arr6 = to_six_band(arr, sensor)
    rgb_img = arr_to_rgb(arr6)

    x = torch.from_numpy(arr6).unsqueeze(0).to(dev)                 # (1,6,H,W)
    x_up = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    with torch.no_grad():
        # Stage 1 — segmentation
        seg_logits = seg(x_up)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()

        # CLS token (Stage-2 encoder path was trained at 224)
        x_enc = x_up.unsqueeze(2)                                   # (1,6,1,224,224)
        tokens = seg.backbone(x_enc)
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        cls_feat = tokens[:, 0]

        pred_area = float((seg_prob > 0.5).sum() * PIXEL_M2.get(sensor, 23.5 ** 2) / 1e6)

        # Stage 2 — risk
        phy = torch.tensor([[pred_area, change_pct, dist_km]], dtype=torch.float32).to(dev)
        logits, _ = rh(cls_feat, phy)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    risk_id = int(probs.argmax())
    risk_cls = CLASSES[risk_id]

    return {
        "rgb_image": rgb_img,
        "seg_mask": seg_prob,
        "seg_mask_bin": (seg_prob > 0.5).astype(np.uint8),
        "pred_area_km2": pred_area,
        "risk_class": risk_cls,
        "risk_id": risk_id,
        "confidence": float(probs[risk_id]),
        "probs": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
        "reasoning": REASONING_TEMPLATE.format(
            area=pred_area, change=change_pct, dist=dist_km, risk=risk_cls),
        "physics": {"area_km2": pred_area, "change_pct": change_pct, "dist_km": dist_km},
    }
