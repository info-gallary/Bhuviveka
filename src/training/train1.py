r"""
BhuViveka-VLM — Unified Vision-Language Model with Meta-Learning + Physics RL + XAI
====================================================================================
A single end-to-end model that, from one LISS satellite patch, produces:
  (1) a glacial-lake segmentation mask,
  (2) a physics-grounded reasoning sentence (mini autoregressive decoder),
  (3) a GLOF risk class + value estimate (actor-critic),
trained with:
  - Reptile meta-learning (fast adaptation across glacial events/regions),
  - physics-informed Deep RL reward (area / change / GLOF-proximity),
  - a cross-head consistency loss (text area == segmented area),
and explained with:
  - Grad-CAM, attention rollout, and physics-consistency XAI.

----------------------------------------------------------------------------------
HONEST SCOPE NOTE
This combines four hard ingredients (multitask + meta-learning + RL + a language
decoder). Such combinations are powerful but unstable; expect to tune loss weights
and inner/outer LRs. The language head is a SMALL grounded decoder, not a pretrained
LLM. Validate each head alone before trusting the joint model, and benchmark against
your existing two-stage pipeline (Seg IoU 0.293 / RL F1 0.862) — the unified model is
only worth shipping if it matches or beats that.
----------------------------------------------------------------------------------

ARCHITECTURE
  Prithvi-EO (frozen) + LoRA
        |-- patch tokens --> Segmentation head     --> mask (B,1,128,128)
        |-- CLS token  -----> Language head (dec)   --> reasoning tokens
        \-- CLS + physics --> Actor-Critic head     --> risk logits + V(s)

USAGE
  uv run train_bhuviveka_vlm.py --smoke-test          # mock backbone, verify wiring
  uv run train_bhuviveka_vlm.py --train --epochs 20   # real training
  uv run train_bhuviveka_vlm.py --xai --image P.npy   # explain one patch
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

G, Y, R, C = "\033[92m", "\033[93m", "\033[91m", "\033[96m"
B, D, X = "\033[1m", "\033[2m", "\033[0m"
def ok(m="OK"): print(f"  {G}\u2713 {m}{X}")
def info(m): print(f"  {D}  {m}{X}")
def warn(m): print(f"  {Y}\u26a0 {m}{X}")
def fatal(m): print(f"\n{R}{B}\u2717 STOP: {m}{X}"); sys.exit(1)
def banner(t):
    bar = "=" * 70
    print(f"\n{C}{bar}\n  {B}{t}{X}{C}\n{bar}{X}")

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path.cwd()
BACKBONE_DIR = PROJECT_ROOT / "models" / "prithvi-300m-tl"
RISK_CSV     = PROJECT_ROOT / "data" / "processed" / "risk_from_seg" / "risk_labeled_all.csv"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "vlm_meta"

EMBED_DIM   = 1024
IMG_MODEL   = 224
VOCAB_SIZE  = 32
MAX_TEXT    = 48
N_ACTIONS   = 3
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]

# Loss weights (TUNE THESE)
W_SEG, W_LANG, W_POLICY, W_VALUE, W_CONSIST, W_ENTROPY = 0.45, 0.15, 0.20, 0.10, 0.07, 0.01

# Reptile meta-learning
INNER_STEPS = 5
LR_INNER    = 1e-4
LR_META     = 0.30      # interpolation toward adapted params
LR_HEADS    = 1e-4

TRAIN_EVENT = "sikkim_south_lhonak_2016"
META_EVENTS = ["uttarakhand_chamoli_2021", "uttarakhand_chorabari_2013",
               "uttarakhand_kedarnath_2013", "himachal_gepang_gath"]


# ============================================================
# TOKENIZER (grounded, compact)
# ============================================================
class Tok:
    PAD, BOS, EOS, UNK, NUM = 0, 1, 2, 3, 4
    def __init__(self):
        vocab = ["[PAD]","[BOS]","[EOS]","[UNK]","[NUM]",
                 "lake","area","km2","annual","change","percent",
                 "distance","to","known","glof","risk","is",
                 "high","medium","low","stable","growing","shrinking",
                 "near","far","proximal","large","small",".",",","and","with"]
        self.w2i = {w:i for i,w in enumerate(vocab)}
        self.i2w = {i:w for w,i in self.w2i.items()}
        self.size = len(vocab)
    def encode(self, text):
        toks=[self.BOS]
        for w in text.lower().replace("."," .").replace(","," ,").split():
            if w in self.w2i: toks.append(self.w2i[w])
            elif any(ch.isdigit() for ch in w): toks.append(self.NUM)
            else: toks.append(self.UNK)
        toks.append(self.EOS)
        return toks[:MAX_TEXT] + [self.PAD]*(MAX_TEXT-len(toks))
    def decode(self, ids):
        return " ".join(self.i2w.get(int(t),"?") for t in ids
                        if int(t) not in (self.PAD,self.BOS,self.EOS))


def physics_to_sentence(area, change, dist, risk):
    trend = "growing" if change > 5 else "shrinking" if change < -5 else "stable"
    prox  = "proximal" if dist < 10 else "near" if dist < 50 else "far"
    size  = "large" if area > 0.15 else "small"
    return (f"lake area is {size} {area:.2f} km2 annual change {change:.0f} percent "
            f"{trend} distance to known glof {dist:.0f} {prox} risk is {risk.lower()}")


# ============================================================
# HEADS
# ============================================================
class SegHead(nn.Module):
    def __init__(self, dim=EMBED_DIM, grid=14):
        super().__init__()
        self.grid = grid
        self.dec = nn.Sequential(
            nn.Conv2d(dim, 256, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, 2), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, 2),  nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),    nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self.last_conv = self.dec[-1]   # Grad-CAM hook target
    def forward(self, patch_tokens):
        Bn, Np, Dn = patch_tokens.shape
        feat = patch_tokens.permute(0,2,1).reshape(Bn, Dn, self.grid, self.grid)
        out = self.dec(feat)
        return F.interpolate(out, size=(128,128), mode="bilinear", align_corners=False)


class LangHead(nn.Module):
    """Small autoregressive decoder conditioned on visual+physics context."""
    def __init__(self, dim=EMBED_DIM, vocab=VOCAB_SIZE, hid=256):
        super().__init__()
        self.hid = hid
        self.emb = nn.Embedding(vocab, hid)
        self.pos = nn.Parameter(torch.randn(1, MAX_TEXT+1, hid) * 0.02)
        self.vis_proj = nn.Linear(dim, hid)
        self.phy_proj = nn.Linear(3, hid)
        layer = nn.TransformerEncoderLayer(hid, nhead=4, dim_feedforward=512,
                                           batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hid, vocab)
    def forward(self, cls_feat, physics, input_ids):
        Bn = cls_feat.size(0)
        ctx = (self.vis_proj(cls_feat) + self.phy_proj(physics)).unsqueeze(1)  # (B,1,hid)
        x = self.emb(input_ids)                                                # (B,T,hid)
        x = torch.cat([ctx, x], dim=1) + self.pos[:, :x.size(1)+1]
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.tr(x, mask=mask)
        return self.head(x[:, 1:])                                             # drop ctx slot


class ActorCritic(nn.Module):
    def __init__(self, dim=EMBED_DIM, phy=3, hid=256, n=N_ACTIONS):
        super().__init__()
        self.vis = nn.Linear(dim, hid); self.phy = nn.Linear(phy, hid)
        self.shared = nn.Sequential(nn.Linear(hid*2, hid), nn.ReLU(),
                                    nn.Dropout(0.1), nn.Linear(hid, hid), nn.ReLU())
        self.policy = nn.Linear(hid, n)
        self.value  = nn.Linear(hid, 1)
    def forward(self, cls_feat, physics):
        h = self.shared(torch.cat([self.vis(cls_feat), self.phy(physics)], -1))
        return self.policy(h), self.value(h)


class BhuVivekaVLM(nn.Module):
    def __init__(self, backbone, dim=EMBED_DIM):
        super().__init__()
        self.backbone = backbone
        self.seg = SegHead(dim)
        self.lang = LangHead(dim)
        self.ac = ActorCritic(dim)
        self.tok = Tok()
    def encode(self, img):
        if img.shape[-1] != IMG_MODEL:
            img = F.interpolate(img, size=(IMG_MODEL, IMG_MODEL),
                                mode="bilinear", align_corners=False)
        if img.dim() == 4:
            img = img.unsqueeze(2)            # Prithvi expects a time dim
        tokens = self.backbone(img)           # (B, N+1, dim)
        return tokens[:, 0, :], tokens[:, 1:, :]
    def forward(self, img, physics, text_ids=None):
        cls, patches = self.encode(img)
        seg_logits = self.seg(patches)
        risk_logits, value = self.ac(cls, physics)
        lang_logits = self.lang(cls, physics, text_ids) if text_ids is not None else None
        return {"seg": seg_logits, "risk": risk_logits, "value": value, "lang": lang_logits,
                "cls": cls}
    @torch.no_grad()
    def generate(self, img, physics, max_len=MAX_TEXT, temp=0.7):
        cls, _ = self.encode(img)
        ids = torch.full((img.size(0), 1), self.tok.BOS, device=img.device, dtype=torch.long)
        for _ in range(max_len):
            logits = self.lang(cls, physics, ids)
            nxt = torch.multinomial(F.softmax(logits[:, -1]/temp, -1), 1)
            ids = torch.cat([ids, nxt], 1)
            if (nxt == self.tok.EOS).all(): break
        return ids


# ============================================================
# LOSSES + PHYSICS REWARD
# ============================================================
def seg_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    p = torch.sigmoid(pred)
    inter = (p*target).sum((1,2,3))
    dice = 1 - (2*inter+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    return bce + dice.mean()

def lang_loss(logits, target_ids):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target_ids.reshape(-1), ignore_index=Tok.PAD)

def consistency_loss(seg_logits, stated_area_km2, area_scale=0.3):
    pred_frac = (torch.sigmoid(seg_logits) > 0.5).float().mean((1,2,3))
    target_frac = (stated_area_km2 / area_scale).clamp(0, 1)
    return F.mse_loss(pred_frac, target_frac)

def physics_reward(area, change, dist, pred_cls, true_cls):
    """Vectorized physics-informed reward (numpy floats in, np array out)."""
    r = np.where(pred_cls == true_cls, 1.0, -1.0)
    # consistency bonuses
    high_ok = (pred_cls=="HIGH") & (area>0.1) & (change>15)
    high_bad= (pred_cls=="HIGH") & ((area<0.05)|(change<5))
    med_ok  = (pred_cls=="MEDIUM") & (((area>=0.05)&(area<=0.15))|((change>=5)&(change<=15)))
    low_ok  = (pred_cls=="LOW") & (area<0.05) & (change<5)
    r = r + 0.3*high_ok - 0.2*high_bad + 0.2*med_ok + 0.3*low_ok
    # proximity scaling
    r = np.where(dist>50, r*0.7, np.where(dist<5, r*1.2, r))
    return r.astype(np.float32)

def multitask_loss(model, out, batch, device):
    """Combined VLM + RL loss for one batch. Returns (loss, logdict)."""
    seg_t = batch["mask"].to(device)
    txt_t = batch["text_ids"].to(device)
    area  = batch["area"].cpu().numpy()
    change= batch["change"].cpu().numpy()
    dist  = batch["dist"].cpu().numpy()
    true  = np.array(batch["risk_name"])

    L_seg = seg_loss(out["seg"], seg_t)
    L_lang = lang_loss(out["lang"], txt_t) if out["lang"] is not None else torch.tensor(0., device=device)
    L_con = consistency_loss(out["seg"], batch["area"].to(device))

    # ---- RL: sample actions, reward, advantage, policy gradient ----
    logits, value = out["risk"], out["value"].squeeze(-1)
    probs = F.softmax(logits, -1)
    logp_all = F.log_softmax(logits, -1)
    actions = torch.multinomial(probs, 1).squeeze(-1)
    pred_names = np.array(CLASS_NAMES)[actions.cpu().numpy()]
    reward = torch.tensor(physics_reward(area, change, dist, pred_names, true), device=device)
    advantage = reward - value
    logp_a = logp_all[torch.arange(len(actions)), actions]
    L_policy = -(logp_a * advantage.detach()).mean()
    L_value  = (advantage**2).mean()
    entropy  = -(probs * logp_all).sum(-1).mean()

    loss = (W_SEG*L_seg + W_LANG*L_lang + W_POLICY*L_policy
            + W_VALUE*L_value + W_CONSIST*L_con - W_ENTROPY*entropy)
    log = {"seg":float(L_seg), "lang":float(L_lang), "policy":float(L_policy),
           "value":float(L_value), "consist":float(L_con),
           "entropy":float(entropy), "reward":float(reward.mean())}
    return loss, log


# ============================================================
# DATA
# ============================================================
def to_six_band(arr, sensor):
    H, W = arr.shape[1], arr.shape[2]
    out = np.zeros((6,H,W), np.float32)
    if sensor=="LISS3" and arr.shape[0]>=4: out[1:5]=arr[:4]
    elif sensor=="LISS4" and arr.shape[0]>=3: out[1:4]=arr[:3]
    else:
        n=min(6,arr.shape[0]); out[:n]=arr[:n]
    return out

def safe_load(path):
    p = PROJECT_ROOT / path
    if not p.exists(): return None
    try:
        a = np.load(p, allow_pickle=False)
        if a.ndim!=3 or a.size==0: return None
        return np.nan_to_num(a, 0,0,0).astype(np.float32)
    except Exception:
        return None

class EventDataset:
    """Lightweight per-event dataset (lazy, corruption-safe)."""
    def __init__(self, rows, tok):
        self.rows = rows; self.tok = tok
    def __len__(self): return len(self.rows)
    def sample_batch(self, bs, device):
        items, tries = [], 0
        while len(items) < bs and tries < bs*5:
            tries += 1
            r = self.rows[np.random.randint(len(self.rows))]
            arr = safe_load(r.get("npy_path",""))
            if arr is None: continue
            mp = PROJECT_ROOT / r.get("mask_path","")
            mask = safe_load(r.get("mask_path","")) if mp.exists() else None
            if mask is None:
                mask = np.zeros((1,128,128), np.float32)
            else:
                mask = mask[:1] if mask.ndim==3 else mask[None]
                if mask.shape[-1]!=128:
                    mask = np.array(F.interpolate(torch.from_numpy(mask)[None].float(),
                                    size=(128,128)).squeeze(0))
            area=float(r.get("seg_area_km2",0)); change=float(r.get("change_pct",0))
            dist=float(r.get("dist_km",0)); risk=r.get("risk_label","LOW")
            text = physics_to_sentence(area, change, dist, risk)
            items.append((to_six_band(arr, r["sensor"]), mask,
                          area, change, dist, risk, self.tok.encode(text)))
        if not items: return None
        imgs   = torch.from_numpy(np.stack([i[0] for i in items])).float().to(device)
        masks  = torch.from_numpy(np.stack([i[1] for i in items])).float().to(device)
        return {
            "img": imgs, "mask": masks,
            "area":   torch.tensor([i[2] for i in items]).float(),
            "change": torch.tensor([i[3] for i in items]).float(),
            "dist":   torch.tensor([i[4] for i in items]).float(),
            "risk_name": [i[5] for i in items],
            "text_ids": torch.tensor([i[6] for i in items]).long().to(device),
            "physics": torch.tensor([[i[2],i[3],i[4]] for i in items]).float().to(device),
        }


# ============================================================
# REPTILE META-TRAINING
# ============================================================
def trainable_state(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()
            if model.get_parameter(k).requires_grad} if False else \
           {k:v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}

def load_trainable(model, snap):
    with torch.no_grad():
        for k,v in model.named_parameters():
            if k in snap: v.copy_(snap[k])

def reptile_meta_train(model, tasks, device, meta_iters=200, bs=4):
    """Reptile: adapt on each task, then move init toward adapted params."""
    inner_opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=LR_INNER)
    history = []
    for it in range(1, meta_iters+1):
        task = tasks[np.random.randint(len(tasks))]
        meta_snap = trainable_state(model)            # \theta

        last_log = {}
        for _ in range(INNER_STEPS):                   # inner adaptation
            batch = task.sample_batch(bs, device)
            if batch is None: break
            out = model(batch["img"], batch["physics"], batch["text_ids"])
            loss, last_log = multitask_loss(model, out, batch, device)
            inner_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 1.0)
            inner_opt.step()

        # Reptile meta-update:  \theta <- \theta + eps (\phi - \theta)
        with torch.no_grad():
            for k,p in model.named_parameters():
                if p.requires_grad and k in meta_snap:
                    adapted = p.detach()
                    p.copy_(meta_snap[k] + LR_META * (adapted - meta_snap[k]))

        if it % 10 == 0 and last_log:
            history.append(last_log)
            print(f"  meta-iter {it:4d} | seg {last_log['seg']:.3f} "
                  f"policy {last_log['policy']:.3f} reward {last_log['reward']:+.3f} "
                  f"lang {last_log['lang']:.3f}")
    return history


# ============================================================
# XAI
# ============================================================
def gradcam_seg(model, img, physics, device):
    """Grad-CAM on the segmentation head's last conv."""
    model.eval()
    acts, grads = {}, {}
    def fwd_hook(m,i,o): acts["v"] = o.detach()
    def bwd_hook(m,gi,go): grads["v"] = go[0].detach()
    h1 = model.seg.last_conv.register_forward_hook(fwd_hook)
    h2 = model.seg.last_conv.register_full_backward_hook(bwd_hook)
    out = model(img.to(device), physics.to(device))
    score = torch.sigmoid(out["seg"]).mean()
    model.zero_grad(); score.backward()
    h1.remove(); h2.remove()
    A, Gd = acts["v"], grads["v"]                  # (B,C,h,w)
    w = Gd.mean((2,3), keepdim=True)
    cam = F.relu((w*A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(128,128), mode="bilinear", align_corners=False)
    cam = cam / (cam.amax((2,3), keepdim=True) + 1e-6)
    return cam.squeeze(1).cpu().numpy()

def attention_rollout(model, img, device):
    """Attention rollout across backbone blocks (best-effort: needs attn weights)."""
    attns = []
    handles = []
    def make_hook():
        def hook(m, inp, out):
            # Try to recover attention weights from common module shapes
            if isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]):
                attns.append(out[1].detach())
        return hook
    for name, m in model.backbone.named_modules():
        if name.endswith("attn") or "attention" in name.lower():
            handles.append(m.register_forward_hook(make_hook()))
    with torch.no_grad():
        model.encode(img.to(device))
    for h in handles: h.remove()
    if not attns:
        return None   # backbone doesn't expose attn weights in this form
    result = torch.eye(attns[0].size(-1), device=attns[0].device)
    for a in attns:
        a = a.mean(1) if a.dim()==4 else a            # avg heads
        a = a + torch.eye(a.size(-1), device=a.device)
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    cls_to_patch = result[:, 0, 1:]                   # (B, N)
    n = int(cls_to_patch.size(-1) ** 0.5)
    grid = cls_to_patch.reshape(-1, n, n).cpu().numpy()
    return grid

def physics_consistency_report(model, img, physics, stated_area, device):
    with torch.no_grad():
        out = model(img.to(device), physics.to(device))
        pred_frac = float((torch.sigmoid(out["seg"])>0.5).float().mean())
        measured_area = pred_frac * 0.3   # inverse of area_scale heuristic
        risk = CLASS_NAMES[int(out["risk"].argmax(-1))]
    return {"stated_area_km2": float(stated_area),
            "measured_area_km2": round(measured_area,3),
            "abs_error_km2": round(abs(float(stated_area)-measured_area),3),
            "predicted_risk": risk}


# ============================================================
# BACKBONE LOADERS
# ============================================================
def load_real_backbone():
    from prithvi_model import load_prithvi_backbone as _load
    from peft import get_peft_model, LoraConfig
    base, _ = _load("300M", BACKBONE_DIR)
    for p in base.parameters(): p.requires_grad = False
    tgt = [n for n,m in base.named_modules()
           if isinstance(m, nn.Linear) and n.endswith(".qkv")]
    if not tgt:
        tgt = [n for n,m in base.named_modules()
               if isinstance(m, nn.Linear) and "attn" in n.lower()]
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=tgt,
                     lora_dropout=0.05, bias="none", modules_to_save=["patch_embed"])
    return get_peft_model(base, cfg), base.embed_dim

class MockBackbone(nn.Module):
    """Tiny stand-in to smoke-test wiring without Prithvi weights."""
    def __init__(self, dim=EMBED_DIM, grid=14):
        super().__init__()
        self.dim, self.n = dim, grid*grid
        self.proj = nn.Conv2d(6, dim, kernel_size=16, stride=16)
        self.cls = nn.Parameter(torch.randn(1,1,dim)*0.02)
    def forward(self, img):
        if img.dim()==5: img = img[:, :, 0]            # drop time dim
        f = self.proj(img)                             # (B, dim, 14, 14)
        f = f.flatten(2).transpose(1,2)                # (B, 196, dim)
        cls = self.cls.expand(f.size(0), -1, -1)
        return torch.cat([cls, f], 1)


# ============================================================
# TRAIN / SMOKE / XAI ENTRYPOINTS
# ============================================================
def build_tasks(device):
    if not RISK_CSV.exists():
        fatal(f"missing {RISK_CSV} — run risk_from_seg.py first")
    with open(RISK_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tok = Tok()
    train_rows = [r for r in rows if r.get("event")==TRAIN_EVENT]
    tasks = [EventDataset(train_rows, tok)] if train_rows else []
    for ev in META_EVENTS:
        er = [r for r in rows if r.get("event")==ev]
        if er: tasks.append(EventDataset(er, tok))
    info(f"{len(tasks)} task-events for meta-training")
    return tasks

def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu": fatal("CUDA required for real training")
    banner("Load backbone (Prithvi + LoRA)")
    backbone, dim = load_real_backbone(); ok("backbone ready")
    model = BhuVivekaVLM(backbone, dim).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"BhuViveka-VLM built — {ntr/1e6:.1f}M trainable params")
    banner("Reptile meta-training (physics-RL inner loop)")
    tasks = build_tasks(device)
    if not tasks: fatal("no task data")
    hist = reptile_meta_train(model, tasks, device,
                              meta_iters=args.epochs*10, bs=args.batch)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, CKPT_DIR / "best.pt")
    ok(f"saved {CKPT_DIR/'best.pt'}")

def run_smoke():
    banner("SMOKE TEST — mock backbone, verify all components wire up")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"device: {device}")
    model = BhuVivekaVLM(MockBackbone().to(device)).to(device)
    # mark backbone frozen, heads trainable (mimics real setup)
    for p in model.backbone.parameters(): p.requires_grad = True  # mock is trainable
    bs = 4
    img = torch.randn(bs,6,128,128, device=device)
    physics = torch.tensor([[0.2,30,3.0],[0.03,-2,60],[0.1,12,8],[0.5,80,1]],
                           device=device).float()
    text_ids = torch.randint(5, VOCAB_SIZE, (bs, MAX_TEXT), device=device)
    text_ids[:,0] = Tok.BOS

    out = model(img, physics, text_ids)
    assert out["seg"].shape == (bs,1,128,128), out["seg"].shape
    assert out["risk"].shape == (bs,3)
    assert out["lang"].shape == (bs, MAX_TEXT, VOCAB_SIZE)
    ok(f"forward shapes OK — seg{tuple(out['seg'].shape)} risk{tuple(out['risk'].shape)} lang{tuple(out['lang'].shape)}")

    batch = {"mask":(torch.rand(bs,1,128,128, device=device)>0.5).float(),
             "text_ids":text_ids, "area":physics[:,0].cpu(),
             "change":physics[:,1].cpu(), "dist":physics[:,2].cpu(),
             "risk_name":["HIGH","LOW","MEDIUM","HIGH"]}
    loss, log = multitask_loss(model, out, batch, device)
    loss.backward()
    ok(f"multitask loss = {float(loss):.3f}  components: " +
       " ".join(f"{k}={v:.2f}" for k,v in log.items()))

    # mini Reptile step
    tok = Tok()
    fake_rows = [{"npy_path":"__none__","mask_path":"__none__","sensor":"LISS3",
                  "seg_area_km2":"0.2","change_pct":"30","dist_km":"3",
                  "risk_label":"HIGH"} for _ in range(8)]
    # patch safe_load to synthesize arrays for smoke
    global safe_load
    _orig = safe_load
    def fake_load(path):
        return np.random.rand(4,128,128).astype(np.float32)
    safe_load = fake_load
    task = EventDataset(fake_rows, tok)
    hist = reptile_meta_train(model, [task], device, meta_iters=20, bs=4)
    safe_load = _orig
    ok(f"Reptile meta-loop ran ({len(hist)} logged iters)")

    # XAI
    cam = gradcam_seg(model, img, physics, device)
    ok(f"Grad-CAM OK — shape {cam.shape}, range [{cam.min():.2f},{cam.max():.2f}]")
    roll = attention_rollout(model, img, device)
    if roll is None:
        warn("attention rollout: mock backbone exposes no attn weights (expected). "
             "On real Prithvi it hooks block attentions.")
    else:
        ok(f"attention rollout OK — shape {roll.shape}")
    rep = physics_consistency_report(model, img[:1], physics[:1], 0.2, device)
    ok(f"physics-consistency XAI OK — {rep}")

    txt = model.generate(img[:1], physics[:1])
    ok(f"language generation OK — sample: \"{model.tok.decode(txt[0].tolist())[:60]}...\"")
    print(f"\n{G}{B}SMOKE TEST PASSED — architecture wiring is sound.{X}")
    print(f"{D}Next: run with real Prithvi via  --train{X}\n")

def run_xai(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cpu": fatal("CUDA required")
    backbone, dim = load_real_backbone()
    model = BhuVivekaVLM(backbone, dim).to(device)
    ck = CKPT_DIR / "best.pt"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"])
        ok("loaded VLM checkpoint")
    else:
        warn("no checkpoint — running XAI on untrained weights")
    arr = safe_load(args.image) if args.image else None
    if arr is None: fatal("provide --image PATH to a valid .npy patch")
    x = torch.from_numpy(to_six_band(arr, "LISS3")).float().unsqueeze(0)
    physics = torch.tensor([[0.12,15,3.0]]).float()
    cam = gradcam_seg(model, x, physics, device)
    rep = physics_consistency_report(model, x, physics, 0.12, device)
    np.save(PROJECT_ROOT/"figures"/"xai"/"gradcam.npy", cam)
    print(f"\n{B}XAI report:{X} {rep}")
    ok("saved gradcam.npy under figures/xai/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--xai", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    print(f"\n{B}BhuViveka-VLM — Meta-Learning + Physics-RL + XAI{X}")
    if args.smoke_test: run_smoke()
    elif args.train:    run_train(args)
    elif args.xai:      run_xai(args)
    else:
        warn("nothing to do. Use --smoke-test, --train, or --xai")


if __name__ == "__main__":
    main()