r"""
BhuViveka — Baselines Comparison (Point 4)
==========================================
Runs all required baselines on the same test set as Stage 2.

Baselines:
  B0  Rule matrix         — deterministic threshold on GT descriptors
  B1  Logistic Regression — on GT descriptors (sklearn)
  B2  XGBoost             — on GT descriptors
  B3  MLP                 — on GT descriptors (2-layer, 64-d)
  B4  CNN classifier      — ResNet-18 fine-tuned on LISS patches
  B5  ViT classifier      — Prithvi linear probe (frozen backbone)
  B6  Supervised Prithvi  — Prithvi + cross-entropy head (no GRPO)
  B7  GRPO (our method)   — Stage 2 with GT descriptors (oracle)

Outputs a comparison table: method | macro-F1 | bal-acc | per-class F1

Usage:
    uv run experiment_baselines.py
    uv run experiment_baselines.py --baselines B0 B1 B2   # subset
"""
from __future__ import annotations
import argparse, csv, json, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT     = Path.cwd()
RISK_CSV = ROOT / "data" / "processed" / "risk_from_seg" / "risk_labeled_all.csv"
OUT_DIR  = ROOT / "data" / "results" / "baselines"
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
    else:
        n = min(6, arr.shape[0]); out[:n] = arr[:n]
    return out


def label_to_id(label): return CLASSES.index(label) if label in CLASSES else 0
def id_to_label(i): return CLASSES[i]


def get_test_data(max_per_event=400):
    rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    from collections import defaultdict
    count = defaultdict(int)
    test_rows = []
    for r in rows:
        ev = r.get("event","")
        if ev in TEST_EVENTS and count[ev] < max_per_event:
            test_rows.append(r); count[ev] += 1
    return test_rows


def metrics(preds, labels):
    f1 = f1_score(labels, preds, average="macro", labels=CLASSES, zero_division=0)
    ba = balanced_accuracy_score(labels, preds)
    per = {}
    for i, c in enumerate(CLASSES):
        tp = sum(1 for p,l in zip(preds,labels) if p==c and l==c)
        fp = sum(1 for p,l in zip(preds,labels) if p==c and l!=c)
        fn = sum(1 for p,l in zip(preds,labels) if p!=c and l==c)
        pr = tp/(tp+fp) if (tp+fp)>0 else 0
        rc = tp/(tp+fn) if (tp+fn)>0 else 0
        f1c = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
        per[c] = {"p": round(pr,3), "r": round(rc,3), "f1": round(f1c,3)}
    return {"macro_f1": round(f1,4), "bal_acc": round(ba,4), "per_class": per}


# ─────────────────────────────────────────────────────────────────────────────
# B0: RULE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def baseline_rule(rows):
    """The deterministic threshold rule used to GENERATE the labels.
    Expected to match perfectly on patches where labels were derived this way.
    This quantifies how much of Stage-2 performance comes from re-applying the rule."""
    preds, labels = [], []
    for r in rows:
        area   = float(r.get("seg_area_km2", 0))
        change = float(r.get("change_pct", 0))
        dist   = float(r.get("dist_km", 0))
        if area > 0.15 and change > 15:
            pred = "HIGH"
        elif (0.05 <= area <= 0.15) or (5 <= change <= 15):
            pred = "MEDIUM"
        else:
            pred = "LOW"
        preds.append(pred)
        labels.append(r.get("risk_label", "LOW"))
    return preds, labels


# ─────────────────────────────────────────────────────────────────────────────
# B1/B2/B3: SKLEARN BASELINES ON PHYSICS DESCRIPTORS
# ─────────────────────────────────────────────────────────────────────────────
def baseline_sklearn(rows, method="logreg"):
    X = np.array([[float(r.get("seg_area_km2",0)),
                   float(r.get("change_pct",0)),
                   float(r.get("dist_km",0))] for r in rows])
    y = np.array([label_to_id(r.get("risk_label","LOW")) for r in rows])
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Use all non-test rows as "train" for these descriptors-only baselines
    all_rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    train_rows = [r for r in all_rows if r.get("event") not in TEST_EVENTS]
    if not train_rows:
        print(f"  WARNING: no train rows for sklearn; using test rows for fit (leaky demo only)")
        train_rows = rows[:len(rows)//2]

    X_tr = np.array([[float(r.get("seg_area_km2",0)),
                       float(r.get("change_pct",0)),
                       float(r.get("dist_km",0))] for r in train_rows])
    y_tr = np.array([label_to_id(r.get("risk_label","LOW")) for r in train_rows])

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_s = scaler.transform(X_tr), scaler.transform(X)

    if method == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif method == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=200, max_depth=4,
                                eval_metric="mlogloss", verbosity=0)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=100)
            print(f"  XGBoost not installed, using sklearn GBT")
    elif method == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64,64), max_iter=500)

    clf.fit(X_tr_s, y_tr)
    preds_id = clf.predict(X_s)
    preds = [id_to_label(i) for i in preds_id]
    labels = [r.get("risk_label","LOW") for r in rows]
    return preds, labels


# ─────────────────────────────────────────────────────────────────────────────
# B4: CNN CLASSIFIER (ResNet-18)
# ─────────────────────────────────────────────────────────────────────────────
def baseline_cnn(rows, device):
    """ResNet-18 fine-tuned on LISS imagery (first 3 channels as RGB proxy)."""
    try:
        import torchvision.models as tvm
    except ImportError:
        print("  torchvision not installed; skipping CNN baseline")
        return None, None

    model = tvm.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)  # 3-channel
    model.fc    = nn.Linear(512, 3)
    model = model.to(device)

    # Build train set from non-test events
    all_rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    tr = [r for r in all_rows if r.get("event") not in TEST_EVENTS]
    if not tr:
        print("  CNN: no train rows available; returning random predictions")
        labels = [r.get("risk_label","LOW") for r in rows]
        return [np.random.choice(CLASSES) for _ in rows], labels

    print(f"  CNN: training on {len(tr)} patches...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(5):
        np.random.shuffle(tr)
        for i in range(0, min(len(tr), 1000), 8):
            batch = tr[i:i+8]
            imgs, lbls = [], []
            for r in batch:
                arr = safe_load(r.get("npy_path",""))
                if arr is None: continue
                # Use first 3 channels as RGB proxy
                img = torch.from_numpy(arr[:3] if arr.shape[0]>=3 else
                      np.stack([arr[0]]*3)).float()
                img = F.interpolate(img.unsqueeze(0), size=(128,128)).squeeze(0)
                imgs.append(img)
                lbls.append(label_to_id(r.get("risk_label","LOW")))
            if not imgs: continue
            x = torch.stack(imgs).to(device)
            y = torch.tensor(lbls).long().to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    preds, labels = [], []
    for r in rows:
        arr = safe_load(r.get("npy_path",""))
        if arr is None: continue
        img = torch.from_numpy(arr[:3] if arr.shape[0]>=3 else
              np.stack([arr[0]]*3)).float().unsqueeze(0).to(device)
        img = F.interpolate(img, size=(128,128))
        with torch.no_grad():
            pred = id_to_label(int(model(img).argmax(-1).item()))
        preds.append(pred)
        labels.append(r.get("risk_label","LOW"))
    return preds, labels


# ─────────────────────────────────────────────────────────────────────────────
# B5: PRITHVI LINEAR PROBE
# ─────────────────────────────────────────────────────────────────────────────
def baseline_prithvi_probe(rows, device):
    """Frozen Prithvi CLS token → linear classifier (no GRPO, no physics)."""
    from train_seg import BhuVivekaSeg
    from prithvi_model import load_prithvi_backbone as _load
    from peft import get_peft_model, LoraConfig

    base, dim = _load("300M", ROOT/"models"/"prithvi-300m-tl")
    for p in base.parameters(): p.requires_grad = False
    tgt = [n for n,m in base.named_modules()
           if isinstance(m, nn.Linear) and n.endswith(".qkv")]
    seg = BhuVivekaSeg(get_peft_model(base,
          LoraConfig(r=16,lora_alpha=32,target_modules=tgt or ["qkv"],
                     lora_dropout=0.05,bias="none",modules_to_save=["patch_embed"])),
          dim)
    seg.load_state_dict(torch.load(ROOT/"checkpoints"/"seg_300m"/"best.pt",
                        map_location="cpu",weights_only=False)["model_state_dict"])
    seg = seg.to(device).eval()
    head = nn.Linear(dim, 3).to(device)

    # Quick train of head on non-test events
    all_rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    tr = [r for r in all_rows if r.get("event") not in TEST_EVENTS][:1000]
    if not tr:
        print("  Probe: no train rows; returning random")
        return [np.random.choice(CLASSES) for _ in rows], [r.get("risk_label","LOW") for r in rows]

    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    print(f"  Probe: training head on {len(tr)} patches...")
    for _ in range(3):
        for i in range(0,len(tr),8):
            batch = tr[i:i+8]
            feats, lbls = [], []
            for r in batch:
                arr = safe_load(r.get("npy_path",""))
                if arr is None: continue
                x = torch.from_numpy(to6(arr,r["sensor"])).float().unsqueeze(0).to(device)
                x_up = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False).unsqueeze(2)
                with torch.no_grad(): cls = seg.backbone(x_up)[:,0,:]
                feats.append(cls.squeeze(0)); lbls.append(label_to_id(r.get("risk_label","LOW")))
            if not feats: continue
            feat_t = torch.stack(feats).to(device)
            lbl_t  = torch.tensor(lbls).long().to(device)
            loss = F.cross_entropy(head(feat_t), lbl_t)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    preds, labels = [], []
    for r in rows:
        arr = safe_load(r.get("npy_path",""))
        if arr is None: continue
        x = torch.from_numpy(to6(arr,r["sensor"])).float().unsqueeze(0).to(device)
        x_up = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False).unsqueeze(2)
        with torch.no_grad():
            cls = seg.backbone(x_up)[:,0,:]
            pred = id_to_label(int(head(cls).argmax(-1).item()))
        preds.append(pred); labels.append(r.get("risk_label","LOW"))
    return preds, labels


# ─────────────────────────────────────────────────────────────────────────────
# B6: SUPERVISED PRITHVI HEAD (cross-entropy, no GRPO)
# ─────────────────────────────────────────────────────────────────────────────
def baseline_supervised_head(rows, device):
    """Prithvi CLS + physics → supervised cross-entropy head (no RL).
    Shows whether GRPO adds anything over standard supervised training."""
    from train_seg import BhuVivekaSeg
    from train_stage2_grpo import ReasoningHead
    from prithvi_model import load_prithvi_backbone as _load
    from peft import get_peft_model, LoraConfig

    base, dim = _load("300M", ROOT/"models"/"prithvi-300m-tl")
    for p in base.parameters(): p.requires_grad = False
    tgt = [n for n,m in base.named_modules() if isinstance(m,nn.Linear) and n.endswith(".qkv")]
    seg = BhuVivekaSeg(get_peft_model(base,
          LoraConfig(r=16,lora_alpha=32,target_modules=tgt or ["qkv"],
                     lora_dropout=0.05,bias="none",modules_to_save=["patch_embed"])),dim)
    seg.load_state_dict(torch.load(ROOT/"checkpoints"/"seg_300m"/"best.pt",
                        map_location="cpu",weights_only=False)["model_state_dict"])
    seg = seg.to(device).eval()
    head = ReasoningHead(visual_dim=dim, physics_dim=3).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-4)

    all_rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    tr = [r for r in all_rows if r.get("event") not in TEST_EVENTS][:2000]
    if not tr:
        print("  Supervised head: no train rows"); return None, None

    print(f"  Supervised head: training on {len(tr)} patches (CE only, no GRPO)...")
    for _ in range(5):
        for i in range(0,len(tr),8):
            batch = tr[i:i+8]
            feats, phys, lbls = [], [], []
            for r in batch:
                arr = safe_load(r.get("npy_path",""))
                if arr is None: continue
                x = torch.from_numpy(to6(arr,r["sensor"])).float().unsqueeze(0).to(device)
                x_up = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False).unsqueeze(2)
                with torch.no_grad(): cls = seg.backbone(x_up)[:,0,:]
                feats.append(cls.squeeze(0))
                phys.append([float(r.get("seg_area_km2",0)),float(r.get("change_pct",0)),float(r.get("dist_km",0))])
                lbls.append(label_to_id(r.get("risk_label","LOW")))
            if not feats: continue
            f_t = torch.stack(feats).to(device)
            p_t = torch.tensor(phys).float().to(device)
            l_t = torch.tensor(lbls).long().to(device)
            logits, _ = head(f_t, p_t)
            loss = F.cross_entropy(logits, l_t)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    preds, labels = [], []
    for r in rows:
        arr = safe_load(r.get("npy_path",""))
        if arr is None: continue
        x = torch.from_numpy(to6(arr,r["sensor"])).float().unsqueeze(0).to(device)
        x_up = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False).unsqueeze(2)
        phy = torch.tensor([[float(r.get("seg_area_km2",0)),float(r.get("change_pct",0)),float(r.get("dist_km",0))]]).float().to(device)
        with torch.no_grad():
            cls = seg.backbone(x_up)[:,0,:]
            logits, _ = head(cls, phy)
            pred = id_to_label(int(logits.argmax(-1).item()))
        preds.append(pred); labels.append(r.get("risk_label","LOW"))
    return preds, labels


def print_table(results):
    print(f"\n{'Method':<30} {'Macro-F1':>10} {'Bal-Acc':>10} {'F1-L':>8} {'F1-M':>8} {'F1-H':>8}")
    print("-"*76)
    for method, m in results.items():
        if m is None: continue
        pc = m.get("per_class", {})
        print(f"  {method:<28} {m['macro_f1']:>10.4f} {m['bal_acc']:>10.4f} "
              f"{pc.get('LOW',{}).get('f1',0):>8.3f} "
              f"{pc.get('MEDIUM',{}).get('f1',0):>8.3f} "
              f"{pc.get('HIGH',{}).get('f1',0):>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    default=["B0","B1","B2","B3","B4","B5","B6"])
    ap.add_argument("--max", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = get_test_data(args.max)
    print(f"Test set: {len(rows)} patches")

    results = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "B0" in args.baselines:
        print("\nB0: Rule matrix...")
        p,l = baseline_rule(rows); results["B0_rule"] = metrics(p,l)
        print(f"    macro-F1: {results['B0_rule']['macro_f1']}")

    if "B1" in args.baselines:
        print("\nB1: Logistic Regression...")
        p,l = baseline_sklearn(rows,"logreg"); results["B1_logreg"] = metrics(p,l)
        print(f"    macro-F1: {results['B1_logreg']['macro_f1']}")

    if "B2" in args.baselines:
        print("\nB2: XGBoost/GBT...")
        p,l = baseline_sklearn(rows,"xgb"); results["B2_xgb"] = metrics(p,l)
        print(f"    macro-F1: {results['B2_xgb']['macro_f1']}")

    if "B3" in args.baselines:
        print("\nB3: MLP...")
        p,l = baseline_sklearn(rows,"mlp"); results["B3_mlp"] = metrics(p,l)
        print(f"    macro-F1: {results['B3_mlp']['macro_f1']}")

    if "B4" in args.baselines:
        print("\nB4: CNN (ResNet-18)...")
        p,l = baseline_cnn(rows, device)
        if p: results["B4_cnn"] = metrics(p,l); print(f"    macro-F1: {results['B4_cnn']['macro_f1']}")

    if "B5" in args.baselines:
        print("\nB5: Prithvi linear probe...")
        p,l = baseline_prithvi_probe(rows, device)
        if p: results["B5_prithvi_probe"] = metrics(p,l); print(f"    macro-F1: {results['B5_prithvi_probe']['macro_f1']}")

    if "B6" in args.baselines:
        print("\nB6: Supervised Prithvi head (no GRPO)...")
        p,l = baseline_supervised_head(rows, device)
        if p: results["B6_supervised"] = metrics(p,l); print(f"    macro-F1: {results['B6_supervised']['macro_f1']}")

    print_table(results)
    out = OUT_DIR / "baselines_comparison.json"
    json.dump(results, open(out,"w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__": main()
