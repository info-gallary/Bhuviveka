r"""
BhuViveka-VLM — Evaluation vs Two-Stage Baseline
==================================================
Runs the trained unified VLM on the test set and reports IoU + F1, so you
can tell immediately whether meta-learning helped or hurt vs your existing
pipeline (Seg IoU 0.293 / RL F1 0.862).

Corruption-safe loading. Reports per-event breakdown + class distribution
(to catch the "policy collapsed to one class" failure mode).

USAGE
  uv run eval_vlm.py
  uv run eval_vlm.py --max 600
"""
from __future__ import annotations
import argparse, csv, sys, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings("ignore")

G,Y,R,Cc,B,D,X="\033[92m","\033[93m","\033[91m","\033[96m","\033[1m","\033[2m","\033[0m"
def ok(m="OK"): print(f"  {G}\u2713 {m}{X}")
def info(m): print(f"  {D}  {m}{X}")
def warn(m): print(f"  {Y}\u26a0 {m}{X}")
def fatal(m): print(f"\n{R}{B}\u2717 {m}{X}"); sys.exit(1)

ROOT = Path.cwd()
RISK_CSV = ROOT/"data"/"processed"/"risk_from_seg"/"risk_labeled_all.csv"
CKPT = ROOT/"checkpoints"/"vlm_meta"/"best.pt"
CLASS = ["LOW","MEDIUM","HIGH"]
TEST_EVENTS = ["sikkim_south_lhonak_2023","uttarakhand_chamoli_2021",
               "uttarakhand_chorabari_2013","uttarakhand_kedarnath_2013",
               "himachal_gepang_gath"]
BASELINE_IOU, BASELINE_F1 = 0.293, 0.862

def six(arr, s):
    H,W=arr.shape[1],arr.shape[2]; o=np.zeros((6,H,W),np.float32)
    if s=="LISS3" and arr.shape[0]>=4: o[1:5]=arr[:4]
    elif s=="LISS4" and arr.shape[0]>=3: o[1:4]=arr[:3]
    else: n=min(6,arr.shape[0]); o[:n]=arr[:n]
    return o

def load(path):
    p=ROOT/path
    if not p.exists(): return None
    try:
        a=np.load(p,allow_pickle=False)
        if a.ndim!=3 or a.size==0: return None
        return np.nan_to_num(a,0,0,0).astype(np.float32)
    except Exception: return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--max",type=int,default=600); a=ap.parse_args()
    print(f"\n{B}BhuViveka-VLM — Eval vs Two-Stage Baseline{X}\n")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type=="cpu": fatal("CUDA required")

    from train1 import BhuVivekaVLM, load_real_backbone
    backbone,dim=load_real_backbone()
    model=BhuVivekaVLM(backbone,dim).to(dev)
    if not CKPT.exists(): fatal(f"no checkpoint at {CKPT} — train first")
    model.load_state_dict(torch.load(CKPT,map_location="cpu",weights_only=False)["model_state_dict"])
    model.eval(); ok("VLM checkpoint loaded")

    with open(RISK_CSV,encoding="utf-8") as f: rows=list(csv.DictReader(f))
    test=[r for r in rows if r.get("event") in TEST_EVENTS]
    info(f"{len(test)} test patches")

    per_event=defaultdict(int)
    ious=[]; preds=[]; labels=[]; skipped=0
    for r in test:
        ev=r.get("event","?")
        if per_event[ev]>=a.max: continue
        arr=load(r.get("npy_path",""))
        if arr is None: skipped+=1; continue
        try:
            x=torch.from_numpy(six(arr,r["sensor"])).float().unsqueeze(0).to(dev)
            phy=torch.tensor([[float(r.get("seg_area_km2",0)),
                               float(r.get("change_pct",0)),
                               float(r.get("dist_km",0))]]).float().to(dev)
            with torch.no_grad():
                out=model(x,phy)
                pred_mask=(torch.sigmoid(out["seg"])>0.5).float()
                risk=CLASS[int(out["risk"].argmax(-1))]
            # IoU vs stored mask if available
            m=load(r.get("mask_path",""))
            if m is not None:
                m=m[:1] if m.ndim==3 else m[None]
                mt=torch.from_numpy(m).float().to(dev)
                if mt.shape[-1]!=128:
                    mt=F.interpolate(mt.unsqueeze(0),size=(128,128)).squeeze(0)
                inter=(pred_mask[0]*mt).sum(); union=((pred_mask[0]+mt)>0).float().sum()
                if union>0: ious.append(float(inter/union))
            preds.append(risk); labels.append(r.get("risk_label","LOW"))
            per_event[ev]+=1
        except Exception: skipped+=1; continue

    from sklearn.metrics import f1_score, accuracy_score
    miou=float(np.mean(ious)) if ious else 0.0
    f1=f1_score(labels,preds,average="macro",zero_division=0) if preds else 0.0
    acc=accuracy_score(labels,preds) if preds else 0.0

    print(f"\n{B}Results (n={len(preds)}, skipped {skipped}){X}")
    print(f"  Segmentation mIoU:  {miou:.3f}   baseline {BASELINE_IOU:.3f}  "
          f"{'BETTER' if miou>BASELINE_IOU else 'worse'}")
    print(f"  Risk macro-F1:      {f1:.3f}   baseline {BASELINE_F1:.3f}  "
          f"{'BETTER' if f1>BASELINE_F1 else 'worse'}")
    print(f"  Risk accuracy:      {acc:.3f}")

    print(f"\n{B}Prediction distribution{X} (collapse check)")
    dist=Counter(preds)
    for c in CLASS:
        n=dist.get(c,0); bar="#"*int(30*n/max(len(preds),1))
        print(f"  {c:6s} {n:5d}  {bar}")
    if max(dist.values())/max(len(preds),1) > 0.85:
        warn("policy is predicting one class >85% of the time — likely collapse")

    print(f"\n{B}Verdict:{X} ",end="")
    if f1>BASELINE_F1 and miou>BASELINE_IOU:
        print(f"{G}Unified VLM BEATS the two-stage baseline. Ship it.{X}")
    elif f1>=BASELINE_F1*0.95 and miou>=BASELINE_IOU*0.95:
        print(f"{Y}Comparable to baseline. Meta-learning didn't clearly help — report as ablation.{X}")
    else:
        print(f"{R}Below baseline. Keep two-stage as main result; this is a negative-result ablation.{X}")
    print()

if __name__=="__main__": main()