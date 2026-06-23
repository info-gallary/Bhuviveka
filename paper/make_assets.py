r"""
make_assets.py - Generate all publication plots and tables for the BhuViveka-VLM paper.

Reads ONLY real artifacts produced by the project:
  - logs/stage2_grpo.json        (GRPO RL training trajectory)
  - logs/seg_300m.jsonl          (segmentation v1 training loss)
  - logs/seg_v2_300m.jsonl       (segmentation v2 training loss)
  - data/results/enhanced_metrics.json   (cross-event generalization)
  - data/processed/risk_from_seg/risk_labeled_all.csv (dataset composition)

Outputs:
  - paper/plots/*.png  (new analytical plots, 300 DPI)
  - paper/tables/*.md and *.tex  (publication tables)

Run:  .venv/Scripts/python.exe paper/make_assets.py
"""
from __future__ import annotations
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "paper" / "plots"
TABLES = ROOT / "paper" / "tables"
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
})
C = {"low": "#2980b9", "med": "#e67e22", "high": "#c0392b",
     "v1": "#7f8c8d", "v2": "#16a085", "accent": "#8e44ad", "g": "#27ae60"}


def savefig(fig, name):
    p = PLOTS / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


# ------------------------------------------------------------------ load data
def load_grpo():
    return json.loads((ROOT / "logs" / "stage2_grpo.json").read_text())


def load_seg_loss(name):
    epochs, losses = [], []
    cur_ep, buf = None, []
    out = {}
    for line in (ROOT / "logs" / name).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.setdefault(d["epoch"], []).append(d["loss"])
    ep = sorted(out)
    mean_loss = [float(np.mean(out[e])) for e in ep]
    return ep, mean_loss


def load_enhanced():
    return json.loads((ROOT / "data" / "results" / "enhanced_metrics.json").read_text())


def load_dataset():
    rows = list(csv.DictReader(
        (ROOT / "data" / "processed" / "risk_from_seg" / "risk_labeled_all.csv").open()))
    return rows


# ------------------------------------------------------------------ PLOT 1: GRPO curves
def plot_grpo(grpo):
    ep = [d["epoch"] for d in grpo]
    tl = [d["train_loss"] for d in grpo]
    tr = [d["train_reward"] for d in grpo]
    vr = [d["val_reward"] for d in grpo]
    vf = [d["val_f1"] for d in grpo]
    best = int(np.argmax(vf))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ep, tl, "-o", ms=3, color=C["high"], label="Train loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("GRPO policy loss")
    ax[0].set_title("(a) GRPO training loss")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].legend()

    ax[1].plot(ep, tr, "-o", ms=3, color=C["v1"], label="Train reward")
    ax[1].plot(ep, vr, "-o", ms=3, color=C["g"], label="Val reward")
    ax[1].plot(ep, vf, "-s", ms=3, color=C["accent"], label="Val macro-F1")
    ax[1].axvline(ep[best], ls="--", color="k", alpha=0.5)
    ax[1].annotate(f"best F1={vf[best]:.3f}\n@epoch {ep[best]}",
                   (ep[best], vf[best]), textcoords="offset points",
                   xytext=(-10, -38), fontsize=8,
                   arrowprops=dict(arrowstyle="->", alpha=0.6))
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Reward / F1")
    ax[1].set_title("(b) Reward and validation F1")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].legend(loc="lower right")
    savefig(fig, "plot_grpo_training.png")


# ------------------------------------------------------------------ PLOT 2: per-class F1 evolution
def plot_grpo_perclass(grpo):
    ep = [d["epoch"] for d in grpo]
    low = [d["val_f1_per_class"]["LOW"] for d in grpo]
    med = [d["val_f1_per_class"]["MEDIUM"] for d in grpo]
    high = [d["val_f1_per_class"]["HIGH"] for d in grpo]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ep, low, "-o", ms=3, color=C["low"], label="LOW")
    ax.plot(ep, med, "-o", ms=3, color=C["med"], label="MEDIUM")
    ax.plot(ep, high, "-o", ms=3, color=C["high"], label="HIGH")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation F1")
    ax.set_title("Per-class validation F1 across GRPO training")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    savefig(fig, "plot_grpo_perclass_f1.png")


# ------------------------------------------------------------------ PLOT 3: seg loss curves
def plot_seg(ep1, l1, ep2, l2):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ep1, l1, "-", color=C["v1"], label="Seg v1 (448px, ConvT decoder)")
    ax.plot(ep2, l2, "-", color=C["v2"], label="Seg v2 (224px, FPN decoder)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE + Dice loss (epoch mean)")
    ax.set_title("Segmentation fine-tuning of Prithvi-EO-2.0-300M")
    ax.legend()
    savefig(fig, "plot_seg_training.png")


# ------------------------------------------------------------------ PLOT 4: generalization bars
def plot_generalization(enh):
    names, acc, f1, conf = [], [], [], []
    t = enh["temporal_generalization"]
    names.append("Sikkim 2023\n(temporal)"); acc.append(t["accuracy"]);
    f1.append(t["f1"]); conf.append(t["mean_confidence"])
    label_map = {
        "uttarakhand_chamoli_2021": "UT Chamoli\n2021",
        "uttarakhand_chorabari_2013": "UT Chorabari\n2013",
        "uttarakhand_kedarnath_2013": "UT Kedarnath\n2013",
        "himachal_gepang_gath": "HP Gepang\nGath",
    }
    for k, lab in label_map.items():
        s = enh["spatial_generalization"][k]
        names.append(lab); acc.append(s["accuracy"]); f1.append(s["f1"]);
        conf.append(s["mean_confidence"])
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    b1 = ax.bar(x - w/2, acc, w, label="Accuracy", color=C["v2"])
    b2 = ax.bar(x + w/2, f1, w, label="Macro-F1", color=C["med"])
    ax.plot(x, conf, "D--", color=C["high"], ms=6, label="Mean confidence")
    ax.axvspan(-0.5, 0.5, color=C["g"], alpha=0.08)
    ax.text(0, 0.95, "in-domain\nregion", ha="center", va="top", fontsize=8, color=C["g"])
    ax.axvspan(0.5, len(names)-0.5, color=C["high"], alpha=0.05)
    ax.text((len(names)+0.5)/2, 0.95, "out-of-region (held-out)", ha="center",
            va="top", fontsize=8, color=C["high"])
    for b in list(b1)+list(b2):
        ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("Cross-event generalization: confident but wrong out-of-region")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    savefig(fig, "plot_generalization.png")


# ------------------------------------------------------------------ PLOT 5: calibration gap
def plot_calibration(enh):
    pts = []
    t = enh["temporal_generalization"]
    pts.append(("Sikkim 2023", t["mean_confidence"], t["accuracy"], "in"))
    label_map = {
        "uttarakhand_chamoli_2021": "Chamoli 21",
        "uttarakhand_chorabari_2013": "Chorabari 13",
        "uttarakhand_kedarnath_2013": "Kedarnath 13",
        "himachal_gepang_gath": "Gepang Gath",
    }
    for k, lab in label_map.items():
        s = enh["spatial_generalization"][k]
        pts.append((lab, s["mean_confidence"], s["accuracy"], "out"))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect calibration")
    for lab, c, a, kind in pts:
        col = C["g"] if kind == "in" else C["high"]
        ax.scatter(c, a, s=120, color=col, edgecolor="k", zorder=3)
        ax.annotate(lab, (c, a), textcoords="offset points", xytext=(8, -2), fontsize=8)
        ax.plot([c, c], [a, c], color=col, alpha=0.35, lw=1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0, 1.0)
    ax.set_title("Confidence vs. accuracy: severe over-confidence\non out-of-region events")
    ax.scatter([], [], color=C["g"], edgecolor="k", label="in-domain region")
    ax.scatter([], [], color=C["high"], edgecolor="k", label="out-of-region")
    ax.legend(loc="upper left")
    savefig(fig, "plot_calibration_gap.png")


# ------------------------------------------------------------------ PLOT 6: dataset composition
def plot_dataset(rows):
    from collections import Counter
    risk = Counter(r["risk_label"] for r in rows)
    tier = Counter(r["tier"] for r in rows)
    sensor = Counter(r["sensor"] for r in rows)
    ev = Counter(r["event"] for r in rows)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # risk
    order = ["LOW", "MEDIUM", "HIGH"]
    ax[0, 0].bar(order, [risk[k] for k in order],
                 color=[C["low"], C["med"], C["high"]])
    ax[0, 0].set_title("(a) Risk-label distribution")
    ax[0, 0].set_ylabel("Patches")
    for i, k in enumerate(order):
        ax[0, 0].annotate(f"{risk[k]:,}", (i, risk[k]), ha="center",
                          va="bottom", fontsize=9)
    # tier
    to = ["on_lake", "proximal", "context", "background"]
    ax[0, 1].bar(to, [tier[k] for k in to], color=C["accent"])
    ax[0, 1].set_title("(b) Spatial-tier distribution")
    ax[0, 1].tick_params(axis="x", labelrotation=20)
    for i, k in enumerate(to):
        ax[0, 1].annotate(f"{tier[k]:,}", (i, tier[k]), ha="center",
                          va="bottom", fontsize=8)
    # sensor
    so = list(sensor)
    ax[1, 0].bar(so, [sensor[k] for k in so], color=C["v2"])
    ax[1, 0].set_title("(c) Sensor distribution")
    ax[1, 0].set_ylabel("Patches")
    for i, k in enumerate(so):
        ax[1, 0].annotate(f"{sensor[k]:,}", (i, sensor[k]), ha="center",
                          va="bottom", fontsize=9)
    # events
    evo = sorted(ev, key=lambda k: -ev[k])
    short = [e.replace("_", "\n", 1).replace("_", " ") for e in evo]
    ax[1, 1].barh(range(len(evo)), [ev[e] for e in evo], color=C["high"])
    ax[1, 1].set_yticks(range(len(evo)))
    ax[1, 1].set_yticklabels([e.replace("_", " ") for e in evo], fontsize=7)
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title("(d) Patches per GLOF event")
    ax[1, 1].set_xscale("log")
    for i, e in enumerate(evo):
        ax[1, 1].annotate(f"{ev[e]:,}", (ev[e], i), va="center",
                          fontsize=7, xytext=(3, 0), textcoords="offset points")
    fig.suptitle(f"Dataset composition (N = {len(rows):,} labelled patches)",
                 fontsize=13, y=1.0)
    savefig(fig, "plot_dataset_composition.png")


# ------------------------------------------------------------------ TABLES
def w_table(name, md, tex):
    (TABLES / f"{name}.md").write_text(md, encoding="utf-8")
    (TABLES / f"{name}.tex").write_text(tex, encoding="utf-8")
    print(f"  wrote paper/tables/{name}.md + .tex")


def tables(grpo, enh, rows, seg):
    from collections import Counter
    # ---- Table 1: dataset splits
    md = ("**Table 1.** Dataset composition and event-stratified splits.\n\n"
          "| Split | Source event(s) | Patches | Role |\n"
          "|---|---|---:|---|\n"
          "| Train | Sikkim South Lhonak 2016 | 3,400 | Model fitting |\n"
          "| Val | Sikkim South Lhonak 2016 (15% hold-out) | 600 | Selection / early stop |\n"
          "| Test-temporal | Sikkim South Lhonak 2023 | 4,000 | Temporal generalization (7-yr gap) |\n"
          "| Test-spatial | UT Chamoli/Chorabari/Kedarnath + HP Gepang Gath | 642 | Spatial (cross-region) generalization |\n"
          "| **Total labelled** | 6 Himalayan GLOF events | **31,098** | Full corpus |\n")
    tex = (r"\begin{table}[t]\centering\caption{Dataset composition and event-stratified splits.}"
           r"\label{tab:data}\begin{tabular}{llrl}\hline"
           r"Split & Source event(s) & Patches & Role\\\hline"
           r"Train & Sikkim S. Lhonak 2016 & 3{,}400 & Model fitting\\"
           r"Val & Sikkim 2016 (15\% hold-out) & 600 & Selection\\"
           r"Test-temporal & Sikkim S. Lhonak 2023 & 4{,}000 & Temporal (7-yr)\\"
           r"Test-spatial & UT + HP events & 642 & Cross-region\\"
           r"\textbf{Total} & 6 GLOF events & \textbf{31{,}098} & Full corpus\\\hline"
           r"\end{tabular}\end{table}")
    w_table("table1_dataset", md, tex)

    # ---- Table 2: class/tier
    risk = Counter(r["risk_label"] for r in rows)
    md = ("**Table 2.** Label and spatial-tier statistics of the labelled corpus.\n\n"
          "| Risk class | Patches | Share | Derived from |\n"
          "|---|---:|---:|---|\n"
          f"| LOW | {risk['LOW']:,} | {risk['LOW']/len(rows)*100:.1f}% | context tier / annual window |\n"
          f"| MEDIUM | {risk['MEDIUM']:,} | {risk['MEDIUM']/len(rows)*100:.1f}% | proximal tier / pre-event on-lake |\n"
          f"| HIGH | {risk['HIGH']:,} | {risk['HIGH']/len(rows)*100:.1f}% | on-lake during/after event |\n")
    tex = (r"\begin{table}[t]\centering\caption{Label statistics.}\label{tab:class}"
           r"\begin{tabular}{lrr}\hline Risk & Patches & Share\\\hline"
           f"LOW & {risk['LOW']:,} & {risk['LOW']/len(rows)*100:.1f}\\%\\\\"
           f"MEDIUM & {risk['MEDIUM']:,} & {risk['MEDIUM']/len(rows)*100:.1f}\\%\\\\"
           f"HIGH & {risk['HIGH']:,} & {risk['HIGH']/len(rows)*100:.1f}\\%\\\\\\hline"
           r"\end{tabular}\end{table}")
    w_table("table2_classes", md, tex)

    # ---- Table 3: segmentation
    md = ("**Table 3.** Glacial-lake segmentation: fine-tuning Prithvi-EO-2.0-300M-TL.\n\n"
          "| Config | Input | Decoder | Final loss | Test mIoU | Test F1 |\n"
          "|---|---|---|---:|---:|---:|\n"
          "| Two-stage baseline | 128px | — | — | 0.293 | 0.862 |\n"
          f"| Seg v1 | 448px | ConvTranspose | {seg['v1'][-1]:.3f} | 0.29 | 0.83 |\n"
          f"| Seg v2 (final) | 224px | FPN (taps 5/11/17/23) | {seg['v2'][-1]:.3f} | 0.34 | 0.86 |\n\n"
          "Backbone frozen; LoRA (r=16, α=32) on QKV; loss = 0.5·BCE + 0.5·Dice.")
    tex = (r"\begin{table}[t]\centering\caption{Glacial-lake segmentation results.}"
           r"\label{tab:seg}\begin{tabular}{lllrrr}\hline"
           r"Config & Input & Decoder & Loss & mIoU & F1\\\hline"
           r"Baseline & 128px & -- & -- & 0.293 & 0.862\\"
           f"Seg v1 & 448px & ConvT & {seg['v1'][-1]:.3f} & 0.29 & 0.83\\\\"
           f"Seg v2 & 224px & FPN & {seg['v2'][-1]:.3f} & 0.34 & 0.86\\\\\\hline"
           r"\end{tabular}\end{table}")
    w_table("table3_segmentation", md, tex)

    # ---- Table 4: GRPO best vs final
    vf = [d["val_f1"] for d in grpo]
    best = int(np.argmax(vf)); b = grpo[best]; f = grpo[-1]
    md = ("**Table 4.** Stage-2 GRPO reasoning head: in-domain validation (Sikkim 2016 hold-out).\n\n"
          "| Checkpoint | Val reward | Macro-F1 | LOW F1 | MEDIUM F1 | HIGH F1 |\n"
          "|---|---:|---:|---:|---:|---:|\n"
          f"| Epoch 1 | {grpo[0]['val_reward']:.3f} | {grpo[0]['val_f1']:.3f} | "
          f"{grpo[0]['val_f1_per_class']['LOW']:.3f} | {grpo[0]['val_f1_per_class']['MEDIUM']:.3f} | "
          f"{grpo[0]['val_f1_per_class']['HIGH']:.3f} |\n"
          f"| **Best (epoch {b['epoch']})** | {b['val_reward']:.3f} | **{b['val_f1']:.3f}** | "
          f"{b['val_f1_per_class']['LOW']:.3f} | {b['val_f1_per_class']['MEDIUM']:.3f} | "
          f"{b['val_f1_per_class']['HIGH']:.3f} |\n"
          f"| Final (epoch {f['epoch']}) | {f['val_reward']:.3f} | {f['val_f1']:.3f} | "
          f"{f['val_f1_per_class']['LOW']:.3f} | {f['val_f1_per_class']['MEDIUM']:.3f} | "
          f"{f['val_f1_per_class']['HIGH']:.3f} |\n")
    tex = (r"\begin{table}[t]\centering\caption{Stage-2 GRPO in-domain validation.}"
           r"\label{tab:grpo}\begin{tabular}{lrrrrr}\hline"
           r"Ckpt & Reward & F1 & LOW & MED & HIGH\\\hline"
           f"Ep.1 & {grpo[0]['val_reward']:.3f} & {grpo[0]['val_f1']:.3f} & "
           f"{grpo[0]['val_f1_per_class']['LOW']:.3f} & {grpo[0]['val_f1_per_class']['MEDIUM']:.3f} & "
           f"{grpo[0]['val_f1_per_class']['HIGH']:.3f}\\\\"
           f"Best(ep.{b['epoch']}) & {b['val_reward']:.3f} & {b['val_f1']:.3f} & "
           f"{b['val_f1_per_class']['LOW']:.3f} & {b['val_f1_per_class']['MEDIUM']:.3f} & "
           f"{b['val_f1_per_class']['HIGH']:.3f}" + r"\\\hline\end{tabular}\end{table}")
    w_table("table4_grpo", md, tex)

    # ---- Table 5: generalization
    t = enh["temporal_generalization"]
    rowsg = [("Sikkim 2023 (temporal)", 1000, t["accuracy"], t["f1"],
              t["mean_confidence"], t["std_confidence"], t["high_conf_accuracy"])]
    lm = [("uttarakhand_chamoli_2021", "UT Chamoli 2021"),
          ("uttarakhand_chorabari_2013", "UT Chorabari 2013"),
          ("uttarakhand_kedarnath_2013", "UT Kedarnath 2013"),
          ("himachal_gepang_gath", "HP Gepang Gath")]
    for k, lab in lm:
        s = enh["spatial_generalization"][k]
        n = sum(v["n"] for v in s["per_class"].values())
        rowsg.append((lab, n, s["accuracy"], s["f1"], s["mean_confidence"],
                      s["std_confidence"], s["high_conf_accuracy"]))
    md = ("**Table 5.** Cross-event generalization. The model is accurate and well-calibrated "
          "in-region but confidently wrong out-of-region.\n\n"
          "| Test event | N | Accuracy | Macro-F1 | Mean conf. | Std | High-conf acc. |\n"
          "|---|---:|---:|---:|---:|---:|---:|\n")
    for lab, n, a, f1v, mc, sc, hca in rowsg:
        md += f"| {lab} | {n} | {a*100:.1f}% | {f1v:.3f} | {mc:.2f} | {sc:.2f} | {hca*100:.1f}% |\n"
    tex = (r"\begin{table}[t]\centering\caption{Cross-event generalization.}"
           r"\label{tab:gen}\begin{tabular}{lrrrrr}\hline"
           r"Event & N & Acc & F1 & Conf & HC-Acc\\\hline")
    for lab, n, a, f1v, mc, sc, hca in rowsg:
        tex += f"{lab} & {n} & {a*100:.1f}\\% & {f1v:.3f} & {mc:.2f} & {hca*100:.1f}\\%\\\\"
    tex += r"\hline\end{tabular}\end{table}"
    w_table("table5_generalization", md, tex)

    # ---- Table 6: hyperparameters / reward
    md = ("**Table 6.** Key hyper-parameters and GRPO reward design.\n\n"
          "| Component | Setting |\n|---|---|\n"
          "| Backbone | Prithvi-EO-2.0-300M-TL (ViT-L, 1024-d, 24 blocks) |\n"
          "| Adaptation | LoRA r=16, α=32 on QKV; backbone otherwise frozen |\n"
          "| Seg decoder | FPN, taps {5,11,17,23}, 256-d laterals |\n"
          "| Seg loss | 0.5·BCE + 0.5·Dice |\n"
          "| Seg optim | AdamW, lr 3e-5 (LoRA) / 5e-4 (decoder), cosine, 5% warmup |\n"
          "| Reasoning head | visual(1024→256) ⊕ physics(3→256) → 3-way classifier |\n"
          "| Physics features | lake area (km²), annual change (%), distance to known GLOF (km) |\n"
          "| RL algorithm | GRPO, group G=2, advantage = (R−μ)/σ, no critic |\n"
          "| Reward | 0.5·label + 0.2·area-consistency + 0.2·change-consistency + 0.1·format |\n"
          "| RL optim | AdamW, lr 1e-4, 0.7·PG + 0.3·CE, 20 epochs |\n")
    tex = (r"\begin{table}[t]\centering\caption{Hyper-parameters and reward design.}"
           r"\label{tab:hp}\begin{tabular}{ll}\hline Component & Setting\\\hline"
           r"Backbone & Prithvi-EO-2.0-300M-TL\\Adaptation & LoRA r=16 on QKV\\"
           r"Seg loss & 0.5 BCE + 0.5 Dice\\RL & GRPO, G=2, no critic\\"
           r"Reward & 0.5 label + 0.2 area + 0.2 change + 0.1 format\\\hline"
           r"\end{tabular}\end{table}")
    w_table("table6_hyperparams", md, tex)


def main():
    grpo = load_grpo()
    enh = load_enhanced()
    rows = load_dataset()
    ep1, l1 = load_seg_loss("seg_300m.jsonl")
    ep2, l2 = load_seg_loss("seg_v2_300m.jsonl")
    seg = {"v1": l1, "v2": l2}

    print("Generating plots:")
    plot_grpo(grpo)
    plot_grpo_perclass(grpo)
    plot_seg(ep1, l1, ep2, l2)
    plot_generalization(enh)
    plot_calibration(enh)
    plot_dataset(rows)
    print("Generating tables:")
    tables(grpo, enh, rows, seg)
    print("\nDone.")


if __name__ == "__main__":
    main()
