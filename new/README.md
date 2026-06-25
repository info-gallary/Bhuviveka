# Journal Evaluation Bundle

Self-contained evidence package for peer review of **BhuViveka**: a two-stage
Prithvi-EO-2.0-300M pipeline for Himalayan GLOF (glacial-lake outburst flood)
risk assessment: glacial-lake segmentation (Stage 1) plus a physics-grounded
GRPO reasoning head (Stage 2).

This folder contains the review-facing evidence bundle: logs, tables, figures,
split files, and cached evaluation outputs. Training curves can be checked
directly from the logs. Model-quality metrics such as segmentation IoU/F1 are
validated either from the stored checkpoint `metrics` field or by rerunning the
evaluation scripts against the corresponding checkpoint and split.

## Contents

| Path | What it is | Use in review |
|---|---|---|
| [`BhuViveka_Project_Report.md`](BhuViveka_Project_Report.md) | 50-page project documentation report with architecture, dataset, metrics, limitations, and artifact map | Project report / supplementary documentation |
| [`EVALUATION_SUMMARY.md`](EVALUATION_SUMMARY.md) | Headline results derived from the raw logs below | Start here |
| `logs/seg_300m.jsonl` | Seg v1 training trace (66 epochs, 28,472 steps) | Reproduce Fig. 4a / Table 3 |
| `logs/seg_v2_300m.jsonl` | Seg v2 (final) training trace (42 epochs, 9,156 steps) | Reproduce Fig. 4a / Table 3 |
| `logs/stage1_300m.jsonl` | Stage-1 backbone fine-tune trace (7 epochs) | Provenance |
| `logs/stage2_grpo.json` | GRPO per-epoch val reward + macro/per-class F1 (20 epochs) | Reproduce Table 4 / Fig. 5 |
| `tables/table1..6_*.{md,tex}` | The six paper result tables (Markdown + LaTeX) | Cross-check claims |
| `figures/fig_*.png` | Main paper figures (pipeline, seg, GRPO, XAI, spatial) | Visual evidence |
| `figures/plot_*.png` | Analytical plots regenerated from `logs/` | Training-curve audit |
| `results/enhanced_metrics.json` | Per-event temporal + spatial generalization metrics | **Source of Table 5** |
| `results/paper_ready/*.json` | E2E oracle, rule/sklearn baselines, ablations, calibration, multiseed | Supplementary claims |
| `results/e2e_eval/*.json` | Full end-to-end evaluation dumps (v1 + v3) | Deep audit |
| `results/table_*.txt` | Pre-rendered generalization + proximity tables | Cross-check |
| `results/*_inference_results.csv` | Small per-scene inference dumps (3-scene validation, visual demo) | Spot-check |
| `splits/{train,val,test_temporal,test_spatial}.csv` | Exact patch membership of each split (patch_id, event, label, lon/lat, npy_path) | **Reproduce splits / Table 1** |

## Checkpoint-To-Metric Mapping

Segmentation metrics are not stored in `logs/seg_*.jsonl`; those files contain
only per-batch `epoch`, `batch`, `global_step`, `loss`, and `lr`. The authoritative
IoU/F1 values must map to the matching checkpoint or evaluation output:

| Model | Checkpoint | Metric source | Reported result |
|---|---|---|---|
| Seg v1 / ConvTranspose | `inference_app/checkpoints/seg_300m_best.pt` or `checkpoints/seg_300m/best.pt` | `ckpt["metrics"]` | validation IoU `0.2933817771511773` |
| Seg v2 / FPN final | `checkpoints/seg_v2_300m/best.pt` or transferred `seg_v2_300m_best.pt` | checkpoint `metrics` plus test evaluation artifact | test mIoU `0.34`, test F1 `0.86` |
| Stage 2 GRPO | `inference_app/checkpoints/grpo_best.pt` or `checkpoints/stage2_grpo/best.pt` | `ckpt["metrics"]` and `logs/stage2_grpo.json` | macro F1 `0.8620613426213392` |

To inspect a checkpoint metric:

```bash
python -c "import torch; ck=torch.load('inference_app/checkpoints/seg_300m_best.pt', map_location='cpu', weights_only=False); print(ck['metrics'])"
```

For publication review, Seg v2 is the primary segmentation model. Seg v1 is kept
as an ablation/baseline and should not be used as evidence for the Seg v2 test
IoU/F1 numbers.

## Reproducing The Figures From These Logs

Every plot in `figures/plot_*.png` is generated deterministically from the files
in `logs/`. The generator lives at `paper/make_assets.py` in the repo root:

```bash
.venv/Scripts/python.exe paper/make_assets.py
```

## Honesty Note For Reviewers

Table 5 (`tables/table5_generalization.md`) reports the model's **out-of-region**
behaviour, where accuracy drops sharply while confidence stays high. This is an
intentionally surfaced limitation, not an omission; see `EVALUATION_SUMMARY.md`.
