# BhuViveka-VLM — Paper Publication Folder

Everything needed to write and submit the research paper lives here.

## Quick Start

```bash
# Regenerate all plots and tables from raw logs/results
.venv/Scripts/python.exe paper/make_assets.py
```

## Folder Structure

```
paper/
├── README.md                  ← this file
├── paper_context.md           ← MASTER REFERENCE: methodology, all numbers, journal strategy
├── make_assets.py             ← reproducible plot + table generation script
│
├── figures/                   ← all figures (30 total)
│   ├── fig_01_pipeline_overview.png       ← Fig 1: system diagram
│   ├── fig_02_seg_qualitative.png         ← Fig 3: segmentation samples
│   ├── fig_03_seg_training_curves.png     ← Fig 4a: seg training
│   ├── fig_04_seg_ablation.png            ← Fig 4b: decoder ablation
│   ├── fig_05_rl_architecture.png         ← Fig 2: GRPO architecture
│   ├── fig_06_rl_confusion.png            ← Fig 5a: confusion matrix
│   ├── fig_08_reward_components.png       ← Fig 5b: reward breakdown
│   ├── fig_11_physics_consistency.png     ← Fig 6: XAI physics
│   ├── fig_13_spatial.png                 ← Fig 7: spatial predictions
│   ├── fig01_study_area.png               ← study area map
│   ├── fig02_temporal_triplets.png        ← temporal triplet examples
│   ├── fig05_patch_grid_per_class.png     ← per-class patch examples
│   ├── xai_gradcam_grid.png               ← Grad-CAM heatmaps
│   ├── xai_attention_rollout.png          ← attention rollout
│   └── ... (30 figures total)
│
├── plots/                     ← newly generated analytical plots
│   ├── plot_grpo_training.png             ← GRPO loss + reward/F1 curves
│   ├── plot_grpo_perclass_f1.png          ← per-class F1 over epochs
│   ├── plot_seg_training.png              ← seg v1 vs v2 loss
│   ├── plot_generalization.png            ← cross-event accuracy/F1 bars
│   ├── plot_calibration_gap.png           ← confidence vs accuracy scatter
│   └── plot_dataset_composition.png       ← dataset stats (4 panels)
│
└── tables/                    ← publication tables (.md readable, .tex LaTeX)
    ├── table1_dataset.md/.tex             ← dataset composition + splits
    ├── table2_classes.md/.tex             ← risk class definitions
    ├── table3_segmentation.md/.tex        ← seg performance v1 vs v2
    ├── table4_grpo.md/.tex                ← GRPO training results
    ├── table5_generalization.md/.tex      ← cross-event generalization
    └── table6_hyperparams.md/.tex         ← full hyperparameter table
```

## Key Numbers (for abstract / intro)

| Metric | Value |
|---|---:|
| Dataset size | 31,098 patches |
| Training events | Sikkim South Lhonak 2016 |
| Train / Val / Test-temporal / Test-spatial | 3,400 / 600 / 4,000 / 642 |
| Sensors | LISS-3 (29,378) + LISS-4 (1,720) |
| Backbone | Prithvi-EO-2.0-300M |
| LoRA rank / alpha | 16 / 32 |
| Seg v1 best loss / IoU | 0.368 / 0.293 |
| Seg v2 best loss | 0.224 |
| GRPO best val macro-F1 | **0.862** (epoch 15) |
| GRPO best val reward | **0.920** |
| Temporal gen. accuracy | 58.3% |
| Spatial gen. accuracy (OOD) | 13–37% |
| Mean confidence (OOD) | 0.66–0.92 (overconfident) |

## Recommended Journal

**First submission**: *Remote Sensing* (MDPI) — IF 5.0, open access, fast review (~6 weeks), good fit for applied EO + deep learning.

**Stretch target**: *Remote Sensing of Environment* (IF 13.5) — requires addressing the generalization gap and adding baselines first.

See `paper_context.md` Section 12 for full feasibility analysis.
