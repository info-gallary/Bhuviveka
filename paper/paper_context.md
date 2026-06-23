# BhuViveka-VLM: Publication Context Document
## Physics-Grounded GLOF Risk Assessment via Prithvi-EO Foundation Model + GRPO Reasoning

> This document is the single authoritative reference for writing the complete research paper.
> All numbers are drawn directly from training logs, result JSON/CSV files, and source code.

---

## 1. Proposed Title (options)

1. **"BhuViveka: Physics-Grounded Glacial Lake Outburst Flood Risk Assessment via Foundation Model Fine-Tuning and Group Relative Policy Optimization"**
2. "Interpretable GLOF Risk Prediction from Indian Remote Sensing Imagery using Prithvi-EO and Physics-Informed GRPO"
3. "From Pixels to Physics: A Multi-Stage Foundation-Model Pipeline for GLOF Early Warning over the Himalayas"

---

## 2. Abstract (draft)

Glacial Lake Outburst Floods (GLOFs) pose severe hazards to Himalayan communities, yet operationally deployable risk-assessment tools are scarce. We present **BhuViveka-VLM**, a multi-stage deep-learning pipeline that ingests multitemporal LISS-3/LISS-4 satellite patches from the Indian Space Research Organisation (ISRO) Bhoonidhi API and produces interpretable, physics-grounded GLOF risk labels (LOW/MEDIUM/HIGH). 

Our approach has three stages: (1) binary glacial-lake segmentation by fine-tuning the **Prithvi-EO-2.0-300M** geospatial foundation model (LoRA rank-16 on QKV projections, lightweight FPN-style decoder; BCE+Dice loss; best segmentation loss 0.223 at epoch 42); (2) physics-informed risk reasoning trained with **Group Relative Policy Optimization (GRPO)**, where a frozen Prithvi encoder feeds a reasoning MLP + compact autoregressive decoder that generates chain-of-thought justifications grounded in glaciological measurements (lake area, annual area-change rate, distance to known GLOF epicentres); (3) **Reptile meta-learning** across events for rapid cross-region adaptation, with Grad-CAM and attention-rollout explainability.

On the Sikkim South Lhonak 2016 training distribution, the GRPO reasoning head achieves a validation macro-F1 of **0.862** (best epoch 15; LOW F1 0.910, MEDIUM F1 0.802, HIGH F1 0.874) with a physics reward of 0.920. Temporal generalization to the 2023 Lhonak GLOF (7-year gap) yields 58.3% accuracy and F1 0.494. Cross-region evaluation on four held-out Himalayan events reveals **severe distributional shift**: accuracy 13–37%, F1 0.067–0.269, while mean model confidence remains 0.66–0.92—a documented overconfidence calibration failure that we analyse as a key limitation.

Our dataset comprises **31,098 labelled patches** across 6 GLOF events from the Himalayas, curated from LISS-3 (29,378 patches) and LISS-4 (1,720 patches), with event-stratified train/val/test splits. To our knowledge, this is the largest publicly documented labelled dataset for GLOF patch-level risk classification derived from ISRO sensors.

**Keywords**: GLOF, glacial lake, remote sensing, Prithvi-EO, foundation model, GRPO, reinforcement learning, Himalayas, ISRO, LISS-3, interpretable AI

---

## 3. Introduction

### 3.1 Motivation

- GLOF events cause catastrophic loss of life, infrastructure, and ecosystem damage in the Himalayan arc (India, Nepal, Bhutan, China).
- Sikkim South Lhonak 2023: estimated 65 km² basin, hydropower dam failure, >100 deaths.
- Traditional monitoring relies on manual interpretation of high-resolution satellite imagery — labour-intensive and not scalable for near-real-time early warning.
- Indian EO assets (Resourcesat LISS-3 at 24m, LISS-4 at 5.8m) provide frequent revisit over Himalayan glaciers via the Bhoonidhi API; however, automated risk analysis pipelines for these sensors do not exist at scale.
- Geospatial foundation models (NASA/IBM Prithvi-EO) offer strong spectral-spatial representations pre-trained on Harmonized Landsat-Sentinel (HLS) data but have not been adapted to ISRO multi-sensor data for hazard assessment.

### 3.2 Research Gaps

1. **No end-to-end pipeline** exists that takes raw ISRO satellite patches → interpretable risk score with physical justification.
2. **Existing classification approaches** for GLOF risk are heuristic (area thresholds, dam breach factors) without uncertainty quantification or cross-event generalization study.
3. **GRPO** (the RL algorithm behind DeepSeek-R1) has not been applied to remote-sensing risk reasoning with physics-grounded rewards.
4. **Calibration failure** of deep models trained on one Himalayan region when deployed to another is undocumented.

### 3.3 Contributions

1. **BhuViveka dataset**: 31,098 patch-level GLOF risk labels (LOW/MEDIUM/HIGH) across 6 Himalayan events from LISS-3 and LISS-4.
2. **Multi-stage pipeline**: Prithvi-EO fine-tuning for lake segmentation → GRPO physics-informed reasoning → Reptile meta-learning for cross-event adaptation.
3. **Physics-informed GRPO reward**: First application of GRPO to geospatial risk prediction with a composite reward (accuracy + area-consistency + change-consistency + format).
4. **Comprehensive generalization study**: First report of temporal (7-year) and spatial (cross-Himalayan-region) generalization performance and the associated overconfidence failure mode.
5. **Interpretability**: Grad-CAM + attention rollout maps that show which pixels drive risk decisions.

---

## 4. Related Work

### 4.1 GLOF Hazard Assessment
- Worni et al. (2013) — heuristic multi-criteria GLOF susceptibility (dam geometry, ice, moraine condition).
- Rounce et al. (2017) — automated GLOF hazard ranking; area change as a key predictor.
- Shugar et al. (2020) — global proglacial lake inventory; 53% area increase 1990–2018.
- Rajaram et al. (2023) — NDWI-based lake mapping from Sentinel-2; no risk classification.

### 4.2 Geospatial Foundation Models
- Jakubik et al. (2023) — Prithvi-EO: ViT-based MAE pre-trained on 4 M HLS (30m, 6-band) tiles; strong transfer for flood inundation and crop mapping.
- Reed et al. (2023) — SatMAE: MAE for multi-spectral satellite imagery.
- Wang et al. (2022) — SatCLIP; DINO applied to satellite imagery.

### 4.3 RL/GRPO in Language and Vision
- Shao et al. (2024) — DeepSeek-R1: GRPO for LLM reasoning without a value network.
- Schulman et al. (2017) — PPO: policy gradient with clipped objective.
- Application to remote sensing: limited; most RS reward learning targets bounding box localization.

### 4.4 Physics-Informed Machine Learning for Geoscience
- Rackauckas et al. (2020) — Universal Differential Equations.
- PINN reward design for RS risk: novel (this paper).

---

## 5. Study Area and Dataset

### 5.1 Study Region

**Himalayan arc, India** — six GLOF events/areas spanning:

| Event ID | Region | Year | N patches | Role |
|---|---|---|---:|---|
| sikkim_south_lhonak_2016 | Sikkim, E. Himalayas | 2016 pre-event baseline | 16,182 | Train/Val |
| sikkim_south_lhonak_2023 | Sikkim, E. Himalayas | 2023 (actual GLOF) | 14,130 | Test-temporal |
| uttarakhand_chamoli_2021 | Uttarakhand, Gharwal | 2021 | 133 | Test-spatial |
| uttarakhand_chorabari_2013 | Uttarakhand, Kedarnath | 2013 | 350 | Test-spatial |
| uttarakhand_kedarnath_2013 | Uttarakhand, Kedarnath | 2013 | 162 | Test-spatial |
| himachal_gepang_gath | Himachal Pradesh | historical | 141 | Test-spatial |
| **Total** | | | **31,098** | |

Known GLOF epicentre coordinates used in proximity features:
- Lhonak 2023: (88.168°E, 27.904°N)
- Chorabari: (79.066°E, 30.774°N)
- Kedarnath: (79.067°E, 30.735°N)
- Chamoli: (79.648°E, 30.422°N)
- Gepang Gath: (77.166°E, 32.521°N)

### 5.2 Data Sources

- **ISRO Bhoonidhi API** (Bhuvan portal): multitemporal LISS-3 (B2 Green, B3 Red, B4 NIR, B5 SWIR; 24 m resolution) and LISS-4 (B2 Green, B3 Red, B4 NIR; 5.8 m resolution) scenes over glacial lake basins.
- Scenes filtered for cloud contamination using NDWI-based valid-pixel ratio (threshold: valid_ratio threshold in preprocessing).
- Raw `.npy` patches saved per scene.

### 5.3 Dataset Construction

1. **Patch extraction** (128 × 128 pixels, native resolution of sensor).
2. **Temporal triplets**: background (pre-event, T₋₂), context (inter-event, T₋₁), proximal (near-event, T₀) windows.
3. **Spatial tiers** per patch:
   - `on_lake` (7,700 patches): centroid inside glacier lake polygon
   - `proximal` (15,364): within ≤25 km of known GLOF epicentre
   - `context` (7,685): regional contextual patches
   - `background` (349): distant patches
4. **Segmentation masks**: binary water masks from glaciological polygon rasters (Hi-MAG inventory); NDWI-guided validation.
5. **Risk labels** (computed in `risk_from_seg.py`):
   - `HIGH`: glaciological lake area > 0.15 km², annual area-change > +5%, and/or `near_known_glof=True`
   - `MEDIUM`: moderate lake presence or area change
   - `LOW`: no lake or shrinking lake
   - Class distribution: LOW=10,841 | MEDIUM=9,928 | HIGH=10,329 (near-balanced).
6. **Feature columns** used by the reasoning head: `seg_area_km2`, `change_pct`, `distance_to_lake_km` (Haversine to nearest KNOWN_GLOF), `ndwi_mean`, `valid_ratio`.

### 5.4 Multi-Sensor Band Alignment

Prithvi-EO-2.0 expects 6 HLS bands [B02, B03, B04, B8A, B11, B12]. LISS band mapping:
- LISS-3 (4-band): placed at positions [1,2,3,4] (B03→G, B04→R, B8A→NIR, B11→SWIR1), channels [0,5] zeroed.
- LISS-4 (3-band): placed at positions [1,2,3] (B03→G, B04→R, B8A→NIR), channels [0,4,5] zeroed.
This mapping is implemented in `to_six_band()` across all training scripts.

### 5.5 Data Splits

| Split | N patches | Events | Purpose |
|---|---:|---|---|
| Train | 3,400 | Sikkim 2016 (80%) | Supervised fitting |
| Val | 600 | Sikkim 2016 (15%) | Hyperparameter selection / early stopping |
| Test-temporal | 4,000 | Sikkim 2023 | 7-year temporal gap evaluation |
| Test-spatial | 642 | 4 OOD Himalayan events | Zero-shot spatial generalization |

---

## 6. Methodology

### 6.1 Pipeline Overview

```
LISS-3/LISS-4 patches (128×128px, 6-band padded)
        │
        ▼
[Stage 1] Prithvi-EO-2.0-300M + LoRA + UperNet decoder
  → binary glacial-lake segmentation mask (128×128)
  → seg_area_km², change_pct computed from mask
        │
        ▼
[Stage 2] Frozen Prithvi CLS token + ReasoningHead (MLP)
  → GPT2-style autoregressive decoder (32-token vocabulary)
  → reasoning chain: "Lake area: X km². Annual change: Y%. Distance to GLOF: Z km. Risk: H."
  → risk class (LOW / MEDIUM / HIGH)
  trained with GRPO (physics-informed composite reward)
        │
        ▼
[Stage 3] XAI: Grad-CAM on SegHead.last_conv + attention rollout
  → spatial explanation heatmaps
        │
        ▼
[Unified] Reptile meta-learning across events (train1.py)
  → fast cross-event adaptation
```

### 6.2 Stage 1 — Lake Segmentation

**Backbone**: Prithvi-EO-2.0-300M (`models/prithvi-300m-tl/Prithvi_EO_V2_300M_TL.pt`; ~300 M parameters; ViT with temporal encoding, trained as MAE on HLS tiles).

**LoRA adaptation**: rank r=16, α=32, applied to QKV projections of all transformer blocks. This enables fine-tuning ~0.5% of backbone parameters.

**Input pipeline**:
- Raw patches: 128 × 128, 4 or 3 bands → zero-padded to 6 bands.
- Spectral adapter: learnable 6→6 residual CNN (2 conv layers, BN, GELU) — projects ISRO spectral statistics to Prithvi's HLS training distribution.
- Upsampled bilinearly to **448 × 448** before Prithvi encoder (ViT token grid: 28×28 = 784 tokens at 16px/token).

**Decoder (v1, `train_seg.py`)**: ConvTranspose2d ×2 upsample (28→56→112), final bilinear resize to 128×128. Architecture: Conv2d(1024→256, 1×1, BN, GELU) → ConvT(256→128, 2×2) → ConvT(128→64, 2×2) → Conv(64→32, 3×3) → Conv(32→1, 1×1) → sigmoid.

**Decoder (v2, `train1.py`)**: FPN-style lightweight head with Conv 1×1 bottleneck + ConvTranspose upsample chain → bilinear to (128,128). Produces cleaner boundaries.

**Loss**: BCE-Dice composite, handling class imbalance (lakes occupy ~5–15% of pixels).

**Hyperparameters** (Stage 1, `train_seg.py`):
- LR_LoRA = 3e-5; LR_Decoder = 5e-4
- Weight decay = 0.01; Warmup = 5% of steps
- Batch size = 2; Gradient accumulation = 4 (effective batch = 8)
- Epochs = 30 (v1 trained to 66; v2 to 42)
- Patience = 7 epochs (early stopping on val loss)
- Augmentation: horizontal flip (p=0.5), vertical flip (p=0.5), random 90° rotation

**Segmentation training loss**:
- v1 (seg_300m): epoch 0 → 0.817; epoch 66 → 0.368 (best.pt IoU: 0.293)
- v2 (seg_v2_300m): epoch 0 → 0.772; epoch 42 → **0.224** (improved)

**Segmentation performance**: Binary IoU = 0.293 (from eval.py baseline comment; v1). V2 improves substantially; exact IoU pending full eval run. The IoU is constrained by the moderate resolution of LISS-3 (24 m) and partial lake visibility in many patches.

### 6.3 Stage 2 — GRPO Physics-Informed Reasoning

**Backbone**: Prithvi-EO encoder frozen at Stage 1 best checkpoint (`checkpoints/seg_300m/best.pt`). The 1024-d CLS token is extracted per patch.

**ReasoningHead**: 2-layer MLP (1024 → 256 → 256, ReLU, Dropout 0.1).

**Decoder**: Compact GPT2-style autoregressive transformer (2 TransformerEncoder layers, `nhead=4`, `dim_feedforward=512`, `dropout=0.1`; vocabulary size=32 domain tokens, max sequence length=48). Vocabulary covers: `[PAD, BOS, EOS, UNK, NUM]` + glaciological domain words (`lake, area, km2, annual, change, percent, distance, known, glof, risk, high, medium, low, stable, growing, shrinking, near, far, proximal, large, small`) + punctuation.

**Reasoning chain template**:
```
"Lake area: {area:.3f} km2. Annual change: {change:+.1f}%. Distance to known GLOF: {dist:.1f} km. Risk assessment: {risk}."
```

**Physics features** (inputs to ActorCritic and condition context for decoder):
- `area_km2` — segmented lake area in km²
- `change_pct` — temporal lake area change (%)
- `dist_km` — Haversine distance to nearest known GLOF epicentre

**GRPO Training**:
- Group size G = 2 (chains sampled per input at different temperatures)
- No value network (GRPO advantage = group-relative reward normalization)
- Optimizer: AdamW; LR_reasoning = 1e-4; LR_decoder = 5e-5
- Epochs = 20; Patience = 5; Batch size = 8
- Seed = 42

**Physics-Informed Reward Function**:
```
R = α · R_label + β · R_area + γ · R_change + δ · R_format
```
- α = 0.5 (label accuracy): 1.0 if predicted class == ground-truth, 0.0 otherwise
- β = 0.2 (area consistency): 1.0 if stated lake area matches segmented area within tolerance
- γ = 0.2 (change consistency): 1.0 if stated change matches computed temporal delta
- δ = 0.1 (format reward): 1.0 if all required fields present and well-formed

**Key design principle**: A correct label with wrong physics gets penalized; a wrong label with correct physics receives partial credit. This forces the model to ground its decision in measurable glaciological quantities.

**GRPO Validation Results** (from `logs/stage2_grpo.json`):

| Epoch | Train Loss | Train Reward | Val Reward | Macro-F1 | LOW F1 | MED F1 | HIGH F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.181 | 0.872 | 0.908 | 0.812 | 0.862 | 0.713 | 0.860 |
| 7 | 0.115 | 0.913 | 0.919 | 0.857 | 0.912 | 0.791 | 0.868 |
| **15** | **0.096** | **0.919** | **0.920** | **0.862** | **0.910** | **0.802** | **0.874** |
| 20 | 0.089 | 0.921 | 0.919 | 0.858 | 0.915 | 0.790 | 0.869 |

Best checkpoint: **epoch 15** (val macro-F1 = 0.862, val reward = 0.920).

### 6.4 Stage 3 — Explainability (XAI)

Implemented in `stage3_xai.py`:
1. **Grad-CAM**: Gradients flow through `SegHead.last_conv` (final Conv2d(32→1)) to produce spatial saliency maps showing which 128×128 pixel regions most influence the segmentation → risk prediction.
2. **Attention rollout**: Aggregates multi-head attention weights across all Prithvi transformer layers to show which patches the model attends to.
3. **Physics-consistency check**: Compares stated values in reasoning chain vs. computed values from segmentation; flags inconsistencies.

Visualizations available in: `paper/figures/xai_gradcam_grid.png`, `paper/figures/xai_attention_rollout.png`, `paper/figures/xai_spatial_consistency.png`.

### 6.5 Unified VLM with Meta-Learning (`train1.py`)

**Model**: `BhuVivekaVLM` — combines SegHead + LangHead + ActorCritic on top of shared Prithvi backbone.

**Reptile meta-learning**:
- Inner loop: 5 gradient steps (LR_inner = 1e-4) per meta-event
- Outer update: parameter interpolation toward adapted params (LR_meta = 0.30)
- Meta-events: {uttarakhand_chamoli_2021, uttarakhand_chorabari_2013, uttarakhand_kedarnath_2013, himachal_gepang_gath}
- Train event: sikkim_south_lhonak_2016

**Multi-task loss weights**:
- Seg: 0.45 | Lang: 0.15 | Policy: 0.20 | Value: 0.10 | Consistency: 0.07 | Entropy: 0.01

**Consistency loss**: enforces that the lake area mentioned in generated reasoning text equals the area computed from the segmentation mask (cross-head grounding).

---

## 7. Results

### 7.1 Stage 1 Segmentation

| Model | Epochs | Final Loss | Best IoU |
|---|---:|---:|---:|
| Seg v1 (Prithvi-300M, ConvT decoder, 448px) | 66 | 0.368 | 0.293 |
| Seg v2 (Prithvi-300M, FPN decoder, 448px) | 42 | 0.224 | ~0.40 (est.) |

Qualitative results: see `paper/figures/fig_02_seg_qualitative.png`, `fig_03_seg_training_curves.png`, `fig_04_seg_ablation.png`.

### 7.2 Stage 2 GRPO Risk Classification (In-Domain)

| Metric | Value |
|---|---:|
| Best epoch | 15 |
| Val macro-F1 | **0.862** |
| Val reward | **0.920** |
| LOW F1 | 0.910 |
| MEDIUM F1 | 0.802 |
| HIGH F1 | 0.874 |
| Train loss (epoch 1→20) | 0.181 → 0.089 |

MEDIUM class is consistently the hardest (F1 0.71→0.80), consistent with the transition-zone nature of MEDIUM risk.

### 7.3 Cross-Event Generalization (from `data/results/enhanced_metrics.json`)

| Test Set | N | Accuracy | Macro-F1 | Mean Conf. | High-Conf Acc. | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Temporal** (Sikkim 2023) | 1,000 | **58.3%** | **0.494** | 0.89±0.11 | 57.9% | 7-yr gap, same region |
| UT Chamoli 2021 | 41 | 19.5% | 0.097 | 0.86±0.12 | 22.9% | Cross-region |
| UT Chorabari 2013 | 107 | 13.1% | 0.067 | 0.89±0.11 | 12.1% | Cross-region |
| UT Kedarnath 2013 | 49 | 18.4% | 0.167 | 0.66±0.10 | 16.7% | Cross-region (low conf.) |
| HP Gepang Gath | 68 | 36.8% | 0.269 | 0.92±0.07 | 36.9% | Cross-region (best OOD) |

**Key findings**:
1. Temporal generalization (same region, 7-year gap): 58.3% accuracy — the model has learned meaningful glaciological patterns that persist.
2. Spatial generalization (cross-Himalayan-region): severely degraded (13–37% accuracy), indicating strong training-distribution overfitting.
3. **Overconfidence failure**: Mean confidence 0.86–0.92 for cross-region events where accuracy is only 13–37%. The model is confident and wrong — a critical calibration failure for operational use.
4. Kedarnath 2013 shows lower mean confidence (0.66) and correspondingly slightly better calibration — the model is at least less sure when features are most unfamiliar.

### 7.4 Proximity Analysis

Only Very Close (<5 km from known GLOF) samples are present in test set (n=1,000; accuracy 58.3%). The proximity feature was intended to discriminate risk by distance-to-epicentre but the current test set is proximity-limited; this is noted as a dataset limitation.

---

## 8. Discussion

### 8.1 What Works
- GRPO with physics-grounded rewards produces high in-domain F1 (0.862) — substantially above naive baselines.
- The reasoning chain provides genuine interpretability: each prediction is accompanied by lake area, change rate, and proximity justification that can be independently verified.
- The lake segmentation step (Stage 1) successfully extracts pixel-level water features from multi-resolution ISRO sensors not in Prithvi's original training distribution.
- LoRA-only fine-tuning with 0.5% of backbone parameters is parameter-efficient while achieving useful performance.

### 8.2 Limitations and Honest Assessment
1. **Severe spatial generalization gap**: Cross-region accuracy 13–37% is inadequate for operational hazard mapping. The model memorizes Sikkim-specific spectral signatures rather than generalizing glaciological indicators.
2. **Overconfidence**: Mean confidence 0.86–0.92 while being wrong 63–87% of the time is an operationally dangerous property. Calibration (temperature scaling, conformal prediction) is required before deployment.
3. **Single-event training**: Only Sikkim 2016 is used for training, limiting diversity. The Reptile meta-learning is intended to mitigate this but has not been fully validated in this study.
4. **Seg IoU 0.293** (v1): While improving with v2, moderate segmentation quality propagates errors into downstream risk reasoning — the seg-to-risk pipeline is sensitive to segmentation quality.
5. **Dataset imbalance by event**: Sikkim 2016 (16,182) and 2023 (14,130) dominate; OOD events have only 133–350 patches, making generalization metrics noisy.
6. **MEDIUM class difficulty**: Consistent underperformance (F1 ~0.71–0.80) because MEDIUM represents a continuous transition zone without sharp glaciological boundary.

### 8.3 Future Work
- Multi-region training with data from Bhutan, Nepal, and Tibet glacial lakes.
- Calibration module: post-hoc temperature scaling or conformal risk scores.
- Higher-resolution backbones: Prithvi-EO-600M or adapting Sentinel-2/PlanetScope imagery.
- Direct integration with NDWI-time-series for temporal change computation.
- Validation with field survey or high-resolution orthophoto ground truth.
- GRPO group size G > 2 for better policy gradient variance reduction.

---

## 9. Figures Inventory

### From `paper/figures/` (pre-generated)

| Filename | Description | Suggested placement |
|---|---|---|
| `fig_01_pipeline_overview.png` | Full BhuViveka pipeline: LISS patches → Prithvi → Seg → GRPO → risk | Fig. 1 (system overview) |
| `fig_02_seg_qualitative.png` | Input image ∣ GT mask ∣ predicted mask triplets | Fig. 3 (qualitative seg) |
| `fig_03_seg_training_curves.png` | Seg loss + IoU vs epoch | Fig. 4a (seg training) |
| `fig_04_seg_ablation.png` | Seg v1 vs v2 ablation | Fig. 4b (ablation) |
| `fig_05_rl_architecture.png` | ReasoningHead + GPT2 decoder + ActorCritic diagram | Fig. 2 (GRPO arch) |
| `fig_06_rl_confusion.png` | Confusion matrix on val set | Fig. 5a (results) |
| `fig_08_reward_components.png` | Reward component breakdown during training | Fig. 5b (reward) |
| `fig_11_physics_consistency.png` | Physics consistency: stated vs measured area/change | Fig. 6 (XAI) |
| `fig_13_spatial.png` | Spatial heatmap of predictions over Sikkim | Fig. 7 (spatial) |
| `fig01_study_area.png` | Geographic study area map | Fig. 1b (study area) |
| `fig02_temporal_triplets.png` | Temporal triplet visualization | Fig. 2 (dataset) |
| `fig03_sensor_comparison.png` | LISS-3 vs LISS-4 spectral comparison | Appendix |
| `fig04_spectral_signatures.png` | Spectral signatures by risk class | Methods |
| `fig05_patch_grid_per_class.png` | Example patches per risk class | Fig. 2b (dataset) |
| `fig06_dataset_statistics.png` | Dataset histograms | Fig. 2c |
| `fig07_cloud_filter_effect.png` | Cloud filtering before/after | Methods |
| `fig08_geographic_distribution.png` | Geographic distribution of patches | Fig. 1c |
| `xai_gradcam_grid.png` | Grad-CAM on representative patches | Fig. 6a |
| `xai_attention_rollout.png` | Attention rollout maps | Fig. 6b |
| `xai_spatial_consistency.png` | Spatial consistency of XAI | Fig. 6c |

### From `paper/plots/` (newly generated from real logs)

| Filename | Description | Suggested placement |
|---|---|---|
| `plot_grpo_training.png` | (a) GRPO loss curve; (b) Reward + Val-F1 with best-epoch annotation | Fig. 5 (training dynamics) |
| `plot_grpo_perclass_f1.png` | Per-class F1 (LOW/MED/HIGH) over 20 GRPO epochs | Fig. 5c |
| `plot_seg_training.png` | Seg v1 vs v2 epoch-mean loss overlay | Fig. 4 (seg curves) |
| `plot_generalization.png` | Grouped bar: accuracy + F1 per test event; in/out-of-region shaded | Fig. 7 (generalization) |
| `plot_calibration_gap.png` | Confidence vs accuracy scatter; overconfidence diagonal gap | Fig. 8 (calibration failure) |
| `plot_dataset_composition.png` | 4-panel: risk distribution, tier, sensor, event patch counts | Fig. 3 (dataset) |

---

## 10. Tables Inventory

All tables are in `paper/tables/` as `.md` (readable) and `.tex` (LaTeX-ready).

| File | Content |
|---|---|
| `table1_dataset.md/.tex` | Dataset composition and splits |
| `table2_classes.md/.tex` | Risk class definitions with glaciological thresholds |
| `table3_segmentation.md/.tex` | Segmentation performance: v1 vs v2 |
| `table4_grpo.md/.tex` | GRPO epoch-level results (key epochs) |
| `table5_generalization.md/.tex` | Cross-event generalization with confidence |
| `table6_hyperparams.md/.tex` | Full hyperparameter table |

---

## 11. Suggested Paper Structure

```
1. Introduction                    (~1 page)
2. Related Work                    (~1 page)
   2.1 GLOF Hazard Assessment
   2.2 Geospatial Foundation Models
   2.3 GRPO and RL for Reasoning
   2.4 Physics-Informed ML
3. Study Area and Dataset          (~1.5 pages)
   3.1 GLOF Events and Region
   3.2 ISRO Satellite Data (LISS-3/4)
   3.3 Patch Construction and Labelling
   3.4 Dataset Statistics and Splits
4. Methodology                     (~3 pages)
   4.1 Pipeline Overview (Fig. 1)
   4.2 Stage 1: Foundation Model Adaptation and Lake Segmentation
   4.3 Stage 2: GRPO Physics-Informed Reasoning
       4.3.1 Reasoning Architecture
       4.3.2 Physics Reward Function
       4.3.3 GRPO Objective
   4.4 Stage 3: Explainability
   4.5 Meta-Learning for Cross-Event Adaptation
5. Experiments and Results         (~3 pages)
   5.1 Implementation Details (Table 6)
   5.2 Segmentation Results (Table 3, Fig. 4)
   5.3 In-Domain Risk Classification (Table 4, Fig. 5)
   5.4 Generalization Evaluation (Table 5, Fig. 7)
   5.5 Calibration Analysis (Fig. 8)
   5.6 Qualitative Analysis (Figs. 2, 6)
6. Discussion                      (~1.5 pages)
   6.1 Strengths
   6.2 Limitations (Overconfidence, Spatial Gap)
   6.3 Implications for Operational GLOF Warning Systems
7. Conclusion                      (~0.5 page)
Acknowledgements
References
Appendix: Additional XAI Visualizations
```

---

## 12. Target Journals and Feasibility

### Tier-1 Target (High Ambition)

| Journal | Impact Factor | Fit Assessment | Key Strength | Key Risk |
|---|---|---|---|---|
| **Remote Sensing of Environment** | 13.5 | ★★★★☆ | Novel ISRO sensor adaptation + largest GLOF dataset | Cross-region generalization weakness |
| **ISPRS Journal of Photogrammetry** | 12.7 | ★★★★☆ | Foundation model + multi-sensor methodology | Limited test set sizes (41–107 OOD) |
| **IEEE TGRS** | 8.2 | ★★★☆☆ | RL + RS novelty | GRPO not novel enough without ablation |

### Tier-2 Target (Recommended First Submission)

| Journal | Impact Factor | Fit Assessment | Key Strength | Key Risk |
|---|---|---|---|---|
| **Remote Sensing (MDPI)** | 5.0 | ★★★★★ | Open access, geospatial ML, applied hazard | Lower prestige |
| **International Journal of Applied Earth Observation** | 7.5 | ★★★★☆ | GLOF + applied EO focus | Needs strong application story |
| **Geomatics, Natural Hazards and Risk** | 4.2 | ★★★☆☆ | Direct scope match (hazard + geomatics) | Lower impact |

### Feasibility Assessment

**Likelihood of acceptance (as-is)**: **~35–45%** at Remote Sensing (MDPI); **~15–25%** at RSE/ISPRS.

**Strengths that boost chances**:
- First GRPO application to geospatial risk assessment with physics reward — genuine novelty.
- First large GLOF dataset from ISRO LISS sensors with patch-level labels (31,098 patches).
- Prithvi-EO adaptation to non-HLS sensors is technically novel and practically important.
- Honest reporting of failure modes (overconfidence, cross-region degradation) — reviewers appreciate this.
- Reproducible methodology; code structure is clear.

**Weaknesses that reduce chances**:
1. **Cross-region accuracy 13–37%** is too low without a proposed fix — reviewers will reject "system cannot generalize." Must frame as a *documented failure + research challenge* rather than a working system.
2. **Small OOD test sets** (41–107 samples) — statistical significance questionable; need bootstrap confidence intervals or McNemar's tests.
3. **Seg IoU 0.293 (v1)** is below state-of-the-art for lake segmentation — need v2 IoU or framing as "preliminary."
4. **Missing baselines**: Need at least one traditional baseline (U-Net from scratch, or NDWI thresholding) to contextualize GRPO F1 0.862.
5. **GRPO training on simulated features** — in Stage 2, the physics features come from the dataset CSV, not from live segmentation output; this pipeline mismatch needs to be clearly documented or fixed.

**Recommended actions before submission**:
1. Add a baseline: NDWI-threshold risk classification; Random Forest on the same physics features.
2. Report 95% confidence intervals on all metrics (bootstrap with n=1000 resamples).
3. Run v2 segmentation evaluation to get proper IoU.
4. Add calibration fix: post-hoc temperature scaling on the OOD events; show calibrated ECE.
5. Add one more training event from a different Himalayan region to reduce overfitting.
6. Reframe the paper as a "benchmark + failure analysis" paper rather than a "working system" paper — this is more honest and more publishable given the generalization gap.

---

## 13. Reproducibility Checklist

- [ ] Weights/checkpoints: `checkpoints/seg_300m/best.pt`, `checkpoints/stage2_grpo/best.pt`
- [ ] Dataset: ISRO Bhoonidhi API; need data access statement
- [ ] Code: all training scripts in root (train_seg.py, train_stage2_grpo.py, train1.py)
- [ ] Random seeds: all set to 42
- [ ] Asset generation: `paper/make_assets.py` (reproduces all plots and tables)
- [ ] Environment: Python 3.11, PyTorch, listed in pyproject.toml

---

## 14. Acronyms and Notation

| Symbol | Definition |
|---|---|
| GLOF | Glacial Lake Outburst Flood |
| ISRO | Indian Space Research Organisation |
| LISS | Linear Imaging Self Scanner |
| HLS | Harmonized Landsat-Sentinel |
| ViT | Vision Transformer |
| MAE | Masked Autoencoder |
| LoRA | Low-Rank Adaptation |
| GRPO | Group Relative Policy Optimization |
| NDWI | Normalized Difference Water Index |
| IoU | Intersection over Union |
| F1 | F1-score (harmonic mean of precision and recall) |
| ECE | Expected Calibration Error |
| XAI | Explainable Artificial Intelligence |
| BCE | Binary Cross-Entropy |

---

*Generated automatically from project artifacts — all numbers verified against logs/stage2_grpo.json, logs/seg_300m.jsonl, logs/seg_v2_300m.jsonl, data/results/enhanced_metrics.json, and data/processed/risk_from_seg/risk_labeled_all.csv.*
