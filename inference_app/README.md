# BhuViveka-VLM — Inference App (POC)

Self-contained, **inference-only** package for the two-stage GLOF (Glacial Lake
Outburst Flood) risk pipeline. Nothing here depends on the training code base.

```
inference_app/
├── app.py                    # Streamlit POC (pick a sample / upload .npy → run)
├── inference.py              # load_models() + predict()  — standalone
├── models.py                 # all network definitions inlined
├── requirements.txt
├── checkpoints/
│   ├── seg_300m_best.pt      # Stage 1: Prithvi-300M + LoRA + decoder (~1.2 GB)
│   └── grpo_best.pt          # Stage 2: ReasoningHead (~1.6 MB)
├── samples/
│   ├── samples.csv           # metadata for the bundled patches
│   └── *.npy                 # 15 real LISS-3/4 patches (LOW / MEDIUM / HIGH)
└── scripts/                  # full reproducible pipeline (preprocess → train → eval)
    └── README.md             # pipeline order + per-script roles
```

## Reproducing the models

`scripts/` holds the complete source pipeline used to build the checkpoints —
data download, preprocessing, mask generation, risk labelling, Stage-1 / Stage-2
training, and evaluation/XAI. See [`scripts/README.md`](scripts/README.md) for
the run order. (These require the raw `data/` tree and the Prithvi backbone,
which are not shipped here.)

> The Stage-1 checkpoint already embeds the full backbone weights, so the
> original 1.3 GB `Prithvi_EO_V2_300M_TL.pt` file is **not** needed here.

## Pipeline

| Stage | Model | Output |
|---|---|---|
| 1 | Prithvi-EO-2.0-300M (LoRA) + lightweight decoder | per-pixel lake mask → lake area |
| 2 | ReasoningHead (visual CLS + physics features) | risk class `LOW / MEDIUM / HIGH` + confidence |

Physics features into Stage 2: predicted lake area (km², from the mask),
annual area change (%), distance to nearest known GLOF (km).

## Run the demo

```bash
cd inference_app
pip install -r requirements.txt          # or use the project's .venv
streamlit run app.py
```

With the existing project venv (no install needed):

```bash
cd d:/A-FM/inference_app
../.venv/Scripts/python.exe -m streamlit run app.py
```

Pick a bundled patch (or upload your own `.npy` shaped `(bands, 128, 128)`),
adjust the physics sliders, and press **Run Inference**.

## Programmatic use

```python
from inference import load_models, predict
models = load_models()                      # ~30 s, once
out = predict(models, "samples/<file>.npy", sensor="LISS3",
              change_pct=12.0, dist_km=3.4)
print(out["risk_class"], out["confidence"], out["pred_area_km2"])
```

Runs on CPU or GPU automatically (`cuda` if available).

### `predict()` output contract

`predict(models, npy_path, sensor, change_pct=0.0, dist_km=99.0)` returns a dict:

| Key | Type | Meaning |
|---|---|---|
| `rgb_image` | `uint8 (H,W,3)` | display-ready R-G-NIR composite |
| `seg_mask` | `float32 (H,W)` | lake probability map `[0,1]` |
| `seg_mask_bin` | `uint8 (H,W)` | binary mask at threshold 0.5 |
| `pred_area_km2` | `float` | lake area from the mask (sensor pixel size) |
| `risk_class` / `risk_id` | `str` / `int` | `LOW`/`MEDIUM`/`HIGH` and `0`/`1`/`2` |
| `confidence` | `float` | softmax prob of the predicted class |
| `probs` | `dict` | per-class probabilities |
| `reasoning` | `str` | templated physics explanation |
| `physics` | `dict` | the `area / change / dist` values fed to Stage 2 |

Input `.npy` must be shaped `(bands, 128, 128)`: LISS-3 = 4 bands, LISS-4 = 3 bands
(see `to_six_band()` in [inference.py](inference.py) for the band→Prithvi mapping).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: seg_300m_best.pt` | The 1.2 GB checkpoint is git-ignored; fetch it via Git LFS / external host into `checkpoints/`. |
| `ModuleNotFoundError: peft` | `pip install -r requirements.txt`. |
| First inference is slow (~20–30 s) | One-time model load; Streamlit caches it via `@st.cache_resource`. |
| Out of GPU memory | Force CPU: `load_models(device="cpu")`. |
| Uploaded patch errors | Confirm shape `(bands,128,128)` and pick the matching sensor in the sidebar. |

## Extending & improving

Entry points for future work, smallest change first:

- **Swap in a better checkpoint** — drop a new file in `checkpoints/` and update
  `SEG_CKPT` / `GRPO_CKPT` in [inference.py](inference.py). It must load into the
  same architecture (`load_state_dict` is strict); if you change the architecture,
  update [models.py](models.py) to match.
- **Add demo samples** — copy `.npy` patches into `samples/` and add a row to
  `samples/samples.csv` (`file, sensor, risk_label, event, …`). They appear in the
  sidebar automatically.
- **Tune risk behaviour** — risk classes/colors live at the top of
  [inference.py](inference.py) and [app.py](app.py); the segmentation threshold
  (`> 0.5`) and pixel→area constants (`PIXEL_M2`) are in `predict()`.
- **Support a new sensor / band layout** — extend `to_six_band()` and `PIXEL_M2`
  in [inference.py](inference.py), and add the option to the sidebar `sensor`
  selectbox in [app.py](app.py).
- **Change the UI** — [app.py](app.py) is plain Streamlit: `sidebar()` builds
  inputs, `render_result()` draws the three image panels + risk card. No model
  code lives here.
- **Retrain / change the models** — work in [scripts/](scripts/) (the full
  reproducible pipeline), produce new checkpoints, then re-point `inference.py`.
  See [scripts/README.md](scripts/README.md) for the run order.

**Architecture map** (where each piece lives):

```
app.py        UI only        → calls inference.predict()
inference.py  orchestration   → preprocessing, load_models(), predict()
models.py     definitions     → PrithviViTEncoder, BhuVivekaSeg (Stage 1),
                                ReasoningHead (Stage 2), build_lora_backbone()
scripts/      reproducibility → data → train → eval (not needed at runtime)
```

**Conventions for contributions**

- Keep `app.py` / `inference.py` / `models.py` runtime-only — no training or data
  prep imports, so the folder stays portable.
- If you touch `models.py`, keep class names and submodule layout identical to the
  checkpoint keys, or the strict `load_state_dict` will fail.
- Don't commit large weights (>100 MB) directly — they're git-ignored; use Git LFS.
