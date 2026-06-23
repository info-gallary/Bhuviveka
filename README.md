# BhuViveka: Physics-Grounded Natural-Language GLOF Risk Reasoning via Prithvi-EO Adaptation over Indian Resourcesat LISS Imagery

Source code, dataset documentation, and paper artefacts for:

> **N. Vashi, K. Vaghasia, J. Saxena.** *"BhuViveka: Physics-Grounded Natural-Language GLOF Risk Reasoning via Prithvi-EO Foundation Model Adaptation and Deep Reinforcement Learning over Indian Resourcesat LISS Imagery."* 2026.

The full manuscript is at [paper/BhuViveka_IEEE_Conference.pdf](paper/BhuViveka_IEEE_Conference.pdf). The master methodology, all reported numbers, and the journal-strategy document live at [paper/paper_context.md](paper/paper_context.md).

> **Status (2026-06-23):** preliminary release accompanying a workshop-grade preprint. Several experiments noted in the paper's §VI-D — leakage-free Stage 2 evaluation, end-to-end Stage 1 → Stage 2 numbers, baseline comparisons, ablations, and the bug-corrected cross-region run — are ongoing and will land in a follow-up revision.

---

## Contents

| Path | Purpose |
|---|---|
| [paper/](paper/) | Camera-ready PDF, figures, plots, tables, and the full paper-context reference |
| [src/data/](src/data/) | NRSC Bhoonidhi + ICIMOD/HMA inventory download, preprocessing, dataset construction, segmentation-mask and risk-label generation |
| [src/models/](src/models/) | Prithvi-EO-2.0-300M backbone wrapper and physics-based risk scoring head |
| [src/training/](src/training/) | Stage-1 lake segmentation training (`train_seg.py`), Stage-2 GRPO reasoning training (`train_stage2_grpo.py`), and the unified end-to-end variant (`train1.py`) |
| [src/evaluation/](src/evaluation/) | Held-out and cross-event evaluation, tile inference, three-patch inference |
| [src/xai/](src/xai/) | Grad-CAM, attention rollout, physics-consistency audit |
| [src/viz/](src/viz/) | Figure-generation utilities |
| [experiments/](experiments/) | Ablation and baseline experiment drivers |
| [inference_app/](inference_app/) | Streamlit demo app with checkpoints and sample inputs |

---

## Quick start

```bash
# 1. Create the environment (uv)
uv sync

# 2. Pre-populate credentials (see .env.example)
cp .env.example .env
# edit .env to add your BHOONIDHI_USER, BHOONIDHI_PASS, HF_TOKEN, MODELS_DIR

# 3. Download Prithvi-EO-2.0-300M weights
uv run python src/data/prithvi.py

# 4. Pull Bhoonidhi imagery for the six configured GLOF events
uv run python src/data/download.py

# 5. Preprocess + build the labelled dataset
uv run python src/data/preprocess.py
uv run python src/data/dataset.py
uv run python src/data/labels.py
uv run python src/data/make_seg_masks.py
uv run python src/data/balance.py

# 6. Train Stage-1 (lake segmentation)
uv run python src/training/train_seg.py

# 7. Train Stage-2 (GRPO physics-informed reasoning)
uv run python src/training/train_stage2_grpo.py

# 8. Evaluate + run XAI
uv run python src/evaluation/eval.py
uv run python src/evaluation/generalisation.py
uv run python src/xai/stage3_xai.py

# 9. Regenerate every figure and table from logs
uv run python paper/make_assets.py

# 10. Launch the interactive inference demo
cd inference_app && uv run streamlit run app.py
```

> Note: `paper/make_assets.py` reads `logs/*.json{l}` and `data/results/*.json` produced by the training scripts. Both directories are intentionally `.gitignore`d — copy them in from your training environment to regenerate.

---

## Reproducing paper results

Single command to regenerate every figure (`paper/plots/`) and every table (`paper/tables/*.md` and `*.tex`) from cached training logs:

```bash
uv run python paper/make_assets.py
```

To re-run the underlying experiments end-to-end (≈24 h on a single 8 GB GPU):

```bash
uv run python experiments/baselines.py    # foundation-model bake-off
uv run python experiments/ablations.py    # reward-component ablations
uv run python experiments/e2e_eval.py     # leave-one-event-out pipeline
```

---

## Citing this work

```bibtex
@inproceedings{vashi2026bhuviveka,
  title  = {BhuViveka: Physics-Grounded Natural-Language GLOF Risk Reasoning
            via Prithvi-EO Foundation Model Adaptation and Deep Reinforcement
            Learning over Indian Resourcesat LISS Imagery},
  author = {Vashi, Nisarg and Vaghasia, Krish and Saxena, Jaya},
  year   = {2026},
  note   = {See paper/BhuViveka_IEEE_Conference.pdf in this repository}
}
```

---

## Authors

- **Nisarg Vashi** — Institute of Technology, Nirma University, Ahmedabad — `23bce370@nirmauni.ac.in`
- **Krish Vaghasia** — Institute of Technology, Nirma University, Ahmedabad — `23bce364@nirmauni.ac.in`
- **Dr. Jaya Saxena** — Scientist "SF" & Head, Student Project Interface Division, Training, Education & Outreach Group, Management Systems Area (MSA), **NRSC (ISRO), Hyderabad** — `jaya.saxena@nrsc.gov.in`

This work was conducted as a B.Tech student project under the mentorship of Dr. J. Saxena via the NRSC Student Project Interface Division.

---

## Acknowledgements

- **NRSC / ISRO Bhoonidhi** for Resourcesat-2A LISS-3 / LISS-4 imagery access.
- **NASA / IBM** for the open release of Prithvi-EO-2.0.
- **ICIMOD** for the Hi-MAG glacial-lake inventory used as ground truth.

---

## Licence

Source code: MIT — see [LICENSE](LICENSE).
Third-party assets (Prithvi weights, NRSC imagery, ICIMOD inventories) retain their upstream licences. Please respect those terms when redistributing.
