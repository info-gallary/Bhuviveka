"""
BhuViveka-VLM — GLOF Risk Prediction POC (self-contained)
==========================================================
Pick a bundled satellite patch OR upload your own .npy, then run the
two-stage pipeline live:
  Stage 1  Prithvi-EO-300M (LoRA) lake segmentation
  Stage 2  GRPO physics-grounded risk classification  (LOW / MEDIUM / HIGH)

Run:
  cd d:/A-FM/inference_app
  streamlit run app.py
  # or, with the project venv:
  ../.venv/Scripts/python.exe -m streamlit run app.py
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
MANIFEST = SAMPLES / "samples.csv"
CLASSES = ["LOW", "MEDIUM", "HIGH"]

RISK_COLOR = {"LOW": "#27ae60", "MEDIUM": "#e67e22", "HIGH": "#c0392b"}
RISK_BG    = {"LOW": "#eafaf1", "MEDIUM": "#fef9e7", "HIGH": "#fdedec"}
RISK_ICON  = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}

EVENT_LABELS = {
    "sikkim_south_lhonak_2016": "Sikkim S. Lhonak 2016 (in-domain)",
    "sikkim_south_lhonak_2023": "Sikkim S. Lhonak 2023 (temporal test)",
    "himachal_gepang_gath": "Himachal Gepang Gath (out-of-distribution)",
}

st.set_page_config(page_title="BhuViveka-VLM — GLOF Risk POC",
                   page_icon="🏔️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .risk-card { padding:18px 22px; border-radius:12px; margin-bottom:12px; border-left:6px solid; }
  .metric-box { background:#f8f9fa; border-radius:8px; padding:12px 16px; text-align:center; }
  .reasoning-box { background:#f0f4ff; border-radius:8px; padding:14px 18px;
                   font-family:monospace; border-left:4px solid #5b7fa6; font-size:0.95rem; }
  h1 { font-size:1.7rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
@st.cache_data
def load_manifest():
    if not MANIFEST.exists():
        return []
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    for r in rows:
        r["_abs"] = str(SAMPLES / r["file"])
    return [r for r in rows if Path(r["_abs"]).exists()]


@st.cache_resource(show_spinner="Loading Prithvi-EO-300M models (one-time, ~30s)...")
def get_models():
    from inference import load_models
    return load_models()


def _label(r):
    return f"{RISK_ICON[r['risk_label']]} {EVENT_LABELS.get(r['event'], r['event'])} · {r['sensor']}  (true: {r['risk_label']})"


# ─────────────────────────────────────────────
def sidebar(manifest):
    st.sidebar.title("🏔️ BhuViveka-VLM")
    st.sidebar.caption("GLOF Risk Assessment · Prithvi-EO + GRPO")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio("Input source", ["Bundled sample", "Upload .npy"], index=0)

    selected = {"source": mode}
    if mode == "Bundled sample":
        if not manifest:
            st.sidebar.error("No samples found in samples/.")
            return None
        idx = st.sidebar.selectbox("Select a patch", range(len(manifest)),
                                   format_func=lambda i: _label(manifest[i]))
        r = manifest[idx]
        selected.update(npy=r["_abs"], sensor=r["sensor"], true_label=r["risk_label"],
                        event=r["event"], patch_id=r["patch_id"])
    else:
        up = st.sidebar.file_uploader("Upload a (C,128,128) .npy patch", type=["npy"])
        sensor = st.sidebar.selectbox("Sensor", ["LISS3", "LISS4"])
        if up is None:
            st.sidebar.info("Upload a .npy array shaped (bands, H, W).")
            return None
        tmp = HERE / "_uploaded.npy"
        tmp.write_bytes(up.getbuffer())
        selected.update(npy=str(tmp), sensor=sensor, true_label=None,
                        event="uploaded", patch_id=up.name)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Physics inputs** (Stage 2)")
    selected["change_pct"] = st.sidebar.slider("Annual area change (%)", -100.0, 200.0, 10.0, 1.0)
    selected["dist_km"]   = st.sidebar.slider("Distance to known GLOF (km)", 0.0, 100.0, 5.0, 0.5)
    st.sidebar.caption("Lake area is computed automatically from the predicted mask.")
    return selected


# ─────────────────────────────────────────────
def render_result(selected, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = result["risk_class"]
    gt = selected.get("true_label")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Input patch (R-G-NIR)**")
        st.image(result["rgb_image"], use_container_width=True,
                 caption=f"Sensor: {selected['sensor']} · 128×128")
    mask = result["seg_mask"]
    with c2:
        st.markdown("**Lake segmentation (Stage 1)**")
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(mask, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, label="probability")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Pred area: {result['pred_area_km2']:.4f} km²", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c3:
        st.markdown("**Segmentation overlay**")
        rgb = result["rgb_image"].astype(float) / 255.0
        lake = np.zeros((*mask.shape, 4)); lake[mask > 0.5] = [0.1, 0.5, 1.0, 0.6]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(rgb); ax.imshow(lake)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Blue = detected lake", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")
    rc, pc = st.columns([3, 2])
    with rc:
        bc = RISK_COLOR[pred]
        if gt is not None:
            match = "✅ Correct" if pred == gt else f"❌ Incorrect (true: <b>{gt}</b>)"
        else:
            match = "Uploaded patch — no ground-truth label"
        st.markdown(f"""
<div class="risk-card" style="border-color:{bc}; background:{RISK_BG[pred]};">
  <div style="font-size:1.5rem; font-weight:700; color:{bc};">{RISK_ICON[pred]}&nbsp; {pred} RISK</div>
  <div style="font-size:1.1rem; margin-top:6px;">Confidence: <b>{result['confidence']*100:.1f}%</b></div>
  <div style="margin-top:6px; font-size:0.95rem;">{match}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("**Class probabilities**")
        for cls, prob in result["probs"].items():
            a, b = st.columns([3, 1])
            a.progress(prob, text=cls); b.write(f"{prob*100:.1f}%")

        st.markdown("**Reasoning chain**")
        st.markdown(f'<div class="reasoning-box">{result["reasoning"]}</div>', unsafe_allow_html=True)

    with pc:
        st.markdown("**Physics features used**")
        phy = result["physics"]
        st.markdown(f"""
<div class="metric-box" style="margin-bottom:10px;">
  <div style="font-size:0.8rem;color:#666;">Lake area (from segmentation)</div>
  <div style="font-size:1.4rem;font-weight:700;">{phy['area_km2']:.4f} km²</div></div>
<div class="metric-box" style="margin-bottom:10px;">
  <div style="font-size:0.8rem;color:#666;">Annual change rate</div>
  <div style="font-size:1.4rem;font-weight:700;">{phy['change_pct']:+.1f}%</div></div>
<div class="metric-box">
  <div style="font-size:0.8rem;color:#666;">Distance to known GLOF</div>
  <div style="font-size:1.4rem;font-weight:700;">{phy['dist_km']:.1f} km</div></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
def main():
    st.markdown("## 🏔️ BhuViveka-VLM — GLOF Risk Assessment POC")
    st.caption("ISRO LISS-3/4 patch → Prithvi-EO-2.0-300M lake segmentation → "
               "GRPO physics-grounded risk classification (LOW / MEDIUM / HIGH)")

    manifest = load_manifest()
    selected = sidebar(manifest)
    if selected is None:
        st.info("👈 Choose a bundled sample or upload a `.npy` patch, set the physics "
                "inputs, then press **Run Inference**.")
        return

    # quick preview before running
    if st.button("▶  Run Inference", type="primary"):
        try:
            models = get_models()
        except Exception as e:
            st.error(f"Model load failed: {e}"); st.exception(e); return
        with st.spinner("Running pipeline..."):
            from inference import predict
            try:
                result = predict(models, selected["npy"], selected["sensor"],
                                 change_pct=selected["change_pct"], dist_km=selected["dist_km"])
            except Exception as e:
                st.error(f"Inference failed: {e}"); st.exception(e); return
        render_result(selected, result)
    else:
        try:
            from inference import to_six_band, arr_to_rgb
            arr = np.nan_to_num(np.load(selected["npy"]), 0, 0, 0).astype(np.float32)
            rgb = arr_to_rgb(to_six_band(arr, selected["sensor"]))
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.markdown("#### Selected patch preview")
                st.image(rgb, use_container_width=True, caption=str(selected["patch_id"])[:48])
                if selected.get("true_label"):
                    st.caption(f"True label: {RISK_ICON[selected['true_label']]} **{selected['true_label']}**")
            st.markdown("*Press **Run Inference** to execute the model.*")
        except Exception as e:
            st.warning(f"Could not preview patch: {e}")


if __name__ == "__main__":
    main()
