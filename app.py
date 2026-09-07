"""Hackathon demo. Starting Streamlit never imports torch or downloads model weights."""

import hashlib
import os
from pathlib import Path

import numpy as np
import streamlit as st

from satquery.config import Settings, load_settings
from satquery.export import evidence_bundle
from satquery.grounding import validate_boxes, validate_mask
from satquery.pipeline import Pipeline
from satquery.schemas import Result, Status
from satquery.visualization import overlay

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")


@st.cache_resource(show_spinner=False, max_entries=1)
def get_pipeline(config_json: str):
    return Pipeline(Settings.model_validate_json(config_json))


def render_result(output):
    try:
        # Streamlit can retain objects from a previous schema class after source reload.
        # Revalidate our structured record before rendering; never parse model prose here.
        result = Result.model_validate(output.result.model_dump(mode="json", warnings=False))
        output.result = result
        if output.images:
            for image in output.images:
                if image.rgb.ndim != 3 or image.rgb.shape[2] != 3 or image.rgb.dtype != np.uint8:
                    raise ValueError("Invalid stored source preview.")
            shape = output.images[-1].rgb.shape[:2]
            validate_boxes(result.bounding_boxes, shape[1], shape[0])
            if output.mask is not None:
                validate_mask(output.mask, shape, output.valid_mask)
                if (
                    output.valid_mask is None
                    or output.difference is None
                    or output.difference.shape != shape
                    or not np.isfinite(output.difference).all()
                ):
                    raise ValueError("Invalid stored mask/difference.")
        bundle = evidence_bundle(output)
    except (ValueError, TypeError, AttributeError, OSError):
        st.error("Stored spatial evidence is invalid or stale. Run analysis again to regenerate it.")
        return
    st.divider()
    st.subheader("Answer & evidence")
    a, b, c = st.columns(3)
    a.metric("Task", result.task_type.value.replace("_", " "))
    b.metric("Status", result.status.value.replace("_", " "))
    c.metric(
        "Evidence checks",
        "N/A · mock / unavailable" if result.evidence.score is None else f"{result.evidence.score:.2f} / 1",
    )
    st.caption(result.evidence.interpretation)
    if result.status in {Status.INVALID_INPUT, Status.INVALID_OUTPUT, Status.NEED_BETTER_INPUT}:
        st.error(result.answer)
    elif result.status in {Status.LOW_EVIDENCE, Status.UNSUPPORTED}:
        st.warning(result.answer)
    else:
        st.success(result.answer)
    if output.images:
        columns = st.columns(len(output.images) + 1)
        for i, image in enumerate(output.images):
            columns[i].image(
                image.rgb,
                caption=("Source" if len(output.images) == 1 else ("Before" if i == 0 else "After")),
                width="stretch",
            )
        columns[-1].image(
            overlay(output.images[-1].rgb, result.bounding_boxes, output.mask),
            caption="Spatial evidence · preview coordinates",
            width="stretch",
        )
    if output.mask is not None:
        with st.expander("Change mask and numerical difference", expanded=True):
            x, y = st.columns(2)
            x.image(
                output.mask.astype(np.uint8) * 255, caption="Binary appearance-change mask", width="stretch"
            )
            y.image(
                output.difference,
                clamp=True,
                caption="Normalized absolute difference · not a probability",
                width="stretch",
            )
            st.json(result.statistics)
    for note in dict.fromkeys(result.warnings):
        st.caption(f"⚠ {note}")
    st.caption(f"Backend: {result.backend} · Elapsed: {result.latency_seconds:.3f} s")
    with st.expander("Metadata, coordinates & provenance"):
        st.json(result.model_dump(mode="json"))
    st.download_button(
        "Export evidence bundle",
        bundle,
        file_name="satquery-evidence.zip",
        mime="application/zip",
        key="export_bundle",
    )
    st.download_button(
        "Download result JSON",
        result.model_dump_json(indent=2),
        file_name="satquery-result.json",
        mime="application/json",
        key="export_json",
    )


def main():
    st.caption("SIH 2026 / SIH26167 / HACK O NOVA")
    st.title("SatQuery AI")
    st.markdown("Ask about a satellite tile. Inspect the evidence. Keep the source in view.")
    try:
        defaults = load_settings(os.environ.get("SATQUERY_CONFIG") or ROOT / "config" / "local.yaml")
    except (ValueError, OSError) as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    with st.sidebar:
        st.subheader("Analysis setup")
        profile = st.selectbox(
            "Compute profile", ["local", "cloud"], index=0 if defaults.profile == "local" else 1
        )
        if profile != defaults.profile:
            try:
                defaults = load_settings(ROOT / "config" / f"{profile}.yaml")
            except (ValueError, OSError) as exc:
                st.error(f"Profile configuration error: {exc}")
                st.stop()
        chosen = st.selectbox(
            "Optical backend",
            ["MOCK / DEVELOPMENT MODE", "Qwen · real inference"],
            index=0 if defaults.backend == "mock" else 1,
        )
        backend = "mock" if chosen.startswith("MOCK") else "qwen"
        workflow = st.radio("Workflow", ["Single image", "Before / after"])
        source_mode = st.radio("Image source", ["Bundled synthetic sample", "Upload imagery"])
        modality_label = st.selectbox("Sensor type", ["Auto / metadata", "Optical", "SAR · experimental"])
        modality = {"Auto / metadata": "auto", "Optical": "optical", "SAR · experimental": "sar"}[
            modality_label
        ]
        with st.expander("Runtime controls"):
            device = st.selectbox(
                "Device", ["auto", "cpu", "cuda"], index=["auto", "cpu", "cuda"].index(defaults.device)
            )
            allow_download = st.checkbox(
                "Allow model download on real inference", value=defaults.allow_model_download
            )
            st.caption("Qwen weights require several GB. Memory admission runs before any download.")
            threshold = st.slider(
                "Appearance-change threshold", 0.01, 0.95, float(defaults.change_threshold), 0.01
            )
        st.caption("Local demo: no VLM needed. SAR interpretation and learned change detection are planned.")
    if backend == "mock":
        st.info(
            "MOCK / DEVELOPMENT MODE — optical boxes are synthetic demonstrations. "
            "Before/after analysis uses a real numerical baseline."
        )
    else:
        st.info(
            "Qwen is loaded only after Run analysis. Its answers and proposed regions require expert review."
        )
    is_pair = workflow == "Before / after"
    sources, names = [], []
    if source_mode == "Bundled synthetic sample":
        st.warning(
            "SYNTHETIC SAMPLE — generated test pixels, fictional dates and coordinates; not satellite observations."
        )
        scene = st.selectbox("Sample scene", ["flood", "vegetation", "urban", "no_change"])
        phases = ["before", "after"] if is_pair else ["before"]
        sources = [ROOT / "data" / "samples" / f"{scene}_{phase}.tif" for phase in phases]
        if not all(path.is_file() for path in sources):
            st.error("Sample pack missing. Run: python -m satquery samples")
            sources = []
    else:
        columns = st.columns(2 if is_pair else 1)
        for idx in range(2 if is_pair else 1):
            label = ("Before image" if idx == 0 else "After image") if is_pair else "Satellite image"
            uploaded = columns[idx].file_uploader(
                label, type=["png", "jpg", "jpeg", "tif", "tiff"], key=f"file_{idx}"
            )
            if uploaded is not None:
                sources.append(uploaded.getvalue())
                names.append(uploaded.name)
    prompt = (
        "Show appearance differences between these images." if is_pair else "Where is the main water body?"
    )
    with st.form("analysis"):
        question = st.text_area(
            "Your question", value=prompt, max_chars=defaults.max_question_chars, key=f"question_{is_pair}"
        )
        if is_pair:
            st.caption(
                "The baseline marks appearance changes. Water increase and land-cover transitions require a semantic model."
            )
        confirmed = (
            st.checkbox("I independently checked that this pair is co-registered.", value=False)
            if is_pair
            else False
        )
        run = st.form_submit_button("Run analysis", type="primary", width="stretch")
    # Config changes invalidate the visible result, avoiding stale answers with different controls.
    signature = (
        profile,
        backend,
        workflow,
        source_mode,
        modality,
        device,
        allow_download,
        threshold,
        tuple(str(p) if isinstance(p, Path) else hashlib.sha256(p).hexdigest() for p in sources),
    )
    if st.session_state.get("result_signature") != signature:
        st.session_state.pop("output", None)
    if run:
        if len(sources) != (2 if is_pair else 1):
            st.error("Select all required images before running analysis.")
        else:
            try:
                settings = Settings.model_validate(
                    defaults.model_dump()
                    | {
                        "profile": profile,
                        "backend": backend,
                        "device": device,
                        "allow_model_download": allow_download,
                        "change_threshold": threshold,
                    }
                )
                pipeline = get_pipeline(settings.model_dump_json())
                with st.spinner("Checking inputs and analyzing evidence…"):
                    output = pipeline.run(
                        question,
                        sources,
                        names=names or None,
                        modality=modality,
                        task="change" if is_pair else "auto",
                        user_confirmed_alignment=confirmed,
                    )
                st.session_state.output = output
                st.session_state.result_signature = signature
            except (ValueError, OSError) as exc:
                st.error(f"Configuration or input error: {exc}")
    if "output" in st.session_state:
        render_result(st.session_state.output)
    else:
        st.caption(
            "Examples: “Where is the main water body?” · “Which area appears built-up?” · "
            "“What major land-cover change is visible?”"
        )


if __name__ == "__main__":
    main()
