# Proposal-to-engineering requirements

Source: the complete six-page `SatQuery_Final.pdf`, supplied by the team for SIH26167 (ISRO), Smart India Hackathon 2026, HACK O NOVA. Reviewed before implementation. The presentation is product context; the accompanying user request governs implementation and resource limits.

| Proposal | Engineering requirement | Acceptance / scope |
| --- | --- | --- |
| p2 upload/select, plain-language QA, visible proof, export | Streamlit, bundled synthetic fixtures, structured answers, box/mask overlays, downloadable evidence | Local demo with no model download; PNG/JPEG/TIFF uploads |
| p3 controller with optical, temporal, sensor paths | Deterministic router and replaceable backend protocols | Missing pair rejected, SAR never routed through RGB VLM |
| p3 optical QA + boxes, Qwen2.5-VL-3B | Lazy optional Qwen adapter, validated coordinates, bounded images/generation | Integration provided; real weights/inference require separate validation on suitable hardware |
| p3 before/after mask | Real numerical image-difference baseline | Aligned pairs, common valid pixels, masks/statistics; no semantic water/vegetation claims |
| p2/p4 weak input must trigger better tile/date request | Quality gates, alignment checks, transparent evidence indicators | No self-reported model confidence, no final optical answer without grounding |
| p3 metadata, sensor/date/alignment | Preserve source metadata and geospatial transforms | Missing dates stay missing; preview/window coordinates map to correct CRS |
| p3 optical + SAR check | Sensor detection and a multimodal SAR protocol | SAR preview is real; calibration/fusion/inference explicitly unsupported |
| p4 tiles, cached weights, offline demo | Bounded raster reads, window API, cached local model path, synthetic pack | No live service dependency; no weights loaded on import/startup |
| p4/p6 fixed flood, vegetation, urban, no-change validation | JSONL evaluation cases and independent metrics | Synthetic fixtures only initially, with explicit labeling; expand to 20 scenes / 100 questions |
| p3/p4 LoRA | Separate training configuration and data contract | Preflight/PEFT adapter scaffolding only; no training launched |
| p5 faster expert first pass | Traceable provenance, evidence export and reproducible runs | Benefits are goals, not measured product performance |

## Design decisions made before implementation

Keep one Python process and a Streamlit presentation layer. Pydantic schemas and runtime image/mask containers separate serializable evidence from arrays. A pipeline orchestrates loading, routing, validation and backend calls, while optical/change/SAR protocols isolate models. Hardware policy lives in one module. YAML plus validated environment overrides select local/cloud settings. Small modules replace the reference's many one-function directories; this preserves the proposed separation without infrastructure overhead.

Local optical development output is visibly synthetic, with no confidence score. A real baseline can measure appearance change, but cannot identify water increase or land-cover transitions. These semantic parts of the proposal remain experimental/planned. Georegistration and radiometry are prerequisites, not inferred from equal image dimensions. Model coordinates being valid does not establish that the model identified the correct object.

## Resource policy

No global upgrades, huge model/dataset downloads, training or benchmarks. Create an isolated `.venv` with base and test dependencies. A local 4 GiB GPU is below the conservative full-precision Qwen admission threshold; do not try fitting it by accidental CPU offload. Quantized Linux/cloud execution may be configured later with explicit optional dependencies.
