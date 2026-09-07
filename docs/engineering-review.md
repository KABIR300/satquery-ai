# Final engineering review — 2026-09-07

The repository was initially empty except for `.git`. The complete six-page SIH proposal was extracted and visually reviewed before architecture/implementation. Requirements were recorded first in `docs/requirements.md`. No existing application files were overwritten and no global environment was upgraded.

## Detected environment

| Item | Observed |
| --- | --- |
| OS | Windows 11 Home Single Language, build 10.0.26200 (Python's platform string reports Windows-10 for this build) |
| Shell | PowerShell 7.6.5 |
| CPU | AMD Ryzen 7 5800H with Radeon Graphics; differs from the estimated Ryzen 7 2700 in the request |
| Memory | 15.345 GiB OS-visible; approximately 4.3 GiB available during inspection, varies with running apps |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | NVIDIA-SMI: 4096 MiB total, 3954 MiB free at initial inspection; torch device property: 4,294,443,008 bytes total |
| NVIDIA driver | 592.82 |
| Driver CUDA compatibility | NVIDIA-SMI advertises CUDA 13.1; this does not establish a toolkit installation |
| Global Python / pip | Python 3.11.3, pip 25.3 |
| Existing global torch | 2.7.1+cu118; compiled CUDA 11.8; torch.cuda.is_available() True; one CUDA device |
| Project environment | Isolated `.venv`, Python 3.11.3, initial venv pip 22.3.1; torch intentionally not installed |
| Git | 2.51.2.windows.1 |

Windows CIM was initially blocked by the sandbox. Its read-only retry was approved. Dependency-network access likewise required approval; packages were installed only into `.venv`. No CUDA, NVIDIA driver, Python, global torch or unrelated global packages were changed. No weights, datasets, training or large benchmarks were downloaded/run.

## Dependencies

Installed direct base/test packages: NumPy 2.4.6, Pillow 12.3.0, opencv-python-headless 4.14.0.94, Rasterio 1.4.4, Affine 2.4.0, Streamlit 1.63.0, Pydantic 2.13.5, PyYAML 6.0.3, psutil 7.2.2, pytest 9.1.1 and Ruff 0.16.6. `constraints/local-tested.txt` records these. Affine was constrained below 3 because Rasterio 1.4 uses its earlier multiplication API; this also removed deprecation warnings in the final tests.

Optional ML/training/quantization dependencies are separate. Transformers is pinned to 4.57.1 for the implemented Qwen API, with compatible torch/torchvision ranges. No heavy optional package was installed here. All model loading occurs on real inference requests after centralized admission checks. Streamlit caches at most one pipeline configuration to avoid retaining multiple model instances as controls change.

## Architecture and genuine capability

One Python package holds shared contracts, loading, routing, evidence validation and orchestration. Provider protocols isolate optical/change/SAR implementations. The CLI and Streamlit consume the same pipeline; exporting does not depend on UI state or model prose. This retains the proposal/reference separation and avoids excessive one-function folders or infrastructure.

Local paths are complete: uploads and fixtures, deterministic routing, visibly mock optical outputs, real numerical change baseline, quality/alignment gates, masks/boxes, geospatial transforms, metadata/provenance and downloadable evidence. Temporal semantic questions produce a qualified baseline result rather than a fabricated flood/land-cover answer.

Qwen integration is implemented and its token/processor/output contracts are tested with fake inference. Real weights were not loaded or tested. The 4 GiB GPU is below the conservative 10 GiB free FP16/BF16 admission gate. The cloud route is configuration-driven; installing its optional stack, downloading/caching weights and validating real inference remain necessary.

## Verification

The final suite contains **119 passing tests** covering:

- Config/env parsing, invalid YAML, schema/finite-value constraints and mask-count consistency.
- Image decode, upload limits, alpha/nodata, missing bands, sensor inference, corrupted and empty rasters.
- Pixel/normalized/0–1000 boxes, malformed outputs, nodata grounding rejection and boolean mask validation.
- CRS reprojection, rotated affines, pixel centers/edges, downsampling, window offsets and tiling.
- Deterministic routing, quality checks, detected translations, uncertain registration, reversed dates and band mismatch.
- Real fixture change/no-change masks; provider errors; explicit SAR/learned-model unsupported behavior.
- Memory/dtype/quantization policy; no torch/Transformers on import; fake Qwen prompt trimming and generation contract.
- ZIP, GeoJSON and GeoTIFF mask round-trip; UTF-8 CLI output on Windows; evaluation denominators, rejection/failure handling and split leakage.
- Streamlit startup, mock run, temporal run, backend selection, bad config, stale-result invalidation, old schema instances after reload, and invalid stored bounding boxes.

Also verified: Ruff lint and formatting, compileall, pip dependency consistency, CLI evaluation and window analysis, actual Streamlit HTTP startup and browser rendering. Initial UI tests exposed a Streamlit relative-path resolution change; tests now use an absolute path derived from their location. Malformed YAML and unexpected SAR-provider failures were hardened during review. Missing expected evaluation masks now score zero with explicit coverage, instead of disappearing from the denominator. The final live refresh exposed old schema instances retained by Streamlit after code reload; the renderer now revalidates structured records and guards invalid spatial artifacts before rendering/export. Regression tests pass, and both live reload and a fresh browser analysis succeeded after the fix.

Four procedural scenes produced exact expected masks (IoU/F1 1 on these synthetic rectangles, n=4) and expected statuses. This is fixture correctness, **not real remote-sensing accuracy**. The generated evaluation JSON includes measured per-run latency; no answer accuracy or unsupported-answer metric was invented when annotations were absent. A live browser run of the synthetic flood pair marked 4,060 of 65,536 preview pixels (6.2%); the case intentionally changes that rectangle.

The live Streamlit process group showed approximately 125 MiB aggregate resident working set at one observation after a tiny fixture run. This is a Windows working-set snapshot, not peak memory or a general resource benchmark. Plan 1–2 GiB RAM headroom for ordinary bounded previews and more for near-limit compressed input decoding. The baseline/mock path requires no CUDA.

## Honest remaining gaps

Experimental: real Qwen compatibility/quality on actual weights, 4-bit Linux inference and PEFT attachment. Planned/unimplemented: trained neural change detector, semantic water/vegetation/built-up change, SAR calibration/speckle filtering/understanding/fusion, full LoRA trainer and inference adapter loading, cloud masking, robust auto-registration, calibrated correctness probability, full-scene mosaicking and batch inference.

Alignment screening only measures translations and grid compatibility. Quality checks do not detect clouds reliably. Same-sensor images can still have incompatible processing/radiometry. GeoTIFF RGB bands may need explicit configuration. Masks/boxes refer to bounded previews, so small objects can disappear on downsampling. SAR metadata detection covers common tags, not all product formats. The UI is a single-user local demo; no public hosting or multi-tenant service was created. Cloud/macOS/Linux commands are documented but were not executed here.

## Next three improvements

1. Validate real pretrained Qwen and grounding against a licensed, independently adjudicated 20-scene/100-question pack on suitable GPU hardware.
2. Add robust registration and sensor-aware cloud/radiometric checks, followed by measured semantic change methods.
3. Build tested multimodal LoRA collation/training and calibration on held-out scenes; advance SAR through calibrated preprocessing and measured fusion afterward.

Exact Windows, Git Bash and Linux setup, Streamlit/test commands, real-model enablement, cache policy, hardware estimates and the final tree are in `README.md`. This review makes no claim that the complete proposed research system has been delivered.
