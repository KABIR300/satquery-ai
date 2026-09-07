# Evaluation protocol

Keep a fixed, versioned pack of about 20 scenes and 100 questions: flood/water, vegetation, urban-change and normal/no-change groups from the proposal. Include cloud, missing dates, unknown sensors, nodata, misregistration and absent target features. Split train/validation/test by location and event; neighboring tiles are not independent scenes.

No annotated real-world pack was supplied. `data/samples/eval.jsonl` is a four-case procedural fixture manifest. Do not report its rectangle overlap as remote-sensing performance.

## JSONL manifest

One object per line. Image/mask paths resolve relative to the manifest; absolute paths are accepted for local work. Boxes/masks must be on the **analyzed preview/window grid**. No silent annotation resize occurs; transform annotations explicitly if inputs are resized.

```json
{"id":"scene01-q01","scene_id":"scene01","synthetic":false,"images":["before.tif","after.tif"],"question":"Show appearance differences","task":"change","expected_status":"OK","mask":"mask.png","expected_boxes":[[20,30,60,80]],"should_abstain":false}
```

Optional: `acceptable_answers`, `expected_status`, `expected_boxes`, `mask`, `should_abstain`, `modality`, `user_confirmed_alignment`. Missing annotations are unscored. `should_abstain=true` defines an unsupported-answer case. Do not attest alignment merely to make cases pass.

## Metrics

- Answer accuracy proxy: normalized exact text match on acceptable answers. Mock answers excluded. Independent human adjudication is still necessary for free-form QA.
- Grounding IoU: greedy one-to-one box matching, unmatched boxes contribute zero, divided by the larger box count. Labels are ignored; use class-aware matching/manual review for later benchmarks.
- Mask IoU/F1: overlap within shared valid data. Both masks empty scores 1; missing required output scores 0. Mask-output coverage is reported. Shape mismatches fail rather than rescale annotations.
- Unsupported-answer rate: answers emitted on `should_abstain=true` cases, including uncalibrated real optical candidates. Mock demonstrations are not factual claims.
- Evidence rejection rate: fraction returning LOW_EVIDENCE or NEED_BETTER_INPUT. Invalid/unsupported failures retain their own statuses.
- Latency: loading plus pipeline wall time; first Qwen inference includes cold initialization/download if enabled. Run separate cold/warm experiments; the harness does not silently repeat inference.

All summaries include `mean` and annotation denominator `n`. Review full result records and failure coverage. Engineering evidence scores are not calibrated confidence; calibration needs independently labeled correctness data.

```bash
python -m satquery evaluate data/samples/eval.jsonl --config config/local.yaml --output output/synthetic-evaluation.json
python -m satquery evaluate path/to/real-eval.jsonl --config config/cloud.yaml --output output/qwen-evaluation.json
```

Real Qwen requires the optional stack and weights on suitable hardware. Record model commit, prompt version, bands, thresholds, source hashes, sensor/date/CRS and hardware. Any scene/question count is supported without changing the pipeline.

BigEarthNet/MM, VRSBench, RSVQA and CDVQA are proposal references and possible future data sources, not bundled dependencies. Verify rights, annotations and sensor products. None were downloaded here.
