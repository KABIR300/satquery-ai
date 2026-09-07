"""Small JSONL evaluation harness; missing annotations stay unscored."""

import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from pydantic import Field

from satquery.pipeline import Pipeline
from satquery.schemas import Contract, Status


class EvaluationCase(Contract):
    id: str
    scene_id: str
    synthetic: bool = False
    images: list[str] = Field(min_length=1, max_length=2)
    question: str = Field(min_length=1)
    task: str = "auto"
    modality: str = "auto"
    expected_status: Status | None = None
    acceptable_answers: list[str] | None = None
    expected_boxes: list[tuple[float, float, float, float]] | None = None
    mask: str | None = None
    should_abstain: bool | None = None
    user_confirmed_alignment: bool = False


def load_cases(path: str | Path) -> list[EvaluationCase]:
    rows = []
    ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = EvaluationCase.model_validate_json(line)
                if case.id in ids:
                    raise ValueError("Duplicate case id.")
                ids.add(case.id)
                rows.append(case)
            except ValueError as exc:
                raise ValueError(f"Invalid evaluation row {n}: {exc}") from exc
    if not rows:
        raise ValueError("Evaluation manifest is empty.")
    return rows


def mask_metrics(predicted, truth, valid=None):
    a, b = np.asarray(predicted), np.asarray(truth)
    if a.ndim != 2 or a.shape != b.shape or a.dtype != bool or b.dtype != bool:
        raise ValueError(
            "Metrics require matching boolean masks; resize annotations explicitly, never silently."
        )
    if valid is not None:
        if valid.shape != a.shape or valid.dtype != bool or not valid.any():
            raise ValueError("Invalid evaluation footprint.")
        a, b = a[valid], b[valid]
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    total = int(a.sum()) + int(b.sum())
    return {
        "mask_iou": intersection / union if union else 1.0,
        "mask_f1": 2 * intersection / total if total else 1.0,
    }


def box_iou(a, b):
    for box in (a, b):
        if len(box) != 4 or not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Invalid metric box.")
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union)


def grounding_iou(predicted, truth):
    """Greedy one-to-one IoU matching; unmatched boxes contribute zero, labels ignored."""
    if not predicted and not truth:
        return 1.0
    pairs = sorted(
        ((box_iou(a, b), i, j) for i, a in enumerate(predicted) for j, b in enumerate(truth)), reverse=True
    )
    used_a, used_b, total = set(), set(), 0.0
    for iou, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            total += iou
    return total / max(len(predicted), len(truth))


def evaluate(manifest: str | Path, pipeline: Pipeline):
    path = Path(manifest)
    cases = load_cases(path)
    rows = []
    for case in cases:
        output = pipeline.run(
            case.question,
            [path.parent / item for item in case.images],
            task=case.task,
            modality=case.modality,
            user_confirmed_alignment=case.user_confirmed_alignment,
        )
        result = output.result
        metrics = {}
        is_mock = result.backend.startswith("mock/")
        if case.acceptable_answers is not None and not is_mock:

            def normalize(value):
                return " ".join(value.casefold().split())

            metrics["answer_exact_match"] = float(
                normalize(result.answer) in {normalize(a) for a in case.acceptable_answers}
            )
        if case.expected_status is not None:
            metrics["status_match"] = float(result.status == case.expected_status)
        if case.expected_boxes is not None and not is_mock:
            predicted = [(b.x1, b.y1, b.x2, b.y2) for b in result.bounding_boxes]
            metrics["grounding_iou"] = grounding_iou(predicted, case.expected_boxes)
        if case.mask is not None:
            with Image.open(path.parent / case.mask) as im:
                truth = np.array(im.convert("L")) > 0
            metrics["mask_output_coverage"] = float(output.mask is not None)
            if output.mask is not None:
                metrics.update(mask_metrics(output.mask, truth, output.valid_mask))
            else:
                metrics.update(mask_iou=0.0, mask_f1=0.0)
        if case.should_abstain is True:
            emits_claim = result.status == Status.OK or (
                result.status == Status.LOW_EVIDENCE
                and bool(result.bounding_boxes)
                and result.provenance.get("inference_performed") is True
            )
            metrics["unsupported_answer"] = float(emits_claim and not is_mock)
        metrics["evidence_rejected"] = float(result.status in {Status.LOW_EVIDENCE, Status.NEED_BETTER_INPUT})
        metrics["latency_seconds"] = result.latency_seconds
        rows.append(
            {
                "id": case.id,
                "scene_id": case.scene_id,
                "synthetic": case.synthetic,
                "metrics": metrics,
                "result": result.model_dump(mode="json"),
            }
        )
    summary = {}
    for key in {key for row in rows for key in row["metrics"]}:
        values = [row["metrics"][key] for row in rows if key in row["metrics"]]
        summary[key] = {"mean": mean(values), "n": len(values)}
    return {
        "scope": "Fixture validation only"
        if all(c.synthetic for c in cases)
        else "User-supplied evaluation pack",
        "case_count": len(cases),
        "scene_count": len({c.scene_id for c in cases}),
        "notes": [
            "No real-world benchmark claim. Report each metric's annotation denominator.",
            "Latency includes input loading; first real inference also includes cold model load.",
            "Empty predicted and truth masks score 1. Missing required mask output scores 0; coverage is reported.",
            "Exact text match is only a proxy for QA correctness; use independent human adjudication.",
            "Unsupported-answer rate is measured only on annotated should_abstain=true cases.",
        ],
        "summary": summary,
        "rows": rows,
    }


def save_report(report, path: str | Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
