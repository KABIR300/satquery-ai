import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from rasterio.io import MemoryFile

from satquery.backends.base import BackendUnavailable, OpticalPrediction
from satquery.backends.change import BaselineChangeDetector
from satquery.evaluation import mask_metrics
from satquery.evidence import assess_quality, check_alignment, evidence_score
from satquery.export import evidence_bundle
from satquery.pipeline import Pipeline
from satquery.routing import route
from satquery.schemas import Box, Status, TaskType


def test_routing(pair):
    assert route("Locate the river", pair[:1]).task == TaskType.OPTICAL_QA
    assert route("Compare dates", pair).task == TaskType.CHANGE_DETECTION
    assert route("Show water increase", pair[:1]).status == Status.NEED_BETTER_INPUT
    assert route("Describe", pair).status == Status.UNSUPPORTED
    assert route("Describe", pair, task="change").task == TaskType.CHANGE_DETECTION
    assert route("", pair).status == Status.INVALID_INPUT
    pair[1].metadata.modality = "sar"
    assert route("Compare", pair).task == TaskType.SAR


@pytest.mark.parametrize("value", [0, 255, 120])
def test_quality_rejects_flat_inputs(value, pair):
    image = replace(pair[0], rgb=np.full_like(pair[0].rgb, value))
    quality = assess_quality(image)
    assert not quality.usable
    assert quality.score == 0


def test_evidence_is_transparent_and_mock_unscored(pair):
    quality = [assess_quality(i) for i in pair]
    score = evidence_score(pair, quality, spatial_valid=True)
    assert 0 <= score.score <= 1
    assert score.signals["semantic_correctness_verified"] is False
    assert "not a probability" in score.interpretation
    assert evidence_score(pair, quality, spatial_valid=True, demonstration=True).score is None
    assert evidence_score(pair, quality, spatial_valid=False).score < score.score


def test_alignment_shift_rejected(pair, settings):
    before = pair[0]
    after = replace(
        before, rgb=np.roll(before.rgb, 15, axis=1), analysis=np.roll(before.analysis, 15, axis=1)
    )
    decision = check_alignment(before, after, settings)
    assert not decision.accepted
    assert "shifted" in decision.warnings[0]
    assert not check_alignment(before, after, settings, user_confirmed=True).accepted


def test_alignment_grid_rejected(pair, settings):
    meta = pair[1].metadata.model_copy(update={"preview_transform": (10, 0, 500010, 0, -10, 2200000)})
    after = replace(pair[1], metadata=meta)
    assert not check_alignment(pair[0], after, settings).accepted


def test_alignment_uncertainty_requires_attestation(pair, settings, monkeypatch):
    monkeypatch.setattr("satquery.evidence.cv2.phaseCorrelate", lambda *args: ((0.1, 0.1), 0.05))
    assert not check_alignment(*pair, settings).accepted
    decision = check_alignment(*pair, settings, user_confirmed=True)
    assert decision.accepted and decision.score == 0.5


def test_reverse_date_rejected(pair, settings):
    pair[0].metadata.acquisition_date = "2026-02-01"
    assert not check_alignment(*pair, settings).accepted


def test_band_mismatch_rejected(pair, settings):
    pair[1].metadata.selected_bands = [3, 2, 1]
    assert not check_alignment(*pair, settings).accepted


@pytest.mark.parametrize("kind", ["flood", "vegetation", "urban", "no_change"])
def test_real_baseline_synthetic_mask(kind, pack, settings):
    result = Pipeline(settings).run(
        "Show differences", [pack / f"{kind}_before.tif", pack / f"{kind}_after.tif"]
    )
    assert result.result.status == Status.OK, result.result.answer
    assert result.result.backend.startswith("baseline/")
    assert result.mask.dtype == bool
    with Image.open(pack / f"{kind}_mask.png") as im:
        truth = np.asarray(im) > 0
    assert mask_metrics(result.mask, truth)["mask_iou"] == 1
    if kind == "no_change":
        assert not result.mask.any() and not result.result.bounding_boxes


def test_change_ignores_invalid_pixels(pair, settings):
    valid = pair[0].valid.copy()
    valid[:25] = False
    before = replace(pair[0], valid=valid)
    after_data = before.analysis.copy()
    after_data[:25] = 255
    after = replace(before, analysis=after_data)
    predicted = BaselineChangeDetector(settings).analyze(before, after)
    assert not predicted.mask.any()
    assert not predicted.difference[:25].any()


def test_semantic_question_gets_qualified_change_answer(pack, settings):
    output = Pipeline(settings).run(
        "Show where water increased", [pack / "flood_before.tif", pack / "flood_after.tif"]
    )
    assert output.result.status == Status.LOW_EVIDENCE
    assert "cannot identify" in output.result.answer
    assert output.mask is not None


def test_mock_truthfulness(pack, settings):
    output = Pipeline(settings).run("Where is water?", [pack / "flood_before.tif"])
    assert output.result.status == Status.LOW_EVIDENCE
    assert "MOCK / DEVELOPMENT MODE" in output.result.answer
    assert output.result.evidence.score is None
    assert output.result.provenance["inference_performed"] is False
    assert output.result.provenance["synthetic_input"]


class FakeBackend:
    name = "test/provider"

    def __init__(self, boxes=None, failure=None):
        self.boxes = boxes if boxes is not None else []
        self.failure = failure

    def analyze(self, image, question):
        if self.failure:
            raise self.failure
        return OpticalPrediction("A model claim", self.boxes, self.name)


def test_no_proof_no_answer(pair, settings):
    output = Pipeline(settings, optical_backend=FakeBackend()).analyze("Find water", pair[:1])
    assert output.result.status == Status.LOW_EVIDENCE
    assert "model claim" not in output.result.answer
    assert not output.result.bounding_boxes


def test_optical_candidate_is_not_certified(pair, settings):
    backend = FakeBackend([Box(x1=10, y1=10, x2=100, y2=100, label="Candidate")])
    output = Pipeline(settings, optical_backend=backend).analyze("Locate", pair[:1])
    assert output.result.status == Status.LOW_EVIDENCE
    assert output.result.answer == "A model claim"
    assert output.result.geospatial


@pytest.mark.parametrize(
    "failure,status",
    [
        (BackendUnavailable("missing weights"), Status.UNSUPPORTED),
        (RuntimeError("GPU failure"), Status.INVALID_OUTPUT),
    ],
)
def test_backend_failure_handled(pair, settings, failure, status):
    output = Pipeline(settings, optical_backend=FakeBackend(failure=failure)).analyze("Locate", pair[:1])
    assert output.result.status == status
    assert not output.result.bounding_boxes


def test_box_out_of_bounds_rejected(pair, settings):
    backend = FakeBackend([Box(x1=1, y1=1, x2=999, y2=30, label="invalid")])
    output = Pipeline(settings, optical_backend=backend).analyze("Locate", pair[:1])
    assert output.result.status == Status.INVALID_OUTPUT
    assert not output.result.bounding_boxes


def test_nodata_box_rejected(pair, settings):
    valid = pair[0].valid.copy()
    valid[:40, :40] = False
    image = replace(pair[0], valid=valid)
    backend = FakeBackend([Box(x1=1, y1=1, x2=30, y2=30, label="nodata")])
    assert (
        Pipeline(settings, optical_backend=backend).analyze("Locate", [image]).result.status
        == Status.INVALID_OUTPUT
    )


def test_sar_explicitly_unsupported(pack, settings):
    output = Pipeline(settings).run("Interpret", [pack / "flood_before.tif"], modality="sar")
    assert output.result.status == Status.UNSUPPORTED
    assert output.result.task_type == TaskType.SAR
    assert "not implemented" in output.result.answer
    assert output.images


@pytest.mark.parametrize(
    "question,sources,names",
    [("", [], None), ("Hello", [], None), ("Hello", [b"bad"], ["bad.png"]), ("Hello", [b"bad"], [])],
)
def test_pipeline_invalid_input(settings, question, sources, names):
    assert Pipeline(settings).run(question, sources, names=names).result.status == Status.INVALID_INPUT


def test_export_geotiff_and_geojson(pack, settings):
    output = Pipeline(settings).run("Compare", [pack / "flood_before.tif", pack / "flood_after.tif"])
    exported = evidence_bundle(output)
    with zipfile.ZipFile(io.BytesIO(exported)) as archive:
        record = json.loads(archive.read("result.json"))
        assert record["backend"].startswith("baseline/")
        assert "overlay.png" in archive.namelist() and "regions.geojson" in archive.namelist()
        with MemoryFile(archive.read("change.tif")) as memory:
            with memory.open() as ds:
                assert ds.crs.to_string() == "EPSG:32643"
                assert ds.nodata == 255
                assert np.array_equal(ds.read(1) == 1, output.mask)
                assert tuple(ds.transform)[:6] == output.images[-1].metadata.preview_transform


def test_invalid_output_mask_rejected(pair, settings):
    class BrokenChange:
        name = "broken/change"

        def analyze(self, *images):
            output = BaselineChangeDetector(settings).analyze(*images)
            output.mask = output.mask.astype(np.uint8)
            return output

    output = Pipeline(settings, change_backend=BrokenChange()).analyze("Compare", pair)
    assert output.result.status == Status.INVALID_OUTPUT
    assert output.mask is None
