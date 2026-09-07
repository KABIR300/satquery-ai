import math

import pytest
from pydantic import ValidationError

from satquery.config import Settings, load_settings
from satquery.schemas import Box, Evidence, MaskInfo, Result, Status, TaskType


def test_config_files():
    assert load_settings("config/local.yaml").backend == "mock"
    assert load_settings("config/cloud.yaml").backend == "qwen"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SATQUERY_BACKEND", "qwen")
    monkeypatch.setenv("SATQUERY_ALLOW_MODEL_DOWNLOAD", "false")
    config = load_settings("config/local.yaml")
    assert config.backend == "qwen" and not config.allow_model_download
    assert load_settings("config/local.yaml", backend="mock").backend == "mock"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(change_threshold=0),
        dict(evidence_threshold=1.1),
        dict(max_image_size=10000),
        dict(dtype="float16", device="cpu"),
        dict(quantization="4bit", device="cpu"),
        dict(rgb_bands=[0, 1, 2]),
        dict(unknown=3),
        dict(change_threshold=float("nan")),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)


def test_bad_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("[a, b]")
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.mark.parametrize("value", [-0.1, 1.01, math.nan, math.inf])
def test_invalid_score(value):
    with pytest.raises(ValidationError):
        Evidence(score=value)


def test_result_roundtrip():
    result = Result(
        answer="Test",
        task_type=TaskType.OPTICAL_QA,
        status=Status.LOW_EVIDENCE,
        bounding_boxes=[Box(x1=0, y1=0, x2=10, y2=10, label="Fixture")],
    )
    assert Result.model_validate_json(result.model_dump_json()) == result
    assert result.evidence.score is None


def test_no_shared_mutable_defaults():
    a = Result(answer="a", task_type=TaskType.SAR, status=Status.UNSUPPORTED)
    b = Result(answer="b", task_type=TaskType.SAR, status=Status.UNSUPPORTED)
    a.warnings.append("a")
    assert not b.warnings


@pytest.mark.parametrize("positive,valid", [(12, 10), (2, 401)])
def test_mask_counts_are_consistent(positive, valid):
    with pytest.raises(ValidationError):
        MaskInfo(name="mask", width=20, height=20, positive_pixels=positive, valid_pixels=valid)
