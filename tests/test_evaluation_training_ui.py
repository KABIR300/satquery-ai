import json
import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from satquery.evaluation import box_iou, evaluate, grounding_iou, load_cases, mask_metrics
from satquery.pipeline import Pipeline
from training.lora import validate_manifest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_metrics_known_geometry():
    assert box_iou([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1 / 3)
    assert grounding_iou([[0, 0, 2, 2], [5, 5, 6, 6]], [[0, 0, 2, 2]]) == 0.5
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [True, False]])
    assert mask_metrics(a, b) == {"mask_iou": 1 / 3, "mask_f1": 0.5}
    assert mask_metrics(a & False, b & False) == {"mask_iou": 1, "mask_f1": 1}
    with pytest.raises(ValueError):
        mask_metrics(a, np.zeros((5, 5), bool))


def test_fixture_eval_has_real_denominators(pack, settings):
    report = evaluate(pack / "eval.jsonl", Pipeline(settings))
    assert report["case_count"] == 4
    assert report["scope"] == "Fixture validation only"
    assert report["summary"]["mask_iou"] == {"mean": 1, "n": 4}
    assert "answer_exact_match" not in report["summary"]
    assert "unsupported_answer" not in report["summary"]


def test_eval_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "eval.jsonl"
    row = {"id": "same", "scene_id": "scene", "images": ["image.png"], "question": "Where?"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(path)


def test_mock_answers_not_scored_as_accuracy(pack, settings, tmp_path):
    path = tmp_path / "eval.jsonl"
    row = {
        "id": "optical",
        "scene_id": "s",
        "images": [str(pack / "flood_before.tif")],
        "question": "Where is water?",
        "acceptable_answers": ["water"],
        "synthetic": True,
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    report = evaluate(path, Pipeline(settings))
    assert "answer_exact_match" not in report["summary"]


def test_training_preflight_detects_scene_leakage(tmp_path, textured_png):
    path = tmp_path / "training.jsonl"
    row = {
        "image": str(textured_png),
        "question": "Where?",
        "answer": "Here",
        "scene_id": "s",
        "split": "train",
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    assert validate_manifest(path)["status"].startswith("PREFLIGHT ONLY")
    path.write_text(json.dumps(row) + "\n" + json.dumps(row | {"split": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(path)


def test_streamlit_start_and_mock_run():
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert "torch" not in sys.modules
    app.button[0].click().run()
    assert not app.exception
    assert any("MOCK / DEVELOPMENT MODE" in item.value for item in app.warning)
    assert app.session_state["output"].result.backend.startswith("mock/")


def test_streamlit_change_run():
    app = AppTest.from_file(APP_PATH).run()
    workflow = next(radio for radio in app.radio if radio.label == "Workflow")
    workflow.set_value("Before / after").run()
    app.button[0].click().run()
    assert not app.exception
    output = app.session_state["output"]
    assert output.mask is not None
    assert output.result.status.value == "OK"


def test_streamlit_bad_config_reports_error(monkeypatch, tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("change_threshold: nonsense", encoding="utf-8")
    monkeypatch.setenv("SATQUERY_CONFIG", str(path))
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert any("Configuration error" in item.value for item in app.error)
