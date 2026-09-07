import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import create_model
from streamlit.testing.v1 import AppTest

from satquery.config import load_settings
from satquery.evaluation import evaluate
from satquery.pipeline import Pipeline
from satquery.schemas import Status


def test_malformed_yaml_is_configuration_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("backend: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed YAML"):
        load_settings(path)


def test_malformed_env_is_configuration_error(monkeypatch):
    monkeypatch.setenv("SATQUERY_RGB_BANDS", "[")
    with pytest.raises(ValueError, match="environment variable"):
        load_settings()


def test_missing_mask_counts_as_failure(pack, tmp_path, settings):
    manifest = tmp_path / "eval.jsonl"
    row = {
        "id": "missing",
        "scene_id": "s",
        "images": ["missing.tif", "missing-too.tif"],
        "question": "Compare",
        "mask": str(pack / "flood_mask.png"),
    }
    manifest.write_text(json.dumps(row), encoding="utf-8")
    report = evaluate(manifest, Pipeline(settings))
    assert report["summary"]["mask_iou"] == {"mean": 0.0, "n": 1}
    assert report["summary"]["mask_output_coverage"] == {"mean": 0.0, "n": 1}


def test_sar_provider_failure_handled(pair, settings):
    class BrokenSAR:
        name = "broken/sar"

        def analyze(self, images, question):
            raise RuntimeError("provider failed")

    pair[0].metadata.modality = "sar"
    result = Pipeline(settings, sar_backend=BrokenSAR()).analyze("Interpret", pair[:1]).result
    assert result.status == Status.INVALID_OUTPUT


def test_streamlit_qwen_switch_does_not_load_model():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    backend = next(s for s in app.selectbox if s.label == "Optical backend")
    backend.set_value("Qwen · real inference").run()
    assert not app.exception
    assert "output" not in app.session_state


def test_streamlit_rerun_drops_stale_result():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.button[0].click().run()
    assert "output" in app.session_state
    next(s for s in app.selectbox if s.label == "Sample scene").set_value("urban").run()
    assert "output" not in app.session_state
    assert not app.exception


def test_cli_json_is_utf8(pack, tmp_path):
    question = "Locate water — पानी"
    command = [
        sys.executable,
        "-m",
        "satquery",
        "analyze",
        str(pack / "flood_before.tif"),
        "--question",
        question,
        "--output",
        str(tmp_path / "evidence.zip"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["provenance"]["question"] == question


def test_streamlit_revalidates_previous_schema_objects():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.button[0].click().run()
    result = app.session_state["output"].result
    fields = result.bounding_boxes[0].model_dump()
    previous_box_type = create_model("PreviousBox", **{k: (type(v), ...) for k, v in fields.items()})
    result.bounding_boxes = [previous_box_type(**fields)]
    app.run()
    assert not app.exception
    assert not app.error


def test_streamlit_invalid_stored_box_does_not_crash():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.button[0].click().run()
    app.session_state["output"].result.bounding_boxes[0].x2 = 999999
    app.run()
    assert not app.exception
    assert any("invalid or stale" in item.value for item in app.error)
