import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from satquery.backends.base import BackendUnavailable
from satquery.backends.qwen import QwenBackend
from satquery.config import Settings
from satquery.hardware import Hardware, HardwareError, select_runtime
from satquery.pipeline import Pipeline
from satquery.schemas import Status


def hardware(**updates):
    fields = dict(
        os="test",
        python="3.11",
        executable="python",
        cpu="test",
        ram_total_gib=32,
        ram_available_gib=25,
        torch_version="2.7.1",
        torch_cuda_version="11.8",
        cuda_available=True,
        gpu_name="test GPU",
        vram_total_gib=24,
        vram_free_gib=20,
        bf16_supported=True,
    )
    return Hardware(**(fields | updates))


@pytest.mark.parametrize(
    "updates",
    [
        dict(vram_free_gib=3.9, vram_total_gib=4),
        dict(cuda_available=False),
        dict(torch_version=None),
        dict(ram_available_gib=4),
    ],
)
def test_model_admission_rejects_unsafe_hardware(updates):
    with pytest.raises(HardwareError):
        select_runtime(Settings(device="cuda"), hardware(**updates))


def test_cpu_admission_and_dtype():
    assert select_runtime(Settings(device="cpu"), hardware()) == ("cpu", "float32")
    with pytest.raises(HardwareError):
        select_runtime(Settings(device="cpu"), hardware(ram_available_gib=15))
    assert select_runtime(Settings(), hardware()) == ("cuda", "bfloat16")
    assert select_runtime(Settings(), hardware(bf16_supported=False)) == ("cuda", "float16")


def test_bf16_rejected_without_hardware_support():
    with pytest.raises(HardwareError, match="bfloat16"):
        select_runtime(Settings(dtype="bfloat16"), hardware(bf16_supported=False))


def test_quantization_platform_policy(monkeypatch):
    monkeypatch.setattr("satquery.hardware.sys.platform", "win32")
    with pytest.raises(HardwareError, match="Linux"):
        select_runtime(Settings(quantization="4bit"), hardware())
    monkeypatch.setattr("satquery.hardware.sys.platform", "linux")
    assert select_runtime(Settings(quantization="4bit"), hardware(vram_free_gib=7))[0] == "cuda"


def test_imports_do_not_load_torch_or_transformers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import satquery.pipeline; from satquery.config import Settings; "
            "p=satquery.pipeline.Pipeline(Settings(backend='qwen')); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_qwen_hardware_checked_before_ml_import(monkeypatch):
    backend = QwenBackend(Settings(backend="qwen"))
    monkeypatch.setattr("satquery.backends.qwen.inspect_hardware", lambda: hardware(vram_free_gib=4))
    with pytest.raises(BackendUnavailable, match="free VRAM"):
        backend._load()
    assert backend._model is None


def test_missing_ml_becomes_explicit_status(pair, settings, monkeypatch):
    monkeypatch.setattr("satquery.backends.qwen.inspect_hardware", lambda: hardware(torch_version=None))
    pipeline = Pipeline(Settings(backend="qwen"))
    output = pipeline.analyze("Locate a visible feature", pair[:1])
    assert output.result.status == Status.UNSUPPORTED
    assert "PyTorch" in output.result.answer


def test_qwen_generation_contract_without_model_download(pair, monkeypatch):
    seen = {}

    class Batch(dict):
        def to(self, device):
            seen["device"] = device
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            return "chat-template"

        def __call__(self, **kwargs):
            seen["processor"] = kwargs
            return Batch(input_ids=np.array([[1, 2, 3]]))

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [[4, 5]]  # Prompt tokens must not appear in the parsed answer.
            return [
                json.dumps(
                    {
                        "answer": "Test proposal",
                        "coordinate_space": "normalized",
                        "boxes": [{"label": "candidate", "bbox": [0.1, 0.1, 0.5, 0.5]}],
                    }
                )
            ]

    class Model:
        config = SimpleNamespace(_commit_hash="fixture-revision")

        def generate(self, **kwargs):
            seen["generation"] = kwargs
            return np.array([[1, 2, 3, 4, 5]])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    backend = QwenBackend(Settings(backend="qwen"))
    backend._model, backend._processor = Model(), Processor()
    backend._device, backend._dtype = "cpu", "float32"
    prediction = backend.analyze(pair[0], "Find a region")
    assert prediction.boxes[0].x2 == 128
    assert prediction.provenance["resolved_revision"] == "fixture-revision"
    assert seen["generation"]["do_sample"] is False
    assert "temperature" not in seen["generation"]
    assert seen["processor"]["images"][0].size == (256, 256)
