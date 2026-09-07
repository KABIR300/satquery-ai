"""Centralized hardware inspection and conservative model admission policy."""

import importlib.util
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass

import psutil

from satquery.config import Settings


class HardwareError(RuntimeError):
    pass


@dataclass
class Hardware:
    os: str
    python: str
    executable: str
    cpu: str
    ram_total_gib: float
    ram_available_gib: float
    nvidia: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available: bool = False
    gpu_name: str | None = None
    vram_total_gib: float | None = None
    vram_free_gib: float | None = None
    bf16_supported: bool = False
    torch_error: str | None = None

    def as_dict(self):
        return asdict(self)


def inspect_hardware(include_torch: bool = True) -> Hardware:
    memory = psutil.virtual_memory()
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    report = Hardware(
        platform.platform(),
        platform.python_version(),
        sys.executable,
        cpu,
        memory.total / 1024**3,
        memory.available / 1024**3,
    )
    try:
        info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        report.nvidia = info.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    if include_torch and importlib.util.find_spec("torch"):
        try:
            import torch

            report.torch_version = torch.__version__
            report.torch_cuda_version = torch.version.cuda
            report.cuda_available = torch.cuda.is_available()
            if report.cuda_available:
                report.gpu_name = torch.cuda.get_device_name(0)
                report.vram_total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free, _ = torch.cuda.mem_get_info(0)
                report.vram_free_gib = free / 1024**3
                report.bf16_supported = torch.cuda.is_bf16_supported()
        except Exception as exc:
            report.torch_error = f"{type(exc).__name__}: {exc}"
    return report


def select_runtime(settings: Settings, hardware: Hardware) -> tuple[str, str]:
    if hardware.torch_version is None or hardware.torch_error:
        raise HardwareError(
            "Working PyTorch is unavailable in this environment. Install the optional ML stack."
        )
    device = settings.device
    if device == "auto":
        device = "cuda" if hardware.cuda_available else "cpu"
    if device == "cuda" and not hardware.cuda_available:
        raise HardwareError("CUDA was requested but this PyTorch installation cannot use it.")
    dtype = settings.dtype
    if dtype == "auto":
        dtype = "float32" if device == "cpu" else ("bfloat16" if hardware.bf16_supported else "float16")
    if device == "cpu" and dtype != "float32":
        raise HardwareError("CPU execution requires float32 for the portable implementation.")
    if dtype == "bfloat16" and device == "cuda" and not hardware.bf16_supported:
        raise HardwareError("The selected GPU does not support bfloat16; use float16.")
    if settings.quantization == "4bit" and (sys.platform != "linux" or device != "cuda"):
        raise HardwareError("The experimental bitsandbytes path is supported here only on Linux CUDA.")
    # Admission estimates, not measurements or guarantees; no implicit CPU/disk offload.
    if device == "cuda":
        required_vram = 6 if settings.quantization == "4bit" else (18 if dtype == "float32" else 10)
        if (hardware.vram_free_gib or 0) < required_vram:
            raise HardwareError(
                f"Qwen admission requires at least {required_vram} GiB free VRAM for this "
                f"configuration; detected {hardware.vram_free_gib or 0:.2f}. "
                "Use mock mode locally or a larger GPU."
            )
        if hardware.ram_available_gib < 6:
            raise HardwareError("At least 6 GiB available system RAM is required for GPU model loading.")
    elif hardware.ram_available_gib < 20:
        raise HardwareError(
            "CPU float32 Qwen requires at least 20 GiB available RAM by this admission policy."
        )
    return device, dtype
