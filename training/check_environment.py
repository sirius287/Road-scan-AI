"""Phase 1.1: Windows/CUDA environment diagnostics, not model evaluation.

Run from the project root with .venv\\Scripts\\python.exe.
No datasets, pretrained weights, training, or training-batch selection.
Exit 0 means the required checks passed; exit 1 means the phase is blocked.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = (3, 13, 15)
CUDA_VERSION = "12.8"
# Distribution name, import name, exact distribution version.
PACKAGES = (
    ("torch", "torch", "2.11.0+cu128"),
    ("torchvision", "torchvision", "0.26.0+cu128"),
    ("numpy", "numpy", "2.2.6"),
    ("opencv-python", "cv2", "4.12.0.88"),
    ("pandas", "pandas", "2.3.3"),
    ("PyYAML", "yaml", "6.0.3"),
    ("pytest", "pytest", "8.4.2"),
    ("psutil", "psutil", "7.2.2"),
    ("ultralytics", "ultralytics", "8.4.154"),
)
GIB = 1024 ** 3


def require(condition: bool, message: str) -> None:
    """Do not use assert for production checks: python -O disables assertions."""
    if not condition:
        raise RuntimeError(message)


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.facts: dict[str, Any] = {}

    def add(self, status: str, name: str, detail: Any) -> None:
        self.checks.append({"status": status, "name": name, "detail": detail})
        text = detail if isinstance(detail, str) else json.dumps(detail)
        print(f"[{status}] {name}: {text}", flush=True)

    def run(self, name: str, action: Callable[[], Any]) -> bool:
        try:
            self.add("PASS", name, action())
            return True
        except Exception as exc:
            self.add("FAIL", name, f"{type(exc).__name__}: {exc}")
            self.checks[-1]["traceback"] = traceback.format_exc()
            return False

    @property
    def failed(self) -> bool:
        return any(item["status"] == "FAIL" for item in self.checks)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_leaves(value: Any, torch: Any):
    """Handle tensor, tuple/list, and dictionary model outputs."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_leaves(child, torch)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_leaves(child, torch)


def check_tensors(value: Any, torch: Any, device: Any) -> list[list[int]]:
    tensors = list(tensor_leaves(value, torch))
    require(bool(tensors), "The model did not return any tensors.")
    for tensor in tensors:
        require(tensor.numel() > 0, "An output tensor is empty.")
        require(tensor.device == device, f"Unexpected output device: {tensor.device}")
        require(bool(torch.isfinite(tensor).all().item()), "Output contains NaN/Inf.")
    return [list(tensor.shape) for tensor in tensors]


def tensor_and_conv_test(torch: Any, device: Any) -> dict[str, Any]:
    """Exercise actual matrix multiplication and neural-network inference."""
    torch.manual_seed(42)
    with torch.inference_mode():
        matrix = torch.randn(64, 64, device=device)
        identity = torch.eye(64, device=device)
        require(torch.allclose(matrix @ identity, matrix, atol=1e-4, rtol=1e-4),
                "Matrix multiplication sanity check failed.")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2),
        ).to(device).eval()
        image = torch.rand(1, 3, 64, 64, device=device)
        output = model(image)
        sync(torch, device)
        shapes = check_tensors(output, torch, device)
        require(shapes == [[1, 2]], f"Unexpected CNN output shape: {shapes}")
    return {"device": str(device), "output_shape": shapes[0], "finite": True}


def nms_test(torch: Any, torchvision: Any, device: Any) -> dict[str, Any]:
    """Detect mismatched torch/torchvision binaries and missing CUDA NMS kernels."""
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
                         dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device=device)
    with torch.inference_mode():
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        sync(torch, device)
        require(keep.device == device, "NMS did not return indices on the selected device.")
        require(keep.tolist() == [0, 2], f"Unexpected NMS output: {keep.tolist()}")
    return {"device": str(device), "kept_indices": keep.tolist()}


def yolo_test(torch: Any, ultralytics: Any, yaml: Any, device: Any,
              image_size: int) -> dict[str, Any]:
    """Untrained, two-class YOLO11n forward pass using only the installed YAML."""
    from ultralytics.nn.tasks import DetectionModel

    config_path = Path(ultralytics.__file__).resolve().parent / "cfg/models/11/yolo11.yaml"
    require(config_path.is_file(), f"Bundled YOLO11 configuration missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update({"scale": "n", "nc": 2})
    # No .pt file is loaded and no pretrained weights are requested.
    torch.manual_seed(42)
    model = DetectionModel(cfg=config, ch=3, nc=2, verbose=False).to(device).eval()
    require(next(model.parameters()).device == device, "YOLO model is on the wrong device.")
    image = torch.rand(1, 3, image_size, image_size, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(2):
            model(image)  # Warm-up: not included in the diagnostic latency.
        sync(torch, device)
        start = time.perf_counter()
        output = model(image)
        sync(torch, device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        shapes = check_tensors(output, torch, device)
    result = {
        "architecture": "YOLO11n", "weights": "random/untrained", "classes": 2,
        "device": str(device), "input_shape": list(image.shape),
        "output_shapes": shapes, "finite": True,
        "single_forward_ms_not_a_benchmark": round(elapsed_ms, 3),
    }
    if device.type == "cuda":
        result["peak_torch_allocated_mib"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    # This test does not estimate accuracy, detect real damage, or select a training batch.
    return result


def basic_library_test(modules: dict[str, Any]) -> dict[str, Any]:
    np, cv2, pd, yaml = (modules[key] for key in ("numpy", "cv2", "pandas", "yaml"))
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    require(bool(encoded_ok), "OpenCV JPEG encoding failed.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    require(decoded is not None and decoded.shape == image.shape, "JPEG decoding failed.")
    array = np.arange(4, dtype=np.float32)
    require(float(modules["torch"].from_numpy(array).sum().item()) == 6.0,
            "NumPy/PyTorch bridge failed.")
    require(int(pd.DataFrame({"observations": [1, 2]})["observations"].sum()) == 3,
            "Pandas operation failed.")
    example = {"names": {0: "pothole", 1: "crack"}}
    require(yaml.safe_load(yaml.safe_dump(example)) == example, "YAML round-trip failed.")
    return {"jpeg_round_trip": True, "numpy_torch_bridge": True,
            "pandas_operation": True, "yaml_round_trip": True}


def pip_check() -> str:
    completed = subprocess.run([sys.executable, "-m", "pip", "check"],
                               capture_output=True, text=True, timeout=90)
    message = (completed.stdout + completed.stderr).strip()
    require(completed.returncode == 0, message or "pip check failed.")
    return message


def driver_diagnostic(report: Report) -> None:
    candidates = [shutil.which("nvidia-smi"),
                  str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe"),
                  str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) /
                      "NVIDIA Corporation/NVSMI/nvidia-smi.exe")]
    executable = next((item for item in candidates if item and Path(item).is_file()), None)
    if not executable:
        report.add("WARN", "NVIDIA driver query", "nvidia-smi not found; CUDA execution is checked separately.")
        return
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True)
        report.facts["nvidia_smi"] = result.stdout.strip()
        report.add("PASS", "NVIDIA driver query", result.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        report.add("WARN", "NVIDIA driver query", str(exc))


def run_checks(report: Report, gpu_index: int, image_size: int) -> None:
    current_python = tuple(sys.version_info[:3])
    report.facts.update({"python": platform.python_version(), "executable": sys.executable,
                         "os": platform.platform(), "architecture": platform.machine(),
                         "pointer_bits": struct.calcsize("P") * 8})
    report.add("PASS" if current_python == PYTHON_VERSION else "FAIL", "Python version",
               f"{platform.python_version()}; required {'.'.join(map(str, PYTHON_VERSION))}")
    gil_disabled = bool(__import__("sysconfig").get_config_var("Py_GIL_DISABLED"))
    standard_python = platform.python_implementation() == "CPython" and not gil_disabled
    report.add("PASS" if standard_python else "FAIL", "Standard CPython", platform.python_implementation())
    report.add("PASS" if struct.calcsize("P") == 8 else "FAIL", "64-bit Python", report.facts["pointer_bits"])
    report.add("PASS" if platform.system() == "Windows" else "FAIL", "Windows platform", platform.platform())
    report.add("PASS" if sys.prefix != sys.base_prefix else "FAIL", "Virtual environment", sys.executable)
    free_gib = shutil.disk_usage(ROOT).free / GIB
    report.facts["project_disk_free_gib"] = round(free_gib, 2)
    report.add("PASS" if free_gib >= 20 else "WARN", "Project disk free",
               f"{free_gib:.2f} GiB; 20 GiB is a setup headroom warning, not a dataset estimate.")
    driver_diagnostic(report)

    modules: dict[str, Any] = {}
    for distribution, module_name, expected in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            actual = metadata.version(distribution)
            modules[module_name] = module
            report.add("PASS" if actual == expected else "FAIL", distribution,
                       {"installed": actual, "required": expected,
                        "module_version": str(getattr(module, "__version__", "not exposed"))})
        except Exception as exc:
            report.add("FAIL", distribution, f"{type(exc).__name__}: {exc}")
            report.checks[-1]["traceback"] = traceback.format_exc()

    report.run("Dependency consistency", pip_check)
    if "psutil" in modules:
        memory = modules["psutil"].virtual_memory()
        report.facts["ram"] = {"total_gib": round(memory.total / GIB, 2),
                               "available_gib": round(memory.available / GIB, 2)}
        report.add("PASS" if memory.available >= 4 * GIB else "WARN", "System RAM", report.facts["ram"])
    needed = ("torch", "numpy", "cv2", "pandas", "yaml")
    if all(key in modules for key in needed):
        report.run("Library functionality", lambda: basic_library_test(modules))
    else:
        report.add("FAIL", "Library functionality", "Required imports failed; test could not run.")

    if "torch" not in modules:
        report.add("FAIL", "CUDA tests", "PyTorch did not import. No CPU fallback is counted as success.")
        return
    torch = modules["torch"]
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    report.facts["torch_cuda_runtime"] = torch.version.cuda
    report.add("PASS" if torch.version.cuda == CUDA_VERSION else "FAIL", "PyTorch CUDA runtime",
               {"installed": torch.version.cuda, "required": CUDA_VERSION})
    available = bool(torch.cuda.is_available())
    report.facts["cuda_available"] = available
    report.add("PASS" if available else "FAIL", "CUDA available", available)
    if not available:
        report.add("FAIL", "CUDA tests", "CUDA unavailable: actual GPU tests were not run.")
        return
    require(0 <= gpu_index < torch.cuda.device_count(), "Requested CUDA device index is out of range.")
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    arch_list = torch.cuda.get_arch_list()
    gpu = {"index": gpu_index, "name": properties.name,
           "vram_total_gib": round(properties.total_memory / GIB, 2),
           "compute_capability": list(capability), "compiled_architectures": arch_list}
    report.facts["gpu"] = gpu
    report.add("PASS", "GPU hardware", gpu)
    require(properties.total_memory > 0, "GPU VRAM could not be determined.")
    if "5050" not in properties.name:
        report.add("WARN", "Expected GPU", "Selected GPU is not named RTX 5050; check the device selection.")
    arch = f"sm_{capability[0]}{capability[1]}"
    report.add("PASS" if arch in arch_list else "WARN", "Native GPU architecture",
               f"{arch}; actual CUDA execution, not this list alone, determines compatibility.")
    try:
        free, total = torch.cuda.mem_get_info(device)
        report.facts["gpu_memory_snapshot"] = {"free_gib": round(free / GIB, 2),
                                               "total_gib": round(total / GIB, 2)}
        report.add("PASS" if free >= GIB else "WARN", "GPU free memory", report.facts["gpu_memory_snapshot"])
    except Exception as exc:
        report.add("WARN", "GPU free memory", str(exc))
    cnn_ok = report.run("CUDA tensor + CNN inference", lambda: tensor_and_conv_test(torch, device))
    nms_ok = (report.run("CUDA torchvision NMS", lambda: nms_test(torch, modules["torchvision"], device))
              if "torchvision" in modules else False)
    if "torchvision" not in modules:
        report.add("FAIL", "CUDA torchvision NMS", "torchvision import failed.")
    yolo_ok = False
    if cnn_ok and all(key in modules for key in ("ultralytics", "yaml")):
        yolo_ok = report.run("CUDA YOLO11n inference", lambda: yolo_test(
            torch, modules["ultralytics"], modules["yaml"], device, image_size))
    else:
        report.add("FAIL", "CUDA YOLO11n inference", "Required imports or CUDA CNN test failed.")
    report.facts["actual_gpu_tests_passed"] = cnn_ok and nms_ok and yolo_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index, default 0.")
    parser.add_argument("--image-size", type=int, choices=(160, 320, 640), default=320)
    parser.add_argument("--json", type=Path, default=ROOT / "runs/environment_check.json")
    args = parser.parse_args()
    # Apply before importing Ultralytics; never read or print secret environment values.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
    report = Report()
    payload: dict[str, Any] = {"phase": "1.1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                               "status": "RUNNING", "phase_1_1_passed": False,
                               "training_batch_size": None, "dataset_downloaded": False}
    print("Pothole Detection Drone - Phase 1.1 environment check", flush=True)
    print("Untrained diagnostic tensors only. No dataset, weights download, or training.", flush=True)
    try:
        write_json(args.json, payload)  # Invalidate a stale PASS before starting.
        run_checks(report, args.device, args.image_size)
    except Exception as exc:
        report.add("FAIL", "Diagnostic execution", f"{type(exc).__name__}: {exc}")
        report.checks[-1]["traceback"] = traceback.format_exc()
    passed = not report.failed and bool(report.facts.get("actual_gpu_tests_passed"))
    payload.update({"status": "PASS" if passed else "FAIL", "phase_1_1_passed": passed,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "facts": report.facts, "checks": report.checks})
    try:
        write_json(args.json, payload)
    except OSError as exc:
        print(f"[FAIL] Cannot write report: {exc}", flush=True)
        passed = False
    print(f"\nPHASE 1.1: {'PASS' if passed else 'FAIL'}", flush=True)
    print(f"Report: {args.json.resolve()}", flush=True)
    print("Training batch: NOT SELECTED. Phase 1.2 has NOT been started.", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
