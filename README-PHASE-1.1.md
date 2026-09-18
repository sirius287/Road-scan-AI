# Pothole Detection Drone — Phase 1.1

Environment setup and diagnostics only. No dataset download, pretrained-weight
load, training, backend, hardware firmware, or progression to Phase 1.2.

## Target and selected baseline

Target: Windows 10 x64, NVIDIA RTX 5050 Laptop GPU / 8 GB VRAM, 16 GB system RAM.
The CPU and 512 GB SSD are the user's stated specifications; available storage,
driver, actual GPU properties, and runtime behavior must be checked locally.

| Component | Selection |
|---|---|
| Python | CPython 3.13.15, standard Windows x86-64 build, not free-threaded |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA runtime used by torch | 12.8 |
| Ultralytics | 8.4.154 |
| OpenCV distribution | opencv-python 4.12.0.88 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| YAML distribution | PyYAML 6.0.3 |
| pytest | 8.4.2 |
| psutil | 7.2.2 |

These are selected project pins, not a claim that every component is the latest
release or that this exact environment has already passed on the user's laptop.
Python 3.13.15 replaces the provisional Python 3.11 planning suggestion. Official
Windows CPython 3.13 wheels exist for the selected torch/torchvision CUDA pair.
The model choice remains YOLO11n despite the newer Ultralytics package version.

Use an NVIDIA/OEM driver that explicitly supports this laptop GPU and Windows 10
64-bit. Keep a working, current driver; do not install an older generic CUDA
minimum driver simply because a compatibility table lists it. An NVIDIA driver
is needed; a separate CUDA Toolkit, nvcc, cuDNN download, Visual Studio compiler,
WSL, and torchaudio are not needed for this prebuilt-wheel workflow.

Windows 10 standard support ended on October 14, 2025. Keep appropriate Extended
Security Updates in place or plan a supported OS migration; the commands below
remain native Windows PowerShell commands.

## 1. Driver and hardware inspection — PowerShell

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
Get-CimInstance Win32_Processor | Select-Object Name
Get-PSDrive -PSProvider FileSystem
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

If nvidia-smi is not found, inspect the installed driver first. A possible
fallback location is C:\Windows\System32\nvidia-smi.exe. NVIDIA's driver selector
can be opened with:

```powershell
Start-Process "https://www.nvidia.com/en-us/drivers/"
```

Select the laptop/notebook GPU and Windows 10 64-bit, or use the laptop vendor's
supported driver. Reboot after changing the driver. The CUDA/CUDA UMD version
shown by nvidia-smi describes driver capability; it need not equal torch's 12.8.

## 2. Install the exact Python version

```powershell
Start-Process "https://www.python.org/downloads/release/python-31315/"
```

Choose **Windows installer (64-bit)**, not ARM64, 32-bit, embeddable, or an
experimental free-threaded build. Include pip and the Python launcher. Reopen
PowerShell after installation.

```powershell
py -3.13 --version
py -3.13 -c "import sys, struct; print(sys.executable); print(struct.calcsize('P') * 8)"
```

Expected Python 3.13.15 and 64. Multiple Python installations are fine, but this
project must use the selected interpreter. The checker fails an exact-version
mismatch rather than silently validating another baseline.

## 3. Extract and create a clean virtual environment

Save pothole-drone-phase1-1.zip to Downloads. The ZIP contains the files directly,
without another enclosing directory. Existing files are not overwritten by the
Expand-Archive command below.

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\source\pothole-drone" | Out-Null
Expand-Archive -LiteralPath "$HOME\Downloads\pothole-drone-phase1-1.zip" -DestinationPath "$HOME\source\pothole-drone"
Set-Location "$HOME\source\pothole-drone"
if (Test-Path ".\.venv") { throw "Use a clean project directory; do not overwrite an existing virtual environment." }
py -3.13 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
.\.venv\Scripts\python.exe --version
```

All subsequent commands invoke the virtual-environment interpreter directly.
Activation is unnecessary and no PowerShell execution-policy change is needed.
Do not use a Linux-style `source` command.

## 4. Install dependencies — in this order

Run each block successfully before the next one.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
```

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
```

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -c .\constraints-env.txt -r .\requirements-env.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }
.\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency conflicts must be fixed before testing." }
```

The requirements and constraints together pin all explicitly selected runtime
packages. Transitive dependencies are resolved by pip and recorded after success.
The binary-only option avoids silently attempting a source build. Do not install
opencv-python-headless or opencv-contrib-python into this environment alongside
opencv-python. FastAPI, Streamlit, and reporting packages are not selected here.

## 5. Run the environment diagnostic

```powershell
.\.venv\Scripts\python.exe .\training\check_environment.py
$EnvironmentExit = $LASTEXITCODE
Write-Host "Environment check exit code: $EnvironmentExit"
if ($EnvironmentExit -ne 0) { throw "Phase 1.1 is blocked. Review runs\environment_check.json." }
```

The script checks interpreter/version/venv, all listed imports and versions,
pip consistency, JPEG encoding/decoding, NumPy/PyTorch interoperability, Pandas,
YAML, RAM and disk headroom, the NVIDIA driver query where available, CUDA build,
GPU identity, total VRAM, free VRAM where detectable, and compiled architectures.

It then requires actual CUDA execution of:

1. Matrix multiplication and a tiny CNN in inference mode.
2. torchvision.ops.nms on known overlapping boxes; expected indices [0, 2].
3. An untrained two-class YOLO11n forward pass from the installed YAML, using one
   320 x 320 random input tensor, two warm-ups, and a synchronized measured pass.

All returned tensors must be nonempty, finite, and on the selected CUDA device.
No network requests for data/weights and no automatic package installation are
requested. Ultralytics offline mode is enabled before its import. The model is
randomly initialized: this proves execution only, not damage detection quality.
It is a raw model-forward diagnostic, not an image-to-final-box accuracy test.

A report is written to runs\environment_check.json, including on ordinary errors.
A stale successful report is invalidated at startup. No CPU fallback is accepted
as a successful Phase 1.1 run. CUDA errors should be resolved before moving on.
The script never selects a training batch size.

Optional diagnostic at the planned input resolution (still not training):

```powershell
.\.venv\Scripts\python.exe .\training\check_environment.py --image-size 640 --json .\runs\environment_check_640.json
```

## 6. Run the checker's unit tests and freeze the successful environment

```powershell
.\.venv\Scripts\python.exe -m pytest -q .\tests\test_environment_check.py
if ($LASTEXITCODE -ne 0) { throw "Diagnostic unit tests failed." }
.\.venv\Scripts\python.exe -m pip freeze --all | Set-Content -Encoding ascii .\requirements-win-cuda.lock.txt
```

Expected: 13 passed. The unit tests deliberately exercise CPU-safe helper logic;
they do not replace the separate CUDA diagnostic.

## Expected successful output

The following is illustrative, not measured output from the user's laptop.
Exact free memory, driver version, GPU spelling, tensor details and timings vary.

```text
[PASS] Python version: 3.13.15; required 3.13.15
[PASS] torch: {"installed": "2.11.0+cu128", ...}
[PASS] torchvision: {"installed": "0.26.0+cu128", ...}
[PASS] ultralytics: {"installed": "8.4.154", ...}
[PASS] Dependency consistency: No broken requirements found.
[PASS] PyTorch CUDA runtime: {"installed": "12.8", "required": "12.8"}
[PASS] CUDA available: true
[PASS] GPU hardware: {"name": "NVIDIA GeForce RTX 5050 Laptop GPU", ...}
[PASS] CUDA tensor + CNN inference: {"device": "cuda:0", ...}
[PASS] CUDA torchvision NMS: {"device": "cuda:0", "kept_indices": [0, 2]}
[PASS] CUDA YOLO11n inference: {"architecture": "YOLO11n", "weights": "random/untrained", ...}
PHASE 1.1: PASS
Training batch: NOT SELECTED. Phase 1.2 has NOT been started.
```

OpenCV's module version normally prints 4.12.0 even though its distribution
version is 4.12.0.88. VRAM is reported in binary GiB, and reported usable total can
differ slightly from the marketed capacity. Review every WARN line. The measured
single forward time is diagnostic only: it is not end-to-end FPS, training speed,
or a benchmark, and no ML accuracy metrics are produced.

## Troubleshooting

| Symptom | Action |
|---|---|
| Wrong Python or 32-bit interpreter | Use the standard 3.13.15 x64 installer and a fresh venv. |
| CPU-only torch / CUDA runtime null | Reinstall the exact torch/torchvision pair from the cu128 index using the venv interpreter. |
| CUDA unavailable but nvidia-smi works | Check the selected interpreter, CUDA wheel pair, driver and whether the laptop has disabled the discrete GPU. |
| sm_120 / no kernel image / invalid device function | Check the CUDA-enabled binary pair and driver. Setting TORCH_CUDA_ARCH_LIST will not rebuild an installed wheel. |
| torchvision NMS error | Check matched torch/torchvision versions and CUDA suffixes; do not mix CPU torchvision with CUDA torch. |
| DLL import error | Check Python x64, NVIDIA driver and the official Microsoft Visual C++ x64 runtime; do not download individual DLLs from third-party sites. |
| CUDA out of memory | Close GPU-heavy apps, reboot if appropriate, then rerun. Do not infer a training batch from a failed smoke test. |
| NumPy/OpenCV dependency conflict | Use the requirements and constraints in a clean venv; do not add multiple OpenCV distributions. |
| Multiple Python installations | Use .venv\Scripts\python.exe, not an unqualified pip command. |
| pip resolver reports no compatible wheel | Verify interpreter/platform and preserve the error output; do not bypass dependencies with --no-deps. |

## Memory policy for later work

The batch of one is only a minimal inference probe. It is not a proposed training
batch. During the later training smoke-test step, profile the real model and
training resolution, include forward/backward and optimizer memory, leave VRAM
headroom, and validate a dynamically selected batch with real labeled batches.
Ultralytics AutoBatch can inform the initial estimate; handle failure/fallback
explicitly rather than assuming its returned default is safe. Start Windows
DataLoader validation with workers=0, keep image RAM caching off, and test mixed
precision separately. These are future configuration decisions, not actions
performed by this package.

## What was tested when preparing this package

Python syntax compilation, 13 CPU-safe diagnostic unit tests, and a full checker
run on a non-target CPU-only host. The latter correctly returned exit code 1 and
saved a failure report. That host was not Windows, did not have CUDA, and did not
have Ultralytics installed. The exact target installation, YOLO forward path and
RTX 5050 CUDA execution still require the local commands above. No pass report
for the user's laptop is claimed or included.

## Completion gate

All required environment checks pass, CUDA CNN/NMS/YOLO checks succeed, the command
returns 0, warnings are reviewed, and diagnostic unit tests pass. Preserve the
JSON report and the generated Windows lock snapshot. Stop here; do not download
a dataset or start Phase 1.2 automatically.

## Official references checked for the selected setup

- Python release: https://www.python.org/downloads/release/python-31315/
- Python venv documentation: https://docs.python.org/3.13/library/venv.html
- PyTorch version pair / install commands: https://pytorch.org/get-started/previous-versions/
- Windows torch wheels: https://download.pytorch.org/whl/cu128/torch/
- Windows torchvision wheels: https://download.pytorch.org/whl/cu128/torchvision/
- NVIDIA compute capabilities: https://developer.nvidia.com/cuda/gpus
- NVIDIA CUDA 12.8 release notes: https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html
- NVIDIA nvidia-smi reference: https://docs.nvidia.com/deploy/nvidia-smi/index.html
- Ultralytics release: https://pypi.org/project/ultralytics/8.4.154/
- Ultralytics selected source: https://github.com/ultralytics/ultralytics/tree/v8.4.154
- Ultralytics AutoBatch: https://docs.ultralytics.com/reference/utils/autobatch/
