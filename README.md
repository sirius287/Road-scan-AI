

````markdown
# 🚧 Pothole Detection Drone

AI-based road damage detection using **YOLO + PyTorch + CUDA**.  
Detects potholes and other road defects from images/video.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/sirius287/pothole-drone.git
cd pothole-drone
````

### 2. Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements-env.txt
```

For the NVIDIA CUDA setup:

```powershell
pip install --only-binary=:all: torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Verify environment

```powershell
python training/check_environment.py
```

You should see:

```text
PHASE 1.1: PASS
```

## 🚀 Run Detection

### Image detection

Place an image in the project folder and run:

```powershell
python -c "from ultralytics import YOLO; m=YOLO('pothole_best.pt'); m.predict(source='test.jpg',device=0,save=True,conf=0.20,imgsz=640)"
```

Results are saved under:

```text
runs/detect/
```

### 🌐 Run the App

```powershell
python app.py
```

## 📁 Main Files

```text
app.py                  # Application
pothole_best.pt         # Pothole detection model
training/               # Training & environment scripts
dataset_tools/          # Dataset utilities
configs/                # Dataset configuration
tests/                  # Tests
```


