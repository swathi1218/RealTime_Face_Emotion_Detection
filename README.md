# Real-Time Facial Emotion Recognition

This is a desktop application that performs real-time facial emotion recognition using a webcam feed. It detects faces using MediaPipe, classifies emotions using the HSEmotion deep learning library (EfficientNet-based model), and displays live results through a PyQt6 GUI with per-face annotations and a session analytics panel.

---

## Prerequisites

- Python 3.11 (required — Python 3.12+ has dataclass incompatibilities with timm 0.6.13)
- Git (required to install timm from source)
- A working webcam
- Windows 10/11 (tested environment; macOS/Linux should work with minor path differences)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/swathi1218/EmotionLens.git
cd EmotionLens
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `timm` is installed directly from the GitHub source at tag `v0.6.13` rather than PyPI. This is intentional — the PyPI wheel for `timm 0.6.13` is missing the `timm/layers/` subpackage on Windows. The git install builds the wheel locally from source, which includes all files correctly.

### 4. Run the Application

```bash
python -m src.main
```

The GUI will open. The HSEmotion model will be downloaded automatically on first run (~50 MB) and cached in your home directory under `.hsemotion/`. Allow 5–10 seconds for the model to load before the camera feed appears.

---

## What I Learned / Found Challenging

The most challenging aspect of this project was the `timm` version compatibility chain. The HSEmotion library depends on `timm`, but the PyPI wheel for `timm 0.6.13` is missing the `timm/layers/` subpackage on Windows — meaning `pip show timm` reports the correct version, yet `import timm.layers` fails. This created a confusing situation where standard reinstall commands had no effect.

The deeper issue was that the saved model `.pt` file was pickled with direct references to internal timm module paths (`timm.layers.conv2d_same`, `timm.models._efficientnet_blocks`) that no longer exist under those names. Python's pickle module tries to resolve these paths at load time and fails with `ModuleNotFoundError` or `AttributeError` depending on what it finds.

The solution required two independent fixes: installing `timm` from the GitHub source at the `v0.6.13` tag (bypassing the broken PyPI wheel), and patching `sys.modules` at runtime to register all `timm.models.layers` submodules under their old `timm.layers.*` paths before the model loads.

This experience reinforced that `pip show` and `import` behaviour can diverge when a package's wheel is partially extracted, and that pickle-serialised PyTorch models carry hard dependencies on the exact internal structure of every library used at training time.
