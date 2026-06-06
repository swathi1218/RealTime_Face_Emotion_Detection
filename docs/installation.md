# EmotionLens — Installation Guide

## Requirements

| Requirement | Minimum version |
|-------------|-----------------|
| Python | 3.11 |
| OS | Windows 10+, macOS 12+, Ubuntu 20.04+ |
| CPU | Any modern x86-64 or Apple Silicon |
| RAM | 4 GB (8 GB recommended) |
| GPU | **Not required** — CPU-only inference |
| Webcam | Any UVC-compatible webcam |

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/emotionlens.git
cd emotionlens
```

## 2. Create a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate.bat       # Windows
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `hsemotion` will automatically download the EfficientNet model weights
> (~20 MB) on first run into `~/.hsemotion/`.

## 4. Install EmotionLens (optional editable install)

```bash
pip install -e .
```

This registers the `emotionlens` CLI entry point.

---

## Running the Application

### Option A — direct script

```bash
cd src
python main.py
```

### Option B — CLI entry point (after editable install)

```bash
emotionlens
```

---

## Running Tests

```bash
pytest tests/ -v
```

To run only fast unit tests (no camera or model required):

```bash
pytest tests/ -v -k "not slow"
```

---

## Troubleshooting

### Camera not opening
- Check `CAMERA_CONFIG.device_index` in `src/utils/config.py` (default `0`).
- On Linux, ensure your user is in the `video` group: `sudo usermod -aG video $USER`.

### HSEmotion model download fails
- Check your internet connection; the model is hosted on Hugging Face.
- You can manually place the `.pth` file in `~/.hsemotion/` — see HSEmotion docs.

### PyQt6 display issues on Linux
- Install system Qt6 libraries: `sudo apt install python3-pyqt6 libgl1`.

### Slow inference
- Increase `FRAME_SKIP` in `src/utils/config.py` (e.g. `4` → run inference every 4th frame).
- Reduce `CAMERA_WIDTH` / `CAMERA_HEIGHT` to `320×240`.
