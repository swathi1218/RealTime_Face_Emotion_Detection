# EmotionLens 🎭

> Real-time facial emotion recognition powered by MediaPipe + EfficientNet

EmotionLens is a production-quality desktop application that captures your webcam feed, detects faces with MediaPipe, and classifies emotions in real time using an EfficientNet-based model from HSEmotion — all on CPU, no GPU required.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🎥 Live webcam feed | OpenCV capture at 640×480, configurable |
| 👤 Multi-face support | Tracks and labels up to N faces simultaneously |
| 🧠 EfficientNet-B0 inference | HSEmotion `enet_b0_8_best_afew`, 7 emotion classes |
| ⏱ Temporal smoothing | Majority vote + EMA to eliminate prediction flicker |
| 📊 Live statistics panel | Per-emotion distribution, session time, avg confidence |
| 🗂 CSV logging | Auto-flushed, auto-rotated prediction log |
| ⚡ Frame skipping | Configurable skip gate for CPU performance tuning |
| 🎨 Dark HUD overlay | Colour-coded bounding boxes + mini probability bars |

**Emotions detected:** Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise

---

## 🏗 Architecture

```
WebcamThread ──► InferenceWorker ──► MainWindow (GUI)
  (OpenCV)          ├─ FaceDetector (MediaPipe)
                    ├─ EmotionRecognizer (HSEmotion / EfficientNet)
                    ├─ SmootherPool (per-face temporal smoothing)
                    ├─ EmotionLogger (CSV)
                    └─ Drawing utilities (OpenCV overlays)
```

All heavy work runs off the GUI thread. Qt signals ensure thread-safe communication.

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and [`docs/methodology.md`](docs/methodology.md) for algorithm details.

---

## 📦 Installation

```bash
git clone https://github.com/your-org/emotionlens.git
cd emotionlens
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See [`docs/installation.md`](docs/installation.md) for full platform-specific instructions.

---

## 🚀 Usage

```bash
cd src
python main.py
```

Click **▶ Start** to open the webcam. The pipeline begins automatically.

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| Coming soon | — |

### Configuration

All tuneable parameters live in [`src/utils/config.py`](src/utils/config.py):

```python
CAMERA_WIDTH = 640          # Capture resolution
CAMERA_HEIGHT = 480
FACE_CONFIDENCE_THRESHOLD = 0.5   # MediaPipe detection threshold
SMOOTHING_WINDOW = 10       # Temporal smoothing history size
FRAME_SKIP = 2              # Run inference every N frames
HSEMOTION_MODEL = "enet_b0_8_best_afew"
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Tests cover: face detection helpers, temporal smoother, CSV logger, FPS counter, and drawing utilities. No webcam or GPU is required to run the tests.

---

## 📁 Project Structure

```
EmotionLens/
├── src/
│   ├── main.py                   ← Application entry point
│   ├── camera/webcam.py          ← OpenCV capture thread
│   ├── detection/face_detector.py← MediaPipe face detection
│   ├── emotion/
│   │   ├── emotion_recognizer.py ← HSEmotion inference wrapper
│   │   └── smoothing.py          ← Temporal stabilisation
│   ├── ui/
│   │   ├── main_window.py        ← PyQt6 main window + inference worker
│   │   └── widgets.py            ← Reusable custom widgets
│   ├── logs/emotion_logger.py    ← CSV logging
│   └── utils/
│       ├── config.py             ← Centralised configuration
│       ├── drawing.py            ← OpenCV overlay utilities
│       └── fps.py                ← Rolling FPS counter
├── tests/                        ← pytest test suite
├── docs/                         ← Architecture & methodology docs
├── data/                         ← emotion_logs.csv (auto-created)
└── requirements.txt
```

---

## 🔭 Future Improvements

- [ ] Action unit (AU) detection via OpenFace integration
- [ ] ONNX runtime export for faster CPU inference
- [ ] Multi-camera source selector in the GUI
- [ ] Session replay from saved CSV logs
- [ ] REST API mode for headless server deployment
- [ ] Attention heatmap visualisation (Grad-CAM)

---

## 📄 License

MIT — see [LICENSE](LICENSE).
