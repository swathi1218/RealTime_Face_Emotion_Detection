# EmotionLens — System Architecture

## Overview

EmotionLens is structured as a **producer–worker–consumer** pipeline where each stage runs in its own thread, communicating exclusively via Qt signals to keep the GUI responsive at all times.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GUI Thread (PyQt6)                          │
│                                                                     │
│   MainWindow                                                        │
│     ├── VideoDisplay (QLabel)        ← annotated frames             │
│     ├── StatsPanel  (custom widget)  ← aggregated statistics        │
│     └── StatusBar   (custom widget)  ← FPS / camera / model state   │
└────────────────────────┬───────────────────────────────────────────┘
                         │ Qt Signals (thread-safe)
          ┌──────────────┴──────────────────┐
          │                                 │
┌─────────▼──────────┐          ┌───────────▼──────────────────────┐
│  WebcamThread      │  frame   │  InferenceWorker                  │
│  (QThread)         │ ───────► │  (QThread)                        │
│                    │          │                                    │
│  OpenCV VideoCapture          │  FaceDetector (MediaPipe)         │
│  FPSCounter        │          │    └─► crop_face()                 │
│                    │          │  EmotionRecognizer (HSEmotion)     │
└────────────────────┘          │    └─► InferenceThread (skip gate) │
                                │  SmootherPool                     │
                                │    └─► EmotionSmoother (per face)  │
                                │  EmotionLogger (CSV)              │
                                │  Drawing utilities                │
                                └───────────────────────────────────┘
```

## Component Responsibilities

### WebcamThread
- Opens the webcam via `cv2.VideoCapture`
- Reads frames on a tight loop
- Emits `(frame, fps)` signals; never does any image processing

### InferenceWorker
- Maintains an internal frame buffer; processes the most recent pending frame
- **Frame skip gate** (`InferenceThread`): runs detection+recognition every N frames, replays cached results on skipped frames
- Handles the full pipeline: detect → crop → infer → smooth → draw → log

### FaceDetector
- Thin wrapper around `mediapipe.solutions.face_detection`
- Converts relative bounding boxes to absolute pixel coordinates with configurable padding
- Sorts detections left-to-right for consistent face ID assignment

### EmotionRecognizer
- Wraps HSEmotion's `HSEmotionRecognizer`
- Handles model download, loading, and CPU inference
- Returns typed `EmotionResult` dataclasses

### SmootherPool + EmotionSmoother
- Pool manages per-face smoother instances with automatic idle eviction
- Smoother maintains a rolling deque of predictions
- Combines **majority voting** (for label stability) with **EMA** (for smooth confidence values)

### EmotionLogger
- Buffers rows in memory and flushes every `flush_interval` writes
- Auto-rotates the CSV file when `max_rows` is exceeded
- Thread-safe for single-writer use (InferenceWorker)

### Drawing Utilities
- All OpenCV rendering is centralised in `src/utils/drawing.py`
- Functions operate in-place on BGR arrays for minimal allocations

## Threading Model

| Thread | Runs | Communicates via |
|--------|------|-----------------|
| GUI (main) | Qt event loop | — receives signals |
| WebcamThread | OpenCV read loop | `frame_ready`, `camera_error`, `camera_opened` signals |
| InferenceWorker | Inference loop | `result_ready`, `model_ready` signals |

No mutex/lock primitives are needed because Qt's signal–slot mechanism handles cross-thread delivery automatically with queued connections.

## Data Flow (per frame)

```
WebcamThread.read()
    → emit frame_ready(frame, fps)
        → InferenceWorker.submit_frame(frame, fps)
            → [frame skip gate]
            → FaceDetector.detect(frame)
                → for each FaceDetection:
                    crop_face(frame, bbox)
                    → EmotionRecognizer.predict(crop)
                        → EmotionSmoother.update(emotion, confidence, scores)
                            → EmotionLogger.log(...)
            → Drawing utilities annotate output frame
            → emit result_ready(annotated, detections, fps)
                → MainWindow renders pixmap + updates widgets
```
