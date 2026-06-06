# EmotionLens — Methodology

## 1. Face Detection

**Library:** MediaPipe Face Detection  
**Model:** Short-range (`model_selection=0`, optimised for faces within ~2 m)

MediaPipe's BlazeFace architecture uses a lightweight CNN with depthwise-separable convolutions to achieve real-time detection on CPU. It returns normalised bounding boxes relative to the input frame dimensions.

**Post-processing:**
1. Filter detections below the configurable confidence threshold (default 0.5).
2. Expand each bounding box by a `padding_ratio` (default 15 %) to capture hair and chin.
3. Clamp to frame boundaries.
4. Discard crops narrower or shorter than `min_face_size` pixels to avoid noise.
5. Sort left-to-right and assign integer `face_id` values (1, 2, 3, …) for consistent labelling.

## 2. Face Cropping

The `crop_face()` function extracts a rectangular sub-image from the BGR frame using the validated `[x, y, w, h]` bounding box. Boundary validation ensures that no slice operation exceeds the frame dimensions, avoiding silent truncation artefacts that would degrade emotion model accuracy.

## 3. Emotion Recognition

**Library:** HSEmotion (`hsemotion` PyPI package)  
**Model:** `enet_b0_8_best_afew` — EfficientNet-B0 fine-tuned on the AFEW dataset (7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).

**Why EfficientNet?**  
EfficientNet scales depth, width, and resolution jointly, achieving strong accuracy at low parameter counts. The B0 variant is particularly suitable for CPU deployment: it runs inference on a 224×224 face crop in ~30–80 ms on a modern laptop CPU, enabling real-time processing with frame skipping.

**Inference pipeline:**
1. HSEmotion internally resizes the face crop to 224×224 and applies standard ImageNet normalisation.
2. The model outputs a 7-dimensional softmax probability vector.
3. EmotionLens reads the winning label and full score vector.

## 4. Temporal Smoothing

Raw per-frame predictions from any CNN fluctuate between adjacent frames due to minor lighting changes, pose variations, and stochastic rounding. Two techniques are combined:

### 4a. Majority Voting
A rolling deque of the last `window_size` (default 10) predictions is maintained per face. The label that appears most often wins. Ties are broken by choosing the label with the highest average confidence among tied candidates. This produces stable, jump-free emotion labels.

### 4b. Exponential Moving Average (EMA)
The full probability distribution is smoothed using EMA with a configurable `ema_alpha` (default 0.3):

```
ema[t] = alpha * score[t] + (1 - alpha) * ema[t-1]
```

Lower alpha values produce smoother but more lagged bars; higher values are more responsive.

Both techniques are implemented in `EmotionSmoother`, with one smoother instance maintained per tracked face via `SmootherPool`. Stale smoothers (no detections for `max_idle_frames` frames) are evicted to free memory.

## 5. Visualisation

All rendering is done with OpenCV (`cv2`) drawing primitives directly on a copy of the BGR frame:

- **Bounding box:** Colour-coded per emotion; corner accent marks for a modern HUD aesthetic.
- **Label badge:** Semi-transparent filled rectangle behind the emotion label prevents text from blending into the background.
- **Emotion bars:** Mini horizontal bar chart showing the EMA-smoothed probability distribution per face.
- **FPS counter:** Top-right overlay; yellow text for high contrast.
- **No-face hint:** Subtle grey text when no faces are detected.

## 6. CSV Logging

`EmotionLogger` writes four columns per detection event:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `YYYY-MM-DD HH:MM:SS.mmm` | Wall-clock event time |
| `face_id` | integer | Per-frame face index |
| `emotion` | string | Smoothed emotion label |
| `confidence` | float (4 dp) | Smoothed confidence score |

Rows are accumulated in an in-memory buffer and flushed every `flush_interval` rows (default 10) to minimise disk I/O without risking data loss on crash. When the log exceeds `max_rows` rows the file is rotated with a timestamp suffix.
