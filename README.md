# Real-Time Driver Drowsiness Detection System

A real-time drowsiness detection system that uses a standard webcam to monitor driver alertness continuously. The system combines facial landmark geometry with deep learning classification to detect eye closure and yawning, triggering an audio alert when drowsiness is identified.

---

## Overview

Driver drowsiness is a leading cause of road accidents. This system provides a non-intrusive, camera-based solution that runs entirely on CPU with no specialized hardware required.

**Detection signals:**
- Eye Aspect Ratio (EAR) — geometric measure of eye openness from facial landmarks
- Mouth Aspect Ratio (MAR) — geometric measure of mouth opening for yawn detection
- Eye State CNN — EfficientNetB0 classifier for open/closed eye state (displayed as supplementary info)
- Yawn CNN — EfficientNetB0 classifier for yawn/no-yawn detection (displayed as supplementary info)

**Alert levels:**
- `NORMAL` — driver is alert
- `YAWNING` — sustained mouth opening detected
- `DROWSY` — sustained eye closure detected
- `HIGH ALERT` — both conditions active simultaneously

---

## Project Structure

```
DROWSINESS_CV/
├── assets/
│   └── alarm.wav                  # Alert sound file
├── dataset/
│   ├── train_dataset/
│   │   ├── open/                  # Open eye images  (label 0)
│   │   └── closed/                # Closed eye images (label 1)
│   ├── yawn/                      # Yawning mouth images (label 1)
│   └── no yawn/                   # Normal mouth images  (label 0)
├── model/
│   ├── eye_model.keras            # Trained eye state classifier
│   └── yawn_model.keras           # Trained yawn classifier
├── utils/
│   ├── landmarks.py               # MediaPipe face landmark detection
│   ├── ear_mar.py                 # EAR and MAR geometry calculations
│   ├── logic.py                   # Fusion decision engine
│   ├── alert.py                   # Audio alert system
│   ├── model_predict.py           # CNN inference wrapper
│   ├── train.py                   # Eye CNN trainer
│   └── train2.py                  # Yawn CNN trainer
├── test_camera.py                 # Live webcam detection (run this)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- Webcam
- CPU (no GPU required)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/SamaHassa/Drowsiness_CV.git
cd DROWSINESS_CV
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run live detection

```bash
python test_camera.py
```

**Controls:**
- `Q` — quit
- `R` — reset counters
- `S` — save screenshot to `debug/`

---

## Training the Models

If you want to retrain the models on your own dataset:

**Eye state model** (open vs closed):

```bash
python utils/train.py
```

Dataset required:
```
dataset/train_dataset/open/    <- open eye images
dataset/train_dataset/closed/  <- closed eye images
```

**Yawn detection model** (yawn vs no yawn):

```bash
python utils/train2.py
```

Dataset required:
```
dataset/yawn/     <- yawning mouth images
dataset/no yawn/  <- normal mouth images
```

Both trainers automatically split data into 70% train / 15% validation / 15% test, print a classification report on the test set, and save training history plots to `model/`.

---

## Model Performance

| Model | Train Acc | Val Acc | Test Acc | Train-Val Gap |
|-------|-----------|---------|----------|---------------|
| Eye CNN (EfficientNetB0) | 97.20% | 95.46% | 94.09% | 1.74% |
| Yawn CNN (EfficientNetB0) | 99.75% | 98.44% | 98.70% | 1.31% |

Both models show no significant overfitting. Full classification reports are printed at the end of each training run.

---

## How It Works

### Face Landmark Detection

MediaPipe FaceLandmarker detects 478 facial landmarks per frame. Specific landmark indices are selected for the eye and mouth regions.

### Eye Aspect Ratio (EAR)

Based on Soukupova & Cech (2016). Computed from 6 points around each eye:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

A rolling average over 6 frames reduces noise. EAR drops below 0.20 when eyes are truly closed. Threshold was set lower than the common 0.25 to avoid false alarms from squinting.

### Mouth Aspect Ratio (MAR)

Computed from 8 mouth landmark points:

```
MAR = (||p3-p4|| + ||p7-p8||) / (2 * ||p1-p2||)
```

MAR exceeds 0.60 during a genuine yawn. Values during normal speech and smiling stay below this threshold.

### Fusion Logic

Alert triggers after sustained conditions across 3 consecutive frames (~0.6 seconds at 5 FPS). Counters reset to zero immediately when conditions clear, so the alert stops as soon as the driver returns to normal.

### Deep Learning

Both CNNs use EfficientNetB0 pretrained on ImageNet with a two-phase training strategy: Phase 1 trains the classification head only (base frozen, LR=1e-3), Phase 2 fine-tunes the last 30 base layers (LR=1e-4). CNN predictions are displayed but alerts are driven by geometry, which is more reliable on live webcam input due to domain differences from training data.

---

## Dataset

The eye dataset uses images from publicly available eye state datasets. The yawn dataset uses cropped mouth images labeled as yawn/no-yawn. Both datasets are available on Kaggle.

If using your own dataset, ensure:
- Eye images: any size (resized to 224x224 during training)
- Mouth images: any size (resized to 224x224 during training)
- Balanced classes for best results

---

## Known Limitations

- FPS is approximately 4-6 on CPU due to CNN inference overhead
- Performance degrades in very low light conditions
- Glasses frames may interfere with eye landmark precision
- MediaPipe accuracy decreases with head angles beyond 45 degrees

---

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Face landmarks | MediaPipe | 0.10.11 |
| Computer vision | OpenCV | 4.9+ |
| Deep learning | TensorFlow / Keras | 2.16+ |
| CNN backbone | EfficientNetB0 | ImageNet pretrained |
| Numerical operations | NumPy | 1.26+ |
| ML utilities | scikit-learn | 1.4+ |

---

## References

- Soukupova, T. & Cech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. 21st Computer Vision Winter Workshop.
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.
- Google MediaPipe FaceLandmarker: https://developers.google.com/mediapipe/solutions/vision/face_landmarker

---

## License

This project is developed for academic purposes as part of a 4th year Computer Science project.
