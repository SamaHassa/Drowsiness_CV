import os
import cv2
import numpy as np

EYE_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "model", "eye_model.keras")
YAWN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "yawn_model.keras")

IMG_SIZE = (224, 224)

LEFT_EYE_IDX  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_IDX = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
MOUTH_IDX     = [61,291,0,17,78,308,13,14,
                 146,91,181,84,314,405,321,375,
                 191,80,81,82,312,311,310,415]


class DrowsinessModel:

    def __init__(self):
        self._eye_model  = self._load(EYE_MODEL_PATH,  "Eye")
        self._yawn_model = self._load(YAWN_MODEL_PATH, "Yawn")

    @staticmethod
    def _load(path, name):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            print(f"[Model] ⚠  {name} model not found: {abs_path}")
            return None
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(abs_path)
            print(f"  Loaded {name} model -> {abs_path}")
            return model
        except Exception as e:
            print(f"[Model] ✗ {name} failed: {e}")
            return None

    @property
    def eye_available(self):  return self._eye_model  is not None
    @property
    def yawn_available(self): return self._yawn_model is not None

    @staticmethod
    def _crop(frame, indices, landmarks, pad=20):
        """Crop region from frame using landmark bounding box + padding."""
        if landmarks is None:
            return None
        pts = np.array([landmarks[i] for i in indices if i < len(landmarks)])
        if len(pts) == 0:
            return None
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _preprocess(crop):
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype("float32")
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    def predict(self, frame, landmarks):
        return (
            self._predict_eye(frame, landmarks),
            self._predict_yawn(frame, landmarks),
        )

    def _predict_eye(self, frame, landmarks):
        if not self.eye_available or landmarks is None:
            return None
        probs = []
        for idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
            # pad=20 gives more context around the eye → better accuracy
            crop = self._crop(frame, idx, landmarks, pad=20)
            if crop is None or crop.size == 0:
                continue
            raw = float(self._eye_model.predict(
                self._preprocess(crop), verbose=0)[0][0])
            p_closed = 1.0 - raw
            probs.append(p_closed)
        return round(sum(probs) / len(probs), 4) if probs else None

    def _predict_yawn(self, frame, landmarks):
        if not self.yawn_available or landmarks is None:
            return None
        crop = self._crop(frame, MOUTH_IDX, landmarks, pad=20)
        if crop is None or crop.size == 0:
            return None
        raw = float(self._yawn_model.predict(
            self._preprocess(crop), verbose=0)[0][0])
        return round(raw, 4)