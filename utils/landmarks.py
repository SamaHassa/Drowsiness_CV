import cv2
import numpy as np
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat

MODEL_FILE = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(MODEL_FILE):
        print("[landmarks] Downloading face_landmarker.task (~5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[landmarks] ✓ Download complete.")

LEFT_EAR_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_POINTS = [33,  160, 158, 133, 153, 144]
MOUTH_MAR_POINTS = [61, 291, 0, 17, 78, 308, 13, 14]

LEFT_EYE_IDX  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_IDX = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]


class FaceLandmarkDetector:

    def __init__(self, max_faces=1,
                 min_detection_confidence=0.6,
                 min_tracking_confidence=0.6):
        _ensure_model()
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        print("[landmarks] FaceLandmarker ready.")

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        annotated = frame.copy()

        if not result.face_landmarks:
            return None, annotated

        face_lm = result.face_landmarks[0]
        landmarks = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_lm
        ]
        return landmarks, annotated

    def get_eye_points(self, landmarks):
        if landmarks is None:
            return None, None
        return ([landmarks[i] for i in LEFT_EAR_POINTS],
                [landmarks[i] for i in RIGHT_EAR_POINTS])

    def get_mouth_points(self, landmarks):
        if landmarks is None:
            return None
        return [landmarks[i] for i in MOUTH_MAR_POINTS]

    def close(self):
        self._landmarker.close()