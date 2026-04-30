import cv2
import os
import time
import numpy as np

from utils.landmarks     import FaceLandmarkDetector
from utils.ear_mar       import RatioSmoother, compute_ratios, EAR_THRESHOLD, MAR_THRESHOLD
from utils.logic         import DrowsinessLogic, AlertLevel
from utils.alert         import AlertSystem
from utils.model_predict import DrowsinessModel

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

LEVEL_COLOUR = {
    AlertLevel.NORMAL:     (0, 200,  80),
    AlertLevel.YAWNING:    (0, 165, 255),
    AlertLevel.DROWSY:     (30, 80, 255),
    AlertLevel.HIGH_ALERT: (0,   0, 255),
}


def draw_hud(frame, signals, level, fps):
    h, w = frame.shape[:2]
    col  = LEVEL_COLOUR.get(level, (200, 200, 200))

    # Dark background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 200), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    def txt(text, y, colour=(210, 210, 210)):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    colour, 1, cv2.LINE_AA)

    ear = signals.get("ear") or 0.0
    mar = signals.get("mar") or 0.0

    txt(f"FPS : {fps:>4.1f}", 28)

    # EAR
    ear_ok = ear >= EAR_THRESHOLD
    txt(f"EAR : {ear:.3f}  {'OK' if ear_ok else 'CLOSED'}",
        58, (80, 220, 80) if ear_ok else (60, 60, 255))

    # MAR
    mar_ok = mar <= MAR_THRESHOLD
    txt(f"MAR : {mar:.3f}  {'OK' if mar_ok else 'YAWN'}",
        88, (80, 220, 80) if mar_ok else (0, 165, 255))

    # CNN info (display only — not used for alerts)
    eye_prob  = signals.get("eye_prob")
    yawn_prob = signals.get("yawn_prob")

    if eye_prob is not None:
        txt(f"Eye CNN  : {eye_prob:.3f}", 125, (160, 160, 160))
    if yawn_prob is not None:
        txt(f"Yawn CNN : {yawn_prob:.3f}", 153, (160, 160, 160))

    # Alert badge
    cv2.rectangle(frame, (0, h - 52), (w, h), col, -1)
    label = level.value
    lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
    cv2.putText(frame, label, ((w - lw) // 2, h - 13),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    print("\n[test_camera] Starting...")

    detector     = FaceLandmarkDetector()
    ear_smoother = RatioSmoother()
    mar_smoother = RatioSmoother()
    logic        = DrowsinessLogic()
    alert        = AlertSystem()
    dl_model     = DrowsinessModel()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAM_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    os.makedirs("debug", exist_ok=True)

    print("[test_camera] Running — Q=quit  R=reset  S=screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        landmarks, display = detector.detect(frame)

        # Compute EAR + MAR
        left_eye, right_eye = detector.get_eye_points(landmarks)
        mouth               = detector.get_mouth_points(landmarks)
        ratios              = compute_ratios(left_eye, right_eye, mouth)

        if len(ratios) == 3:
            ear_raw, mar_raw, _ = ratios
        else:
            ear_raw, mar_raw = ratios

        if ear_raw is not None:
            ear = ear_smoother.update(ear_raw)
            mar = mar_smoother.update(mar_raw)
        else:
            ear = mar = 0.0

        eye_prob, yawn_prob = dl_model.predict(frame, landmarks)

        level, signals = logic.update(
            ear       = ear       if landmarks else None,
            mar       = mar       if landmarks else None,
            eye_prob  = eye_prob,
            yawn_prob = yawn_prob,
        )

        alert.update(level in (AlertLevel.DROWSY, AlertLevel.HIGH_ALERT, AlertLevel.YAWNING))

        fps = logic.tick()

        if landmarks is None:
            cv2.putText(display, "No face detected",
                        (FRAME_W // 2 - 100, FRAME_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)

        draw_hud(display, signals, level, fps)
        cv2.imshow("Drowsiness Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            logic.reset()
            ear_smoother.reset()
            mar_smoother.reset()
            print("[test_camera] Reset.")
        elif key == ord('s'):
            f = f"debug/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(f, display)
            print(f"[test_camera] Saved: {f}")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("[test_camera] Done.")


if __name__ == "__main__":
    main()