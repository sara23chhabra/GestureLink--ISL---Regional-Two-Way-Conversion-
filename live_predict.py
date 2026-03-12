import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# LOAD TRAINED MODEL
# =========================
model = joblib.load("isl_gesture_model.pkl")
label_map = joblib.load("label_map.pkl")
inv_label_map = {v: k for k, v in label_map.items()}

# =========================
# MEDIAPIPE HAND LANDMARKER
# =========================
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# =========================
# WEBCAM + SMOOTHING BUFFER
# =========================
cap = cv2.VideoCapture(0)
prediction_buffer = deque(maxlen=10)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    prediction_text = "No Hand"

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        landmarks = []

        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(1, -1)

        probs = model.predict_proba(landmarks)[0]
        pred_index = np.argmax(probs)
        confidence = probs[pred_index]

        # ===== TEMPORAL SMOOTHING =====
        if confidence > 0.4:
            prediction_buffer.append(pred_index)

        if len(prediction_buffer) > 0:
            final_pred = max(
                set(prediction_buffer),
                key=prediction_buffer.count
            )
            prediction_text = inv_label_map[final_pred]
        else:
            prediction_text = "Uncertain"

        # Draw landmarks
        for lm in hand_landmarks:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # =========================
    # DISPLAY TEXT (CLEAR)
    # =========================
    cv2.rectangle(frame, (10, 5), (450, 60), (0, 0, 0), -1)

    cv2.putText(
        frame,
        prediction_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    cv2.imshow("ISL Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

