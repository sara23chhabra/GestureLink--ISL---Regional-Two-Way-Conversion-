import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from collections import deque, Counter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

IDLE_MOTION_THRESH = 0.005   # very small, fast


# =========================
# CONFIG (FAST MODE)
# =========================
SEQ_LENGTH = 40
FEATURES = 126
CONF_THRESH = 0.65
VOTE_WINDOW = 6
FREEZE_TIME = 1.2   # fast but visible

# =========================
# LOAD LABEL MAP (6 SIGNS)
# =========================
with open("lstm_data_webcam/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

inv_label_map = {v: k.replace("_Raw", "") for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

# =========================
# BUILD MODEL
# =========================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, FEATURES)),
    Dropout(0.3),
    LSTM(64),
    Dense(NUM_CLASSES, activation="softmax")
])

model.load_weights("isl_lstm.weights.h5")
print("✅ LSTM weights loaded")

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# UI
# =========================
def draw_main_text(frame, text):
    cv2.rectangle(frame, (10, 10), (630, 85), (0, 0, 0), -1)
    cv2.putText(frame, text, (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (0, 255, 0), 3, cv2.LINE_AA)

def draw_wait_text(frame):
    cv2.rectangle(frame, (10, 10), (630, 55), (0, 0, 0), -1)
    cv2.putText(frame, "Please wait for recognition...",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA)

# =========================
# STATE
# =========================
sequence = deque(maxlen=SEQ_LENGTH)
pred_buffer = deque(maxlen=VOTE_WINDOW)

frozen_word = None
freeze_start = 0

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started (press q to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_landmarks = np.zeros(FEATURES, dtype=np.float32)

    motion = 0.0
    if len(sequence) > 0:
        motion = np.mean(np.abs(frame_landmarks - sequence[-1]))


    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            lm = []
            for pt in hand_lms.landmark:
                lm.extend([pt.x, pt.y, pt.z])
            lm = np.array(lm, dtype=np.float32)

            if handedness.classification[0].label == "Left":
                frame_landmarks[:63] = lm
            else:
                frame_landmarks[63:] = lm

            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS
            )

    sequence.append(frame_landmarks)
    now = time.time()

    # =========================
    # FREEZE MODE (TOP PRIORITY)
    # =========================
    if frozen_word is not None:
        if now - freeze_start <= FREEZE_TIME:
            draw_main_text(frame, frozen_word)
        else:
            frozen_word = None
            pred_buffer.clear()
            sequence.clear()

        cv2.imshow("ISL LSTM Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # =========================
    # NORMAL PREDICTION
    # =========================
    if len(sequence) < SEQ_LENGTH or motion < IDLE_MOTION_THRESH:
        draw_wait_text(frame)

    else:
        x = np.expand_dims(np.array(sequence), axis=0)
        preds = model.predict(x, verbose=0)

        class_id = np.argmax(preds)
        confidence = preds[0][class_id]

        if confidence > CONF_THRESH:
            pred_buffer.append(class_id)

            if len(pred_buffer) == VOTE_WINDOW:
                final_class = Counter(pred_buffer).most_common(1)[0][0]
                frozen_word = inv_label_map[final_class].upper()
                freeze_start = now
        else:
            draw_wait_text(frame)

    cv2.imshow("ISL LSTM Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
