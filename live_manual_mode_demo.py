import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ======================================================
# CONFIG
# ======================================================
SEQ_LENGTH = 40
FEATURES = 126

STATIC_FRAMES = 3
SVM_CONF_THRESH = 0.7

LSTM_CONF_THRESH = 0.55
VOTE_WINDOW = 6
sentence_buffer = []

# ======================================================
# LOAD STATIC SVM
# ======================================================
with open("static_svm/svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

with open("static_svm/label_map.pkl", "rb") as f:
    svm_label_map = pickle.load(f)

svm_inv = {v: k for k, v in svm_label_map.items()}

# ======================================================
# LOAD LSTM
# ======================================================
with open("lstm_data_webcam/label_map.pkl", "rb") as f:
    lstm_label_map = pickle.load(f)

lstm_inv = {v: k.replace("_Raw", "") for k, v in lstm_label_map.items()}
NUM_CLASSES = len(lstm_label_map)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, FEATURES)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])
model.load_weights("isl_lstm.weights.h5")

print("✅ Models loaded (Manual demo mode)")

# ======================================================
# MEDIAPIPE
# ======================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ======================================================
# BUFFERS & STATE
# ======================================================
static_buffer = deque(maxlen=STATIC_FRAMES)
sequence = deque(maxlen=SEQ_LENGTH)
pred_buffer = deque(maxlen=VOTE_WINDOW)

frozen_word = None
mode = None   # "STATIC" or "DYNAMIC"

# ======================================================
# UI
# ======================================================
def draw_text(frame, text, size=1.2, color=(0,255,0)):
    cv2.rectangle(frame, (10,10), (630,80), (0,0,0), -1)
    cv2.putText(frame, text, (25,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 3, cv2.LINE_AA)

# ======================================================
# WEBCAM
# ======================================================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started (silent manual mode)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks = np.zeros(FEATURES, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handed in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            lm = []
            for pt in hand_lms.landmark:
                lm.extend([pt.x, pt.y, pt.z])
            lm = np.array(lm, dtype=np.float32)

            if handed.classification[0].label == "Left":
                landmarks[:63] = lm
            else:
                landmarks[63:] = lm

            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS
            )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------
    # KEY CONTROLS (HIDDEN)
    # --------------------------------------------------
    if key == ord('s'):
        mode = "STATIC"
        static_buffer.clear()
        sequence.clear()
        pred_buffer.clear()

    elif key == ord('d'):
        mode = "DYNAMIC"
        static_buffer.clear()
        sequence.clear()
        pred_buffer.clear()

    elif key == 13 and frozen_word is not None:  # ENTER
        frozen_word = None
        mode = None
        static_buffer.clear()
        sequence.clear()
        pred_buffer.clear()

    elif key == ord('q'):
        break

    # --------------------------------------------------
    # SHOW RESULT IF FROZEN
    # --------------------------------------------------
    if frozen_word is not None:
        sentence_buffer.append(frozen_word.lower())
        draw_text(frame, frozen_word.upper())
        cv2.imshow("ISL Demo", frame)
        continue

    # --------------------------------------------------
    # WAITING STATE
    # --------------------------------------------------
    if mode is None:
        draw_text(frame, "Please wait for recognition...",
                  size=0.8, color=(0,255,255))
        cv2.imshow("ISL Demo", frame)
        continue

    # --------------------------------------------------
    # STATIC MODE (SVM)
    # --------------------------------------------------
    if mode == "STATIC":
        static_buffer.append(landmarks)

        if len(static_buffer) == STATIC_FRAMES:
            pose = np.mean(static_buffer, axis=0).reshape(1, -1)
            probs = svm.predict_proba(pose)[0]
            cid = np.argmax(probs)

            if probs[cid] >= SVM_CONF_THRESH:
                frozen_word = svm_inv[cid]

    # --------------------------------------------------
    # DYNAMIC MODE (LSTM)
    # --------------------------------------------------
    elif mode == "DYNAMIC":
        sequence.append(landmarks)

        if len(sequence) >= SEQ_LENGTH:
            x = np.expand_dims(np.array(sequence)[-SEQ_LENGTH:], axis=0)
            preds = model.predict(x, verbose=0)

            cid = np.argmax(preds)
            conf = preds[0][cid]

            if conf >= LSTM_CONF_THRESH:
                pred_buffer.append(cid)

                if len(pred_buffer) == VOTE_WINDOW:
                    frozen_word = lstm_inv[cid]
                    pred_buffer.clear()

    draw_text(frame, "Please wait for recognition...",
              size=0.8, color=(0,255,255))

    cv2.imshow("ISL Demo", frame)

cap.release()
cv2.destroyAllWindows()
hands.close()
