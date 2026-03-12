import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque, Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ======================================================
# CONFIG
# ======================================================
SEQ_LENGTH = 40
FEATURES = 126

STATIC_FRAMES = 3
SVM_CONF_THRESH = 0.75
POSE_STABILITY_THRESH = 0.008

LSTM_CONF_THRESH = 0.55
VOTE_WINDOW = 6

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

print("✅ Fusion models loaded")

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
waiting_for_confirm = False

# ======================================================
# HELPERS
# ======================================================
def is_pose_stable(buffer, thresh=POSE_STABILITY_THRESH):
    buf = np.array(buffer)
    diffs = np.linalg.norm(buf[1:] - buf[:-1], axis=1)
    return np.mean(diffs) < thresh

def draw_text(frame, text, y=60, size=1.2, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (25, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        3,
        cv2.LINE_AA
    )

# ======================================================
# WEBCAM
# ======================================================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started | Press ENTER to confirm | q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (630, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

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

    # ==================================================
    # CONFIRMATION MODE
    # ==================================================
    if frozen_word is not None:
        draw_text(frame, frozen_word.upper(), y=55, size=1.4)
        draw_text(frame, "Press ENTER for next sign", y=90,
                  size=0.7, color=(255, 255, 0))

        cv2.imshow("ISL Fusion", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            frozen_word = None
            static_buffer.clear()
            sequence.clear()
            pred_buffer.clear()
            continue
        elif key == ord('q'):
            break

        continue

    # ==================================================
    # UPDATE BUFFERS
    # ==================================================
    static_buffer.append(landmarks)
    sequence.append(landmarks)

    static_detected = False

    # ==================================================
    # 1️⃣ STATIC FIRST (SVM + STABILITY)
    # ==================================================
    if len(static_buffer) == STATIC_FRAMES and is_pose_stable(static_buffer):
        pose = np.mean(static_buffer, axis=0).reshape(1, -1)
        probs = svm.predict_proba(pose)[0]
        cid = np.argmax(probs)

        if probs[cid] >= SVM_CONF_THRESH:
            frozen_word = svm_inv[cid]
            static_detected = True

    # ==================================================
    # 2️⃣ DYNAMIC ONLY IF STATIC FAILED
    # ==================================================
    if not static_detected and len(sequence) >= SEQ_LENGTH:
        x_seq = list(sequence)[-SEQ_LENGTH:]
        x = np.expand_dims(np.array(x_seq), axis=0)
        preds = model.predict(x, verbose=0)

        cid = np.argmax(preds)
        conf = preds[0][cid]

        if conf >= LSTM_CONF_THRESH:
            pred_buffer.append(cid)

            if len(pred_buffer) == VOTE_WINDOW:
                final_cid = Counter(pred_buffer).most_common(1)[0][0]
                frozen_word = lstm_inv[final_cid]
                pred_buffer.clear()

    # ==================================================
    # DEFAULT UI
    # ==================================================
    draw_text(frame, "Please wait for recognition...",
              y=70, size=0.8, color=(0, 255, 255))

    cv2.imshow("ISL Fusion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()


