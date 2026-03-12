import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# =========================
# CONFIG
# =========================
DATA_DIR = "static_svm/raw_videos"
OUT_DIR = "static_svm"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

X = []
y = []
label_map = {}

# =========================
# PROCESS EACH SIGN
# =========================
idx = 0

for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)

    # Skip files like .DS_Store
    if not os.path.isdir(label_path):
        continue

    label_map[label] = idx
    print(f"🔍 Processing static sign: {label} (class {idx})")

    for video in os.listdir(label_path):
        if not video.lower().endswith(".avi"):
            continue

        video_path = os.path.join(label_path, video)
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            lm_vec = np.zeros(126, dtype=np.float32)

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
                        lm_vec[:63] = lm
                    else:
                        lm_vec[63:] = lm

                frames.append(lm_vec)

        cap.release()

        if len(frames) >= 5:
            feature = np.mean(frames[-5:], axis=0)
            X.append(feature)
            y.append(idx)

    # 🔑 INCREMENT CLASS ID HERE
    idx += 1

hands.close()

# =========================
# SAVE DATASET
# =========================
X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

with open(os.path.join(OUT_DIR, "label_map.pkl"), "wb") as f:
    pickle.dump(label_map, f)

print("\n✅ Static landmark extraction complete")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", label_map)

