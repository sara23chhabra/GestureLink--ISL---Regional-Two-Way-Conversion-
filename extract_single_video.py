import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video_path = "/Users/sarachhabra/Documents/Sara/ISL_AI_Project/dataset/help_Raw/help_001_01.AVI"  # change if needed
if not os.path.exists(video_path):
    print("Video path does not exist:", video_path)
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        frame_landmarks = []

        for lm in hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])

        sequence.append(frame_landmarks)

cap.release()
hands.close()

sequence = np.array(sequence)

print("Sequence shape:", sequence.shape)
