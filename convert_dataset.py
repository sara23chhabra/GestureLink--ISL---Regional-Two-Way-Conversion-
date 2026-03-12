import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# CHANGE THIS PATH to your dataset folder
DATASET_PATH = "../ISL_CSLRT_Dataset/Frames_Word_Level"
OUTPUT_PATH = "data"

os.makedirs(OUTPUT_PATH, exist_ok=True)

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    save_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(save_dir, exist_ok=True)

    count = 0

    for file in os.listdir(label_path):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(label_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if not result.hand_landmarks:
            continue

        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])

        np.save(
            f"{save_dir}/{count}.npy",
            np.array(landmarks)
        )
        count += 1

    print(f"{label}: {count} samples saved")
