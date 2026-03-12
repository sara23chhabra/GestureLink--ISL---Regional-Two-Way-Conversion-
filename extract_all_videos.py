import os
import cv2
import mediapipe as mp
import numpy as np

DATASET_DIR = "webcam_dataset"
OUTPUT_DIR = "landmarks_webcam"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    output_label_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label_path, exist_ok=True)

    print(f"\nProcessing class: {label}")

    for video_file in os.listdir(label_path):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(label_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open {video_file}")
            continue

        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # 🔑 ALWAYS CREATE FIXED-SIZE FRAME
            frame_landmarks = np.zeros(126, dtype=np.float32)

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):
                    label_name = handedness.classification[0].label  # "Left" or "Right"

                    lm = []
                    for point in hand_landmarks.landmark:
                        lm.extend([point.x, point.y, point.z])

                    lm = np.array(lm, dtype=np.float32)

                    if label_name == "Left":
                        frame_landmarks[:63] = lm
                    else:  # Right
                        frame_landmarks[63:] = lm

            sequence.append(frame_landmarks)

        cap.release()

        if len(sequence) < 10:
            print(f"⚠️ Skipped {video_file} (too few frames)")
            continue

        sequence = np.array(sequence)
        save_path = os.path.join(
            output_label_path,
            video_file.replace(".avi", ".npy")
        )

        np.save(save_path, sequence)
        print(f"✅ Saved {video_file} | Shape: {sequence.shape}")

hands.close()
print("\n🎉 Two-hand landmark extraction complete!")


