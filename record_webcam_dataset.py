import cv2
import os
import time

# =========================
# CONFIG
# =========================
DATASET_DIR = "webcam_dataset"
SIGN_NAME = "thief"   # ← CHANGE THIS FOR EACH SIGN
NUM_VIDEOS = 15
FPS = 20
DURATION = 4  # seconds per video

# ✅ FIX 1: correct folder creation
sign_dir = os.path.join(DATASET_DIR, SIGN_NAME)
os.makedirs(sign_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

print(f"\n📹 Recording sign: {SIGN_NAME.upper()}")
print("Press 'r' to start recording")
print("Press 'q' to quit")

video_count = 0

while cap.isOpened() and video_count < NUM_VIDEOS:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    cv2.putText(
        display,
        f"Sign: {SIGN_NAME.upper()}  |  Video {video_count+1}/{NUM_VIDEOS}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "Press 'r' to record",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Recording", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        filename = f"{SIGN_NAME}_{video_count+1:03}.avi"
        filepath = os.path.join(sign_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            filepath,
            fourcc,
            FPS,
            (frame.shape[1], frame.shape[0])
        )

        print(f"▶ Recording {filename}...")
        start_time = time.time()

        # ✅ FIX 2: stable recording loop
        while time.time() - start_time < DURATION:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            out.write(frame)

            cv2.putText(
                frame,
                "RECORDING...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow("Recording", frame)
            cv2.waitKey(1)

        out.release()
        video_count += 1
        print(f"✅ Saved {filename}")
        time.sleep(1)  # ✅ small pause between recordings

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Recording complete.")

