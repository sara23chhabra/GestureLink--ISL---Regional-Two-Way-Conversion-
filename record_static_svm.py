import cv2
import os
import time

# =========================
# CONFIG
# =========================
SIGN_NAME = "I_Me"   # <<< CHANGE THIS
SAVE_DIR = f"static_svm/raw_videos/{SIGN_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

RECORD_DURATION = 2.0   # seconds
FPS = 20
COUNTDOWN = 2           # seconds before recording

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = len(os.listdir(SAVE_DIR))

print(f"🎥 Recording STATIC sign: {SIGN_NAME}")
print("Instructions:")
print("- Move hands into FINAL pose first")
print("- Keep hands steady")
print("- Press 'r' and HOLD")
print("- Recording starts after countdown")
print("- Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"Sign: {SIGN_NAME}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Hold final pose → Press 'r'",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Static SVM Recorder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        # COUNTDOWN
        for i in range(COUNTDOWN, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"Recording in {i}",
                (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )
            cv2.imshow("Static SVM Recorder", frame)
            cv2.waitKey(1000)

        # START RECORDING
        filename = f"{SIGN_NAME}_{count:03d}.avi"
        path = os.path.join(SAVE_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(path, fourcc, FPS, (640, 480))

        print(f"🔴 Recording {filename}...")
        start = time.time()

        while time.time() - start < RECORD_DURATION:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            out.write(frame)
            cv2.imshow("Static SVM Recorder", frame)
            cv2.waitKey(1)

        out.release()
        print("✅ Saved:", filename)
        count += 1
        time.sleep(0.5)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

