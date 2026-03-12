import cv2
import os

# ---------------- CONFIG ----------------
VIDEO_DIR = "isl_videos"
FPS = 25
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Ensure folder exists
os.makedirs(VIDEO_DIR, exist_ok=True)

# ---------------- RECORD FUNCTION ----------------
def record_isl_word(word):
    output_path = os.path.join(VIDEO_DIR, f"{word}.mp4")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
    )

    recording = False
    print(f"\nRecording ISL word: '{word}'")
    print("Press 'r' to START recording")
    print("Press 's' to STOP & save")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        status = "RECORDING" if recording else "PREVIEW"
        color = (0, 0, 255) if recording else (0, 255, 0)

        cv2.putText(
            display, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        cv2.imshow("ISL Recorder", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            recording = True

        elif key == ord('s'):
            break

        elif key == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return

        if recording:
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved: {output_path}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    while True:
        word = input("\nEnter ISL word to record (or 'exit'): ").strip().lower()
        if word == "exit":
            break
        record_isl_word(word)
