import re
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================================================
# CONFIG
# ======================================================
ENGLISH_SENTENCE_FILE = "english_sentence.txt"
OUTPUT_VIDEO = "output_isl_sentence.mp4"
ISL_VIDEO_DIR = "isl_videos"
FONT_PATH = "fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"

FPS = 25

# ---------------- ISL VOCAB ----------------
ALLOWED_ISL_WORDS = {
    "i": "me",
    "me": "me",
    "you": "you",
    "there": "there",
    "here": "here",
    "please": "please",
    "stop": "stop",
    "accident": "accident",
    "doctor": "doctor",
    "help": "help",
    "call": "call",
    "thief": "thief",
    "pain": "pain",
}

# ======================================================
# ENGLISH → ISL WORDS
# ======================================================
def english_to_isl_words(sentence: str):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    words = sentence.split()

    isl_words = []
    for word in words:
        if word in ALLOWED_ISL_WORDS:
            isl_words.append(ALLOWED_ISL_WORDS[word])

    return isl_words

# ======================================================
# ISL VIDEO GENERATION
# ======================================================
def generate_and_play_isl(english_text: str, hindi_text: str = "", show_window=True):
    isl_words = english_to_isl_words(english_text)

    if not isl_words:
        print("❌ No valid ISL words found.")
        return False

    # Collect ISL video clips
    video_paths = []
    for word in isl_words:
        path = os.path.join(ISL_VIDEO_DIR, f"{word}.mp4")
        if not os.path.exists(path):
            print(f"❌ Missing ISL video for word: {word}")
            return False
        video_paths.append(path)

    # Read video size from first clip
    cap = cv2.VideoCapture(video_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Write concatenated video
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )

    for path in video_paths:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
        cap.release()

    out.release()
    print(f"✅ ISL video generated: {OUTPUT_VIDEO}")

    # --------------------------------------------------
    # OPTIONAL: PLAY WITH TEXT OVERLAY (terminal demo)
    # --------------------------------------------------
    if not show_window:
        return True

    try:
        hindi_font = ImageFont.truetype(FONT_PATH, 28)
        english_font = ImageFont.truetype(FONT_PATH, 26)
    except Exception as e:
        print("⚠ Font load failed:", e)
        return True

    cap = cv2.VideoCapture(OUTPUT_VIDEO)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Overlay background
        draw.rectangle([(0, 0), (pil_img.width, 100)], fill=(0, 0, 0))

        if hindi_text:
            draw.text(
                (10, 10),
                f"Hindi: {hindi_text}",
                font=hindi_font,
                fill=(255, 255, 255),
            )

        draw.text(
            (10, 50),
            f"English: {english_text}",
            font=english_font,
            fill=(255, 255, 255),
        )

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("ISL Output", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

# ======================================================
# FILE-DRIVEN ENTRY POINT (FOR STREAMLIT)
# ======================================================
def main():
    if not os.path.exists(ENGLISH_SENTENCE_FILE):
        print("❌ english_sentence.txt not found")
        return

    with open(ENGLISH_SENTENCE_FILE, "r", encoding="utf-8") as f:
        english_text = f.read().strip()

    if not english_text:
        print("❌ english_sentence.txt is empty")
        return

    # For Streamlit: generate video only, no OpenCV window
    generate_and_play_isl(
        english_text=english_text,
        hindi_text="",
        show_window=True,
    )



if __name__ == "__main__":
    main()

