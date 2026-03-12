import re
import os
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

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
    "pain": "pain"
}

# ---------------- ENGLISH → ISL WORDS ----------------
def english_to_isl_words(sentence: str):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    words = sentence.split()

    isl_words = []
    for word in words:
        if word in ALLOWED_ISL_WORDS:
            isl_words.append(ALLOWED_ISL_WORDS[word])

    return isl_words


# ---------------- ISL VIDEO GENERATION + DISPLAY ----------------
def generate_and_play_isl(english_text: str, hindi_text: str = ""):
    isl_words = english_to_isl_words(english_text)

    if not isl_words:
        print("No valid ISL words found.")
        return

    video_dir = "isl_videos"
    output_path = "output_isl_sentence.mp4"
    fps = 25

    # Validate video files
    video_paths = []
    for word in isl_words:
        path = os.path.join(video_dir, f"{word}.mp4")
        if not os.path.exists(path):
            print(f"Missing ISL video for word: {word}")
            return
        video_paths.append(path)

    # Read video properties from first clip
    cap = cv2.VideoCapture(video_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Write concatenated video
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
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

    # ---------------- PLAY WITH TEXT OVERLAY ----------------
    cap = cv2.VideoCapture(output_path)

    # Load Devanagari font
    font_path = "fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
    hindi_font = ImageFont.truetype(font_path, 28)
    english_font = ImageFont.truetype(font_path, 26)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV frame to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Background rectangle
        draw.rectangle([(0, 0), (pil_img.width, 100)], fill=(0, 0, 0))

        # Hindi text (Unicode-safe)
        if hindi_text:
            draw.text(
                (10, 10),
                f"Hindi: {hindi_text}",
                font=hindi_font,
                fill=(255, 255, 255)
            )

        # English text
        draw.text(
            (10, 50),
            f"English: {english_text}",
            font=english_font,
            fill=(255, 255, 255)
        )

        # Convert back to OpenCV
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        cv2.imshow("ISL Output", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



