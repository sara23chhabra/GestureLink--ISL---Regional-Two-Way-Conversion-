import speech_recognition as sr
import threading
import time
import os
from datetime import datetime
from googletrans import Translator

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from english_to_isl import generate_and_play_isl


# ======================================================
# LANGUAGE CONFIG (Variable fonts)
# ======================================================
LANGUAGE_CONFIG = {
    "hi-IN": {
        "name": "Hindi",
        "font": "fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
    },
    "ta-IN": {
        "name": "Tamil",
        "font": "fonts/NotoSansTamil-VariableFont_wdth,wght.ttf"
    },
    "te-IN": {
        "name": "Telugu",
        "font": "fonts/NotoSansTelugu-VariableFont_wdth,wght.ttf"
    },
    "kn-IN": {
        "name": "Kannada",
        "font": "fonts/NotoSansKannada-VariableFont_wdth,wght.ttf"
    },
    "ml-IN": {
        "name": "Malayalam",
        "font": "fonts/NotoSansMalayalam-VariableFont_wdth,wght.ttf"
    },
    "bn-IN": {
        "name": "Bengali",
        "font": "fonts/NotoSansBengali-VariableFont_wdth,wght.ttf"
    },
    "gu-IN": {
        "name": "Gujarati",
        "font": "fonts/NotoSansGujarati-VariableFont_wdth,wght.ttf"
    }
}

FONT_CACHE = {}


def get_font(lang_code, size=32):
    config = LANGUAGE_CONFIG.get(lang_code)
    if not config:
        return None

    font_path = config["font"]
    key = (font_path, size)

    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(font_path, size)

    return FONT_CACHE[key]


def draw_multilingual_text(frame, label, text, lang_code, x, y):
    font = get_font(lang_code)
    if font is None:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    draw.text((x, y), f"{label}:", font=font, fill=(255, 255, 0))
    draw.text((x, y + 40), text, font=font, fill=(255, 255, 0))

    return np.array(img)


# ======================================================
# SPEECH TO TEXT CLASS
# ======================================================
class RealTimeSpeechToText:
    def __init__(self, language="en-US"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language

        self.is_listening = False
        self.transcript = ""
        self.translated_transcript = ""
        self.interim_transcript = ""
        self.error_message = ""

        self.thread = None
        self.translator = Translator()

        print("Adjusting for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready.")

    def set_language(self, language):
        self.language = language
        print(f"Language set to {language}")

    def is_english(self):
        return self.language.startswith("en")

    def translate_text(self, text):
        if self.is_english():
            return text
        return self.translator.translate(
            text, src=self.language.split("-")[0], dest="en"
        ).text

    def clear_transcript(self):
        self.transcript = ""
        self.translated_transcript = ""
        self.interim_transcript = ""
        self.error_message = ""

    def start_listening(self):
        if self.is_listening:
            return
        self.clear_transcript()
        self.is_listening = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print("Listening...")

    def stop_listening(self):
        self.is_listening = False
        if self.thread:
            self.thread.join()
        print("Stopped.")

    def _listen_loop(self):
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source, timeout=5, phrase_time_limit=10
                    )

                text = self.recognizer.recognize_google(
                    audio, language=self.language
                )

                if text:
                    self.transcript += text + " "
                    if not self.is_english():
                        translated = self.translate_text(text)
                        self.translated_transcript += translated + " "
                        print(f"{text} → {translated}")
                    else:
                        print(text)

            except Exception:
                pass

            time.sleep(0.1)

    def get_transcript(self):
        return {
            "original": self.transcript.strip(),
            "translated": self.translated_transcript.strip()
            if not self.is_english()
            else self.transcript.strip(),
        }


# ======================================================
# MAIN
# ======================================================
LANGUAGE_OPTIONS = {
    "en-US": "English (US)",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
}


def main():
    stt = RealTimeSpeechToText()

    while True:
        print("\n1. Set language\n2. Start listening\n3. Stop & generate ISL\n4. Exit")
        choice = input("> ").strip()

        if choice == "1":
            for k, v in LANGUAGE_OPTIONS.items():
                print(f"{k}: {v}")
            lang = input("Language code: ").strip()
            if lang in LANGUAGE_OPTIONS:
                stt.set_language(lang)

        elif choice == "2":
            stt.start_listening()

        elif choice == "3":
            stt.stop_listening()
            data = stt.get_transcript()

            english_text = data["translated"]
            original_text = data["original"]

            if not english_text:
                continue

            print("\nEnglish:", english_text)

            # Generate ISL video
            generate_and_play_isl(
                english_text=english_text,
                hindi_text=original_text,
            )

            # Display window with multilingual text
            frame = np.zeros((300, 900, 3), dtype=np.uint8)

            cv2.putText(
                frame,
                "English:",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                english_text,
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            if not stt.is_english():
                cfg = LANGUAGE_CONFIG.get(stt.language)
                if cfg:
                    frame = draw_multilingual_text(
                        frame,
                        cfg["name"],
                        original_text,
                        stt.language,
                        25,
                        130,
                    )

            cv2.imshow("Pipeline 2 Output", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            stt.clear_transcript()

        elif choice == "4":
            stt.stop_listening()
            break


if __name__ == "__main__":
    main()
