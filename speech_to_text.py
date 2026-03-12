import speech_recognition as sr
import threading
import time
import os
import signal
from googletrans import Translator

ISL_WORDS_FILE = "isl_words.txt"
ENGLISH_SENTENCE_FILE = "english_sentence.txt"


class RealTimeSpeechToText:
    def __init__(self, language="hi-IN"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language

        self.transcript = ""
        self.translator = Translator()
        self.running = True

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def is_english(self):
        return self.language.startswith("en")

    def translate_to_english(self, text):
        if self.is_english():
            return text
        return self.translator.translate(
            text, src=self.language.split("-")[0], dest="en"
        ).text

    def listen_loop(self):
        while self.running:
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

            except Exception:
                pass

            time.sleep(0.1)

    def write_files(self):
        original_text = self.transcript.strip()
        if not original_text:
            return

        english_text = self.translate_to_english(original_text)

        with open(ISL_WORDS_FILE, "w", encoding="utf-8") as f:
            f.write(original_text + "\n")

        with open(ENGLISH_SENTENCE_FILE, "w", encoding="utf-8") as f:
            f.write(english_text + "\n")


def main():
    language = os.environ.get("ISL_LANGUAGE", "hi-IN")
    stt = RealTimeSpeechToText(language)

    def shutdown_handler(signum, frame):
        stt.running = False
        stt.write_files()
        exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    t = threading.Thread(target=stt.listen_loop, daemon=True)
    t.start()

    # keep process alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
