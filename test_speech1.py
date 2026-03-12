import speech_recognition as sr
from googletrans import Translator

r = sr.Recognizer()
mic = sr.Microphone()
translator = Translator()

print("Speak now...")

with mic as source:
    r.adjust_for_ambient_noise(source, duration=1)
    audio = r.listen(source, phrase_time_limit=5)

text = r.recognize_google(audio, language="hi-IN")
english = translator.translate(text, src="hi", dest="en").text

print("Detected:", text)
print("English:", english)

with open("isl_words.txt", "w", encoding="utf-8") as f:
    f.write(text)

with open("english_sentence.txt", "w", encoding="utf-8") as f:
    f.write(english)

print("Files written.")
