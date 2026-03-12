import os
import subprocess
import tempfile
from googletrans import Translator
from gtts import gTTS

INPUT_FILE = "english_sentence.txt"
OUTPUT_FILE = "hindi_tts.txt"   # ✅ ADD THIS

if not os.path.exists(INPUT_FILE):
    exit(0)

with open(INPUT_FILE) as f:
    english_text = f.read().strip()

if not english_text:
    exit(0)

translator = Translator()
translation = translator.translate(english_text, src="en", dest="hi")
hindi_text = translation.text

print("🔊 Speaking Hindi:", hindi_text)
with open("final_sentence.txt", "w", encoding="utf-8") as f:
    f.write(hindi_text)

# ✅ WRITE HINDI FOR WEBCAM
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(hindi_text)

tts = gTTS(text=hindi_text, lang="hi", slow=False)

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
    audio_file = tmp.name
    tts.save(audio_file)

subprocess.run(["afplay", audio_file])
os.unlink(audio_file)

