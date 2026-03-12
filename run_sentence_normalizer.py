import time
import os
import ollama
from sentence_map import ISL_TO_HI

def normalize_sentence(isl_words):
    keywords = " ".join(isl_words)

    prompt = f"""
You are an Indian Sign Language (ISL) emergency sentence normalizer.

Your task is to convert ISL keywords into clear, natural English
used in emergency communication.

STRICT RULES (MANDATORY):
- Output ONLY one English sentence
- Do NOT explain anything
- Do NOT add new entities
- Do NOT change subject-object roles
- Assume the signer is the speaker ("I")
- Use simple emergency verbs ONLY when appropriate:
  - "need", "want", "require"
- If the word "help" appears, you MAY use "need help"
- If the word "doctor" appears with "help" or "pain", you MAY say "need a doctor"
- Do NOT invent causes, reasons, or emotions
- Keep the sentence short, direct, and urgent

ISL keywords:
{keywords}

Output:
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()


print("🧠 Sentence normalizer running (file-watcher mode)...")
print("Waiting for ISL gloss in isl_words.txt")

last_seen = ""

while True:
    try:
        if not os.path.exists("isl_words.txt"):
            time.sleep(0.5)
            continue

        with open("isl_words.txt") as f:
            data = f.read().strip().lower()

        if not data or data == last_seen:
            time.sleep(0.5)
            continue

        isl_words = data.split()

        # English from Ollama
        english = normalize_sentence(isl_words)

        '''# Hindi from ISL semantics
        key = frozenset(isl_words)
        hindi = ISL_TO_HI.get(key, "अनुवाद उपलब्ध नहीं है")'''

        # Write English for OpenCV + TTS
        with open("english_sentence.txt", "w", encoding="utf-8") as f:
            f.write(english)

        # (Optional) keep Hindi logging if you want
        '''with open("final_sentence.txt", "w", encoding="utf-8") as f:
            f.write(english + "\n")
            f.write(hindi + "\n")'''


        print("✔ ISL:", isl_words)
        print("✔ English:", english)
        

        last_seen = data
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Normalizer stopped safely.")
        break

    except Exception as e:
        print("⚠ NLP error:", e)
        time.sleep(1)
