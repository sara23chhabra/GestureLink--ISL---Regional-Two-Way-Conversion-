# ⚠️ Legacy normalizer — do not run with webcam pipeline

import time
import os
import ollama


def normalize_sentence(words):
    if not words:
        return ""

    # Join ISL keywords
    keywords = " ".join(words)

    # STRICT PROMPT — UNCHANGED
    prompt = f"""
You are a sentence normalization engine.

Rules (MANDATORY):
- Output ONLY one grammatically correct English sentence
- Do NOT say you are an AI
- Do NOT explain anything
- Do NOT add extra information
- Preserve the original meaning
- Keep the sentence short and direct

ISL keywords:
{keywords}

Output:
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip().split("\n")[0]


# ---------------- FILE WATCHER MAIN ----------------
print("🧠 English normalizer running (venv_nlp)")
last_seen = ""

while True:
    try:
        if not os.path.exists("isl_words.txt"):
            time.sleep(0.5)
            continue

        with open("isl_words.txt") as f:
            data = f.read().strip()

        if not data or data == last_seen:
            time.sleep(0.5)
            continue

        isl_words = data.split()
        sentence = normalize_sentence(isl_words)

        with open("english_sentence.txt", "w") as f:
            f.write(sentence)
        with open("english_for_tts.txt", "w") as f:
            f.write(sentence)


        print("✔ English:", sentence)
        last_seen = data

        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 English normalizer stopped safely.")
        break
