import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sounddevice as sd
import soundfile as sf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sentence_map import ISL_TO_HI
from PIL import Image, ImageDraw, ImageFont
import subprocess


HINDI_FONT = ImageFont.truetype(
    "fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf",
    28
)


# ======================================================
# CLEAR OLD FILES ON STARTUP
# ======================================================


for f in ["isl_words.txt", "final_sentence.txt", "english_sentence.txt"]:
    if os.path.exists(f):
        open(f, "w").close()


# ======================================================
# CONFIG
# ======================================================
SEQ_LENGTH = 40
FEATURES = 126
STATIC_FRAMES = 3
SVM_CONF_THRESH = 0.7
LSTM_CONF_THRESH = 0.55
VOTE_WINDOW = 6
SENTENCE_BAR_HEIGHT = 140

# ======================================================
# LOAD STATIC SVM
# ======================================================
with open("static_svm/svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

with open("static_svm/label_map.pkl", "rb") as f:
    svm_label_map = pickle.load(f)

svm_inv = {v: k for k, v in svm_label_map.items()}

# ======================================================
# LOAD LSTM
# ======================================================
with open("lstm_data_webcam/label_map.pkl", "rb") as f:
    lstm_label_map = pickle.load(f)

lstm_inv = {v: k.replace("_Raw", "") for k, v in lstm_label_map.items()}
NUM_CLASSES = len(lstm_label_map)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, FEATURES)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])
model.load_weights("isl_lstm.weights.h5")

print("✅ Vision models loaded")

# ======================================================
# MEDIAPIPE
# ======================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ======================================================
# STATE
# ======================================================
static_buffer = deque(maxlen=STATIC_FRAMES)
sequence = deque(maxlen=SEQ_LENGTH)
pred_buffer = deque(maxlen=VOTE_WINDOW)

sentence_buffer = []
frozen_word = None
mode = None  # STATIC / DYNAMIC

english_sentence = ""
hindi_sentence = ""
show_hindi = False


# ======================================================
# HELPERS
# ======================================================
def draw_text(frame, text, y, size=1.0, color=(255,255,255)):
    cv2.putText(frame, text, (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 2, cv2.LINE_AA)


def normalize_isl_words(words):
    words = set(words)
    if ("i" in words) or ("me" in words) or ("you" in words) or ("i_me" in words):
        words.discard("i")
        words.discard("me")
        words.discard("you")
        words.discard("i_me")
    return frozenset(words)



def play_audio_for_key(key):
    filename = "audio/" + "_".join(sorted(key)) + ".wav"
    if os.path.exists(filename):
        data, fs = sf.read(filename)
        sd.play(data, fs)
    else:
        print("⚠ Missing audio:", filename)


def read_english_sentence():
    try:
        with open("english_sentence.txt") as f:
            return f.read().strip()
    except:
        return ""

def draw_hindi_text(frame, text, x, y):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=HINDI_FONT, fill=(255, 255, 0))
    return np.array(img_pil)

def wrap_hindi_text(text, max_chars=22):
    words = text.split(" ")
    lines = []
    current = ""

    for w in words:
        if len(current) + len(w) <= max_chars:
            current += w + " "
        else:
            lines.append(current.strip())
            current = w + " "
    if current:
        lines.append(current.strip())

    return lines

def reset_sentence_state():
    sentence_buffer.clear()
    static_buffer.clear()
    sequence.clear()
    pred_buffer.clear()
    global frozen_word, mode
    frozen_word = None
    mode = None

def normalize_isl_for_hindi(words):
    words = set(words)
    for w in ["i", "me", "you", "i_me", "self"]:
        words.discard(w)
    return frozenset(words)

def read_hindi_tts():
    try:
        with open("hindi_tts.txt", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return ""


# ======================================================
# WEBCAM
# ======================================================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started — press q ONLY to quit")

sentence_buffer.clear()
static_buffer.clear()
sequence.clear()
pred_buffer.clear()
frozen_word = None
mode = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # Header
    cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)

    # Footer
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h-SENTENCE_BAR_HEIGHT), (w, h), (0, 0, 0), -1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks = np.zeros(FEATURES, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handed in zip(results.multi_hand_landmarks,
                                    results.multi_handedness):
            lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark]).flatten()
            if handed.classification[0].label == "Left":
                landmarks[:63] = lm
            else:
                landmarks[63:] = lm
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1) & 0xFF

    # ==================================================
    # CONTROLS
    # ==================================================
    if key == ord('s'):
        mode = "STATIC"
        static_buffer.clear()
        sequence.clear()
        pred_buffer.clear()

    elif key == ord('d'):
        mode = "DYNAMIC"
        static_buffer.clear()
        sequence.clear()
        pred_buffer.clear()

    elif key == 13 and frozen_word:
        sentence_buffer.append(frozen_word.lower())  # KEEP i_me
        frozen_word = None
        mode = None

    elif key in [8, 127]:  # BACKSPACE
        frozen_word = None
        mode = None

    elif key == ord('g'):
        if sentence_buffer:
            with open("isl_words.txt", "w") as f:
                f.write(" ".join(sentence_buffer))

            print("✅ Sentence committed (English generation triggered):", sentence_buffer)

            show_hindi = False

    elif key == ord('c'):
        print("🧹 Clearing sentence and buffers")
        show_hindi = False
        english_sentence = ""
        hindi_sentence = ""
        reset_sentence_state()

        # Optional: also clear files so UI resets
        for f in ["isl_words.txt", "english_sentence.txt", "final_sentence.txt"]:
            if os.path.exists(f):
                open(f, "w").close()


    elif key == ord('h'):
        if not sentence_buffer:
            print("⚠ No active sentence to speak")
        else:
            norm_key = normalize_isl_words(sentence_buffer)
            if norm_key in ISL_TO_HI:
                hindi_sentence = ISL_TO_HI[norm_key]
                play_audio_for_key(norm_key)
                show_hindi = True


    elif key == ord('r'):
        english_sentence = read_english_sentence()

        if english_sentence:
            print("🔁 Triggering Hindi TTS")

            subprocess.Popen([
                os.path.join("venv_tts", "bin", "python"),
                "english_to_hindi_tts_once.py"
            ])

            # Do NOT read hindi here (TTS is async)
            show_hindi = True



    elif key == ord('q'):
        break

    # ==================================================
    # RECOGNITION
    # ==================================================
    if frozen_word:
        draw_text(frame, frozen_word.upper(), 70, 1.4)

    elif mode == "STATIC":
        static_buffer.append(landmarks)
        if len(static_buffer) == STATIC_FRAMES:
            pose = np.mean(static_buffer, axis=0).reshape(1, -1)
            probs = svm.predict_proba(pose)[0]
            cid = np.argmax(probs)
            if probs[cid] >= SVM_CONF_THRESH:
                frozen_word = svm_inv[cid]

    elif mode == "DYNAMIC":
        sequence.append(landmarks)
        if len(sequence) >= SEQ_LENGTH:
            x = np.expand_dims(sequence, axis=0)
            preds = model.predict(x, verbose=0)
            cid = np.argmax(preds)
            if preds[0][cid] >= LSTM_CONF_THRESH:
                pred_buffer.append(cid)
                if len(pred_buffer) == VOTE_WINDOW:
                    frozen_word = lstm_inv[cid]
                    pred_buffer.clear()

    # ==================================================
    # DISPLAY OUTPUT
    # ==================================================
    english_sentence = read_english_sentence()
    FOOTER_TOP = h - SENTENCE_BAR_HEIGHT

    # English
    if english_sentence:
        draw_text(frame, "English:", FOOTER_TOP + 30, 0.7, (0, 255, 0))
        draw_text(frame, english_sentence, FOOTER_TOP + 55, 0.9, (0, 255, 0))

    # Hindi
    # Hindi (mapping OR TTS)
    if show_hindi:
        hindi_sentence = read_hindi_tts()
        if hindi_sentence:
            frame = draw_hindi_text(frame, "Hindi:", 25, FOOTER_TOP + 85)
            frame = draw_hindi_text(frame, hindi_sentence, 25, FOOTER_TOP + 115)



    '''if show_hindi and hindi_sentence:
        lines = wrap_hindi_text(hindi_sentence)

        y = h - 95
        frame = draw_hindi_text(frame, "Hindi:", 25, y)
        y += 30

        for line in lines:
            frame = draw_hindi_text(frame, line, 25, y)
            y += 32'''



    cv2.imshow("ISL Demo", frame)

cap.release()
cv2.destroyAllWindows()
hands.close()
