import os
import numpy as np
import pickle

# =========================
# CONFIG
# =========================
INPUT_DIR = "normalized_landmarks_webcam"
OUTPUT_DIR = "lstm_data_webcam"

SEQ_LENGTH = 40
FEATURES = 126

os.makedirs(OUTPUT_DIR, exist_ok=True)

X = []
y = []
label_map = {}
label_index = 0

print("\n📦 Building LSTM dataset (two-hand landmarks)...")

# =========================
# LOAD DATA
# =========================
for label in sorted(os.listdir(INPUT_DIR)):
    label_path = os.path.join(INPUT_DIR, label)

    if not os.path.isdir(label_path):
        continue

    label_map[label] = label_index
    print(f"Class '{label}' → Label {label_index}")
    label_index += 1

    for file in os.listdir(label_path):
        if not file.endswith(".npy"):
            continue

        seq_path = os.path.join(label_path, file)
        seq = np.load(seq_path)

        # ✅ CORRECT SHAPE CHECK
        if seq.shape != (SEQ_LENGTH, FEATURES):
            print(f"⚠️ Skipping {file}, wrong shape: {seq.shape}")
            continue

        X.append(seq)
        y.append(label_map[label])

# =========================
# FINALIZE
# =========================
X = np.array(X)
y = np.array(y)

print("\n✅ Dataset created successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

with open(os.path.join(OUTPUT_DIR, "label_map.pkl"), "wb") as f:
    pickle.dump(label_map, f)

print("\n💾 Saved:")
print(" - lstm_data/X.npy")
print(" - lstm_data/y.npy")
print(" - lstm_data/label_map.pkl")
print("\n🎉 LSTM DATA PREPARATION COMPLETE (two-hand)")

