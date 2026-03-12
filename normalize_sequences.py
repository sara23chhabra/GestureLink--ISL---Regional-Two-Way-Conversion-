import os
import numpy as np

INPUT_DIR = "landmarks_webcam"
OUTPUT_DIR = "normalized_landmarks_webcam"

FIXED_LENGTH = 40
FEATURES = 126

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_sequence(seq, target_len=40):
    T, D = seq.shape

    # Case 1: exact length
    if T == target_len:
        return seq

    # Case 2: longer → uniform temporal sampling
    if T > target_len:
        indices = np.linspace(0, T - 1, target_len).astype(int)
        return seq[indices]

    # Case 3: shorter → pad last frame
    pad_len = target_len - T
    pad = np.repeat(seq[-1][np.newaxis, :], pad_len, axis=0)
    return np.vstack((seq, pad))


print("\n📏 Normalizing landmark sequences...")

for label in os.listdir(INPUT_DIR):
    label_path = os.path.join(INPUT_DIR, label)
    if not os.path.isdir(label_path):
        continue

    out_label_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_label_path, exist_ok=True)

    print(f"\nClass: {label}")

    for file in os.listdir(label_path):
        if not file.endswith(".npy"):
            continue

        seq = np.load(os.path.join(label_path, file))

        # 🔑 ONLY VALIDITY CHECK
        if seq.shape[1] != FEATURES:
            print(f"⚠️ Skipping {file}, wrong feature size: {seq.shape}")
            continue

        norm_seq = normalize_sequence(seq, FIXED_LENGTH)
        np.save(os.path.join(out_label_path, file), norm_seq)

        print(f"✅ {file}: {seq.shape} → {norm_seq.shape}")

print("\n🎉 ALL SEQUENCES NORMALIZED")

