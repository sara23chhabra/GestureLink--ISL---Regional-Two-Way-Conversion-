import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib

# =========================
# CONFIG
# =========================
DATA_PATH = "data"

SELECTED_LABELS = [
    "YOU",
    "I_ME_MINE_MY",
    "WHAT",
    "FOOD",
    "HELP",
    "WATER",
    "HOW",
    "THANK"
]

# =========================
# LOAD DATA
# =========================
X = []
y = []

print("Using classes:")
for label in SELECTED_LABELS:
    folder_path = os.path.join(DATA_PATH, label)
    count = len([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    print(f"{label}: {count} samples")

# Label mapping
label_map = {label: idx for idx, label in enumerate(SELECTED_LABELS)}
print("\nLabel mapping:", label_map)

# Read .npy files
for label in SELECTED_LABELS:
    label_dir = os.path.join(DATA_PATH, label)
    for file in os.listdir(label_dir):
        if file.endswith(".npy"):
            data = np.load(os.path.join(label_dir, file))
            X.append(data)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("\nTotal samples:", X.shape[0])
print("Feature size:", X.shape[1])

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL (SCALED + BALANCED)
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True
    ))
])

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {acc * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "isl_gesture_model.pkl")
joblib.dump(label_map, "label_map.pkl")

print("\nModel saved as isl_gesture_model.pkl")

