import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
X = np.load("static_svm/X.npy")
y = np.load("static_svm/y.npy")

with open("static_svm/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)
print("Labels:", label_map)

# =========================
# TRAIN / VALIDATION SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# BUILD SVM PIPELINE
# =========================
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=5,
        gamma="scale",
        probability=True
    ))
])

# =========================
# TRAIN
# =========================
svm.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = svm.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"\n✅ Static SVM Accuracy: {acc * 100:.2f}%\n")
print("Classification report:")
print(classification_report(
    y_val, y_pred,
    target_names=[inv_label_map[i] for i in sorted(inv_label_map)]
))

# =========================
# SAVE MODEL
# =========================
with open("static_svm/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("💾 Static SVM model saved as static_svm/svm_model.pkl")

