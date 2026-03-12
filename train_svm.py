import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# =========================
# LOAD FEATURES
# =========================
X = np.load("lstm_data_webcam/X_features.npy")
y = np.load("lstm_data_webcam/y_features.npy")

# =========================
# TRAIN / VAL SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# SVM PIPELINE
# =========================
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        C=10,
        gamma="scale"
    ))
])

# =========================
# TRAIN
# =========================
svm_model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = svm_model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"✅ SVM Validation Accuracy: {acc*100:.2f}%")

# =========================
# SAVE
# =========================
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("💾 SVM model saved")
