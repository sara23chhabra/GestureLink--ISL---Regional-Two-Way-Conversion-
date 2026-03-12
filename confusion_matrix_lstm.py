import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# LOAD DATA
# =========================
X = np.load("lstm_data/X.npy")
y = np.load("lstm_data/y.npy")

with open("lstm_data/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# =========================
# SPLIT DATA (same as training)
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# LOAD MODEL
# =========================
model = load_model("isl_lstm_model.h5")

# =========================
# PREDICT ON VALIDATION SET
# =========================
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_val, y_pred)

print("\nCONFUSION MATRIX (rows = actual, cols = predicted)\n")
print(cm)

# =========================
# CLASSIFICATION REPORT
# =========================
target_names = [inv_label_map[i] for i in sorted(inv_label_map)]

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_val, y_pred, target_names=target_names))
