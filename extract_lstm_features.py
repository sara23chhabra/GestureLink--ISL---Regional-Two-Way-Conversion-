import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model

# =========================
# LOAD DATA
# =========================
X = np.load("lstm_data_webcam/X.npy")
y = np.load("lstm_data_webcam/y.npy")

print("X:", X.shape)  # (N, 40, 126)
print("y:", y.shape)

# =========================
# LOAD TRAINED LSTM MODEL
# =========================
full_model = load_model("isl_lstm_model.h5", compile=False)

# =========================
# CUT MODEL BEFORE SOFTMAX
# =========================
feature_model = Model(
    inputs=full_model.input,
    outputs=full_model.layers[-2].output  # 64-D LSTM output
)

# =========================
# EXTRACT FEATURES
# =========================
X_features = feature_model.predict(X, verbose=1)
print("Extracted features:", X_features.shape)  # (N, 64)

# =========================
# SAVE FEATURES
# =========================
np.save("lstm_data_webcam/X_features.npy", X_features)
np.save("lstm_data_webcam/y_features.npy", y)

print("✅ LSTM features saved")
