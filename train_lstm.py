import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# =========================
# LOAD DATA
# =========================
X = np.load("lstm_data_webcam/X.npy")
# After loading X
assert X.shape[1] == 40, "❌ Sequence length must be 40"

y = np.load("lstm_data_webcam/y.npy")

print("Loaded X:", X.shape)   # (N, 80, 126)
print("Loaded y:", y.shape)

# =========================
# ONE-HOT ENCODE LABELS
# =========================
num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)

print("One-hot y shape:", y_cat.shape)

# =========================
# TRAIN / VALIDATION SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train samples:", X_train.shape[0])
print("Val samples:", X_val.shape[0])

# =========================
# BUILD LSTM MODEL (TWO HANDS)
# =========================
model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(40, 126)   # ✅ IMPORTANT CHANGE
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(
        64,
        return_sequences=False
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
)

model.summary()

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
# =========================
model.save("isl_lstm_model.h5")
print("\n💾 Model saved as isl_lstm_model.h5")

# =========================
# SAVE TRAINING HISTORY
# =========================
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("🎉 LSTM TRAINING COMPLETE")

