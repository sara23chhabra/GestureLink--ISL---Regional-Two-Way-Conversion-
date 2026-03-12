from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =========================
# LOAD ORIGINAL MODEL (weights source)
# =========================
old_model = load_model("isl_lstm_model.h5", compile=False)

# =========================
# REBUILD ARCHITECTURE (NO batch_shape)
# =========================
new_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(80, 63)),
    Dropout(0.3),
    LSTM(64),
    Dense(8, activation="softmax")
])

# =========================
# COPY WEIGHTS
# =========================
new_model.set_weights(old_model.get_weights())

# =========================
# SAVE CLEAN MODEL
# =========================
new_model.save("isl_lstm_model_clean.h5", include_optimizer=False)

print("✅ Clean inference model saved as isl_lstm_model_clean.h5")
