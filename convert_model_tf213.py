import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("TensorFlow version:", tf.__version__)

# 1. Rebuild architecture (must match training)
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(80, 63)),
    Dropout(0.3),
    LSTM(64),
    Dense(8, activation="softmax")
])

# 2. Build model
model(tf.zeros((1, 80, 63)))

# 3. Load ORIGINAL model (this WILL work in TF 2.13)
old_model = load_model("isl_lstm_model.h5", compile=False)

# 4. Copy weights layer by layer
for new_layer, old_layer in zip(model.layers, old_model.layers):
    try:
        new_layer.set_weights(old_layer.get_weights())
    except Exception:
        pass

# 5. Save clean inference model
model.save("isl_lstm_tf213_infer.h5", include_optimizer=False)

print("✅ Model successfully converted for TF 2.13 inference")
