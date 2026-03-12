import numpy as np
from tensorflow.keras.models import load_model

# Load original model (this works ONLY here)
model = load_model("isl_lstm_model.h5", compile=False)

# Extract weights
weights = model.get_weights()

# Save as pure NumPy (no Keras metadata)
np.savez("isl_lstm_weights_numpy.npz", *weights)

print("✅ Weights exported as isl_lstm_weights_numpy.npz")
