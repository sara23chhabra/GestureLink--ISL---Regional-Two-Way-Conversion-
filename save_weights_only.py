from tensorflow.keras.models import load_model

# Load original trained model
model = load_model("isl_lstm_model.h5", compile=False)

# Save ONLY the weights (correct filename)
model.save_weights("isl_lstm.weights.h5")

print("✅ Weights saved as isl_lstm.weights.h5")
