# --- main.py for FastAPI (Revised for TFLite) ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # For input validation
import tensorflow as tf # Still need TensorFlow to load TFLite interpreter
import numpy as np
from scipy import signal as scipy_signal # Use alias to avoid conflict with `signal` module
import os # For checking file existence

# Initialize FastAPI
app = FastAPI(
    title="ECG Classification API (TFLite)",
    description="AI-powered ECG signal classifier for 1D raw data using TensorFlow Lite",
    version="1.0.2" # Updated version number
)

# Enable CORS for frontend access (adjust allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, specify your frontend URL, e.g., ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Label Loading ---
# IMPORTANT: These files MUST be in the same directory as this main.py when deployed.
TFLITE_MODEL_PATH = 'ecg_classifier_300hz.tflite'
LABELS_PATH = 'label_classes.npy'

# Ensure model and labels exist before attempting to load
if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(f"TFLite model file not found at {TFLITE_MODEL_PATH}. Make sure it's uploaded.")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}. Make sure it's uploaded.")

print("Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors() # Allocate tensors before inference
print("✓ TFLite Interpreter loaded!")

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_classes = np.load(LABELS_PATH, allow_pickle=True)
print(f"✓ Label classes loaded! Classes: {label_classes.tolist()}")

# --- Configuration for Preprocessing (MUST MATCH TRAINING) ---
TARGET_LENGTH = 3000 # 10 seconds * 300 Hz (from your Colab script)
ORIGINAL_FS = 300    # Original sampling frequency of PhysioNet data / expected input to API

# --- Signal Preprocessing Function (MATCHES YOUR COLAB TRAINING) ---
def preprocess_signal_for_inference(signal_data, target_length=TARGET_LENGTH, fs=ORIGINAL_FS):
    """
    Preprocess ECG signal for inference.
    This must exactly match the `preprocess_signal` function used during Colab training.
    """
    signal_array = np.array(signal_data, dtype=np.float32)

    # 1. Bandpass filter (0.5-40 Hz)
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40 / nyquist
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    filtered_signal = scipy_signal.filtfilt(b, a, signal_array)
    
    # 2. Normalize to [-1, 1]
    min_val = np.min(filtered_signal)
    max_val = np.max(filtered_signal)
    if (max_val - min_val) > 1e-6: # Avoid division by zero
        normalized = 2 * (filtered_signal - min_val) / (max_val - min_val) - 1
    else:
        normalized = filtered_signal * 0 # If flat, set all to 0
    
    # 3. Pad or truncate to target length
    if len(normalized) < target_length:
        padded = np.pad(normalized, (0, target_length - len(normalized)), mode='constant')
    elif len(normalized) > target_length:
        padded = normalized[:target_length]
    else:
        padded = normalized
        
    return padded

# --- Prediction Function ---
def classify_ecg(signal_data_1d):
    """
    Classify a single preprocessed 1D ECG signal using the TFLite interpreter.
    """
    # TFLite models expect specific data types, typically float32
    input_shape = input_details[0]['shape'] # e.g., [1, 3000, 1]
    input_dtype = input_details[0]['dtype'] # e.g., float32

    # Reshape for TFLite model input: (1, timesteps, features=1)
    # Ensure the data type matches what the TFLite model expects
    signal_reshaped = signal_data_1d.reshape(input_shape).astype(input_dtype)
    
    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], signal_reshaped)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    prediction_raw = interpreter.get_tensor(output_details[0]['index'])
    
    # Get class probabilities and predicted class
    prediction_probs = prediction_raw[0] # TFLite output usually has batch dim 1
    class_idx = np.argmax(prediction_probs)
    
    predicted_class = label_classes[class_idx]
    confidence = float(prediction_probs[class_idx])
    
    # Get all probabilities as a dictionary
    probabilities = {
        label_classes[i]: float(prediction_probs[i])
        for i in range(len(label_classes))
    }
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

# --- Pydantic Model for Input Validation ---
class ECGSignalInput(BaseModel):
    signal: list[float] # Expects a list of floats

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint and basic info."""
    return {
        "status": "online",
        "message": "ECG Classification API is running (TFLite Backend)",
        "version": "1.0.2",
        "model_classes": label_classes.tolist(),
        "expected_signal_length_after_preprocessing": TARGET_LENGTH,
        "note": "Send raw 1D ECG data to /predict_signal endpoint."
    }

@app.post("/predict_signal")
async def predict_ecg_signal(ecg_input: ECGSignalInput):
    """
    Endpoint to receive raw 1D ECG signal data (list of floats) and return classification.
    The client (your local Python script) should perform denoising and initial resampling
    to 300Hz before sending.
    The API will then perform final padding/truncation to 3000 samples and normalization.
    """
    try:
        raw_signal_from_client = ecg_input.signal
        
        # Preprocess the signal, matching the training pipeline
        processed_signal = preprocess_signal_for_inference(raw_signal_from_client, target_length=TARGET_LENGTH, fs=ORIGINAL_FS)
        
        # Classify the preprocessed signal
        result = classify_ecg(processed_signal)
        
        return {
            "success": True,
            "result": result,
            "received_signal_length": len(raw_signal_from_client),
            "processed_signal_length_for_model_input": len(processed_signal)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full traceback for server-side debugging
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}. Check signal format and preprocessing.")

# ============================================================================
# RUN SERVER (for local testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    # For local testing, run: python main.py
    # Then open http://127.0.0.1:8000/docs in your browser to see the API docs.
    uvicorn.run(app, host="0.0.0.0", port=8000)
