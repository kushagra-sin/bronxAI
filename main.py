"""
ECG Classification API - FIXED VERSION
Correctly handles preprocessed signals from client
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

# Initialize FastAPI
app = FastAPI(
    title="BronxAI ECG Classification API",
    description="AI-powered ECG signal classifier using TensorFlow Lite",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
TFLITE_MODEL_PATH = 'ecg_classifier_300hz.tflite'
LABELS_PATH = 'label_classes.npy'

# Check files exist
if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found: {TFLITE_MODEL_PATH}")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

print("Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
print("✓ TFLite Interpreter loaded!")

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_classes = np.load(LABELS_PATH, allow_pickle=True)

# CRITICAL FIX: Swap labels to correct reversed predictions
# The model outputs are reversed, so we swap the label mapping
label_mapping = {
    'Normal': 'Abnormal',
    'Abnormal': 'Normal',
    'Atrial_Fibrillation': 'Atrial_Fibrillation'  # This stays the same
}

# Apply the swap
label_classes_corrected = np.array([label_mapping.get(label, label) for label in label_classes])

print(f"✓ Original labels: {label_classes.tolist()}")
print(f"✓ Corrected labels: {label_classes_corrected.tolist()}")

# Use corrected labels for predictions
label_classes = label_classes_corrected

# Expected input configuration
TARGET_LENGTH = 3000  # 10 seconds * 300 Hz
TARGET_FS = 300

# --- Input Validation Model ---
class ECGSignalInput(BaseModel):
    signal: list[float]
    sample_rate: int = 300
    duration: float = 10.0
    source: str = "unknown"

# --- Minimal Preprocessing (Client already did the work!) ---
def prepare_for_inference(signal_data, target_length=TARGET_LENGTH):
    """
    MINIMAL preprocessing - client already denoised, resampled, and normalized!
    Just ensure correct length and shape.
    
    Args:
        signal_data: Already preprocessed and normalized signal from client
        target_length: Expected length (3000 samples)
    
    Returns:
        Signal ready for model inference
    """
    signal_array = np.array(signal_data, dtype=np.float32)
    
    print(f"📥 Received signal - Length: {len(signal_array)}, "
          f"Range: [{np.min(signal_array):.4f}, {np.max(signal_array):.4f}], "
          f"Mean: {np.mean(signal_array):.4f}, Std: {np.std(signal_array):.4f}")
    
    # Verify normalization looks correct (should have mean≈0, std≈1)
    signal_mean = np.mean(signal_array)
    signal_std = np.std(signal_array)
    
    if abs(signal_mean) > 0.5 or signal_std < 0.5 or signal_std > 2.0:
        print(f"  WARNING: Signal normalization looks unusual!")
        print(f"   Mean: {signal_mean:.4f} (expected ≈0)")
        print(f"   Std:  {signal_std:.4f} (expected ≈1)")
    
    # Pad or truncate to exact target length
    if len(signal_array) < target_length:
        # Pad with zeros
        padded = np.pad(signal_array, (0, target_length - len(signal_array)), mode='constant', constant_values=0)
        print(f"  Signal padded from {len(signal_array)} to {target_length}")
    elif len(signal_array) > target_length:
        # Truncate
        padded = signal_array[:target_length]
        print(f"  Signal truncated from {len(signal_array)} to {target_length}")
    else:
        padded = signal_array
        print(f"✓ Signal length correct: {target_length}")
    
    return padded

# --- Classification Function ---
def classify_ecg(signal_data_1d):
    """
    Classify preprocessed ECG signal using TFLite model
    """
    # Reshape for model: (1, 3000, 1)
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    signal_reshaped = signal_data_1d.reshape(input_shape).astype(input_dtype)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], signal_reshaped)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    prediction_raw = interpreter.get_tensor(output_details[0]['index'])
    prediction_probs = prediction_raw[0]
    
    class_idx = np.argmax(prediction_probs)
    predicted_class = label_classes[class_idx]
    confidence = float(prediction_probs[class_idx])
    
    # All probabilities
    probabilities = {
        label_classes[i]: float(prediction_probs[i])
        for i in range(len(label_classes))
    }
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "status": "online",
        "message": "BronxAI ECG Classification API",
        "version": "2.0.0",
        "model": "1D CNN TFLite",
        "classes": label_classes.tolist(),
        "expected_input": {
            "signal_length": TARGET_LENGTH,
            "sample_rate": f"{TARGET_FS} Hz",
            "duration": f"{TARGET_LENGTH/TARGET_FS} seconds",
            "normalization": "Z-score (mean≈0, std≈1)",
            "note": "Client should send preprocessed & normalized signal"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "interpreter_ready": True,
        "classes": label_classes.tolist(),
        "input_shape": input_details[0]['shape'],
        "output_shape": output_details[0]['shape']
    }

@app.post("/predict_signal")
async def predict_ecg_signal(ecg_input: ECGSignalInput):
    """
    Classify ECG signal
    
    IMPORTANT: Client must send signal that is:
    - Already denoised (bandpass filtered, notch filtered, etc.)
    - Resampled to 300 Hz
    - Z-score normalized (mean≈0, std≈1)
    - Approximately 3000 samples (10 seconds)
    
    Returns:
    - prediction: Predicted class
    - confidence: Confidence score
    - probabilities: Probability for each class
    """
    try:
        raw_signal = ecg_input.signal
        
        print("\n" + "="*70)
        print(f"   Received prediction request from: {ecg_input.source}")
        print(f"   Sample rate: {ecg_input.sample_rate} Hz")
        print(f"   Duration: {ecg_input.duration} seconds")
        print(f"   Signal length: {len(raw_signal)} samples")
        print("="*70)
        
        # Validate input length
        if len(raw_signal) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Signal too short: {len(raw_signal)} samples. Need at least 100."
            )
        
        if len(raw_signal) < TARGET_LENGTH * 0.9:
            print(f" WARNING: Signal shorter than expected ({len(raw_signal)} vs {TARGET_LENGTH})")
        
        # Prepare for inference (minimal processing)
        processed_signal = prepare_for_inference(raw_signal, target_length=TARGET_LENGTH)
        
        # Classify
        result = classify_ecg(processed_signal)
        
        print(f"\n Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print("="*70 + "\n")
        
        return {
            "success": True,
            "source": ecg_input.source,
            "input_info": {
                "received_samples": len(raw_signal),
                "processed_samples": len(processed_signal),
                "sample_rate": ecg_input.sample_rate,
                "duration": ecg_input.duration
            },
            "result": result,
            "debug_info": {
                "input_stats": {
                    "original_mean": float(np.mean(raw_signal)),
                    "original_std": float(np.std(raw_signal)),
                    "original_min": float(np.min(raw_signal)),
                    "original_max": float(np.max(raw_signal))
                },
                "model_input_shape": input_details[0]['shape']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("Starting BronxAI ECG Classification API")
    print("="*70)
    print(f"Classes: {label_classes.tolist()}")
    print(f"Expected input: {TARGET_LENGTH} samples at {TARGET_FS} Hz")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
