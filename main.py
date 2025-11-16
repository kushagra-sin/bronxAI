from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tflite_runtime.interpreter as tflite
from typing import List

app = FastAPI(
    title="ECG Classification API",
    description="AI-powered ECG signal classifier (Raw Signal Input)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
MODEL_PATH = 'ecg_classifier_300hz.tflite'
LABELS_PATH = 'label_classes.npy'

print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
label_classes = np.load(LABELS_PATH, allow_pickle=True)
print(f"✓ Model loaded! Classes: {label_classes}")

# Expected input configuration
EXPECTED_SAMPLES = 3000  # 10 seconds at 300Hz
EXPECTED_FS = 300  # Hz

# Request model
class ECGSignalRequest(BaseModel):
    signal: List[float]
    sample_rate: int = 300
    duration: float = 10.0
    source: str = "unknown"

def preprocess_signal(signal_data, target_length=3000):
    """
    Preprocess signal for model inference
    """
    signal_array = np.array(signal_data, dtype=np.float32)
    
    # Ensure correct length
    if len(signal_array) < target_length:
        # Pad with edge values
        signal_array = np.pad(signal_array, (0, target_length - len(signal_array)), mode='edge')
    elif len(signal_array) > target_length:
        # Truncate
        signal_array = signal_array[:target_length]
    
    # Ensure normalization (model expects normalized input)
    if np.std(signal_array) > 0:
        signal_normalized = (signal_array - np.mean(signal_array)) / np.std(signal_array)
    else:
        signal_normalized = signal_array
    
    # Reshape for model: (1, 3000, 1)
    signal_reshaped = signal_normalized.reshape(1, target_length, 1).astype(np.float32)
    
    return signal_reshaped

def classify_ecg_signal(signal_data):
    """
    Classify ECG signal using TFLite model
    """
    # Preprocess
    processed_signal = preprocess_signal(signal_data, target_length=EXPECTED_SAMPLES)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_signal)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get results
    class_idx = np.argmax(prediction)
    confidence = float(prediction[class_idx])
    predicted_class = label_classes[class_idx]
    
    # All probabilities
    probabilities = {
        label_classes[i]: float(prediction[i])
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
        "message": "ECG Classification API (TFLite - Raw Signal)",
        "version": "3.0.0",
        "model_type": "1D CNN TensorFlow Lite",
        "input_format": "Raw signal array (3000 samples at 300Hz)",
        "classes": label_classes.tolist(),
        "expected_input": {
            "samples": EXPECTED_SAMPLES,
            "sample_rate": f"{EXPECTED_FS} Hz",
            "duration": f"{EXPECTED_SAMPLES/EXPECTED_FS} seconds"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": label_classes.tolist(),
        "expected_samples": EXPECTED_SAMPLES,
        "sample_rate": EXPECTED_FS
    }

@app.post("/predict")
async def predict_ecg(request: ECGSignalRequest):
    """
    Classify ECG signal from raw data
    
    Parameters:
    - signal: List of float values (ECG samples)
    - sample_rate: Sampling frequency (should be 300 Hz)
    - duration: Duration in seconds
    - source: Source identifier (e.g., "AD8232_sensor")
    
    Returns:
    - prediction: Predicted class
    - confidence: Confidence score (0-1)
    - probabilities: Probability for each class
    """
    try:
        # Validate input
        if len(request.signal) < 100:
            raise HTTPException(
                status_code=400,
                detail="Signal too short. Need at least 100 samples."
            )
        
        if request.sample_rate != EXPECTED_FS:
            print(f"Warning: Expected {EXPECTED_FS}Hz, got {request.sample_rate}Hz. Resampling may be needed.")
        
        # Classify
        result = classify_ecg_signal(request.signal)
        
        return {
            "success": True,
            "source": request.source,
            "input_info": {
                "samples_received": len(request.signal),
                "sample_rate": request.sample_rate,
                "duration": request.duration
            },
            "result": result,
            "message": "Classification successful"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )

@app.post("/predict_batch")
async def predict_batch(signals: List[ECGSignalRequest]):
    """
    Classify multiple ECG signals
    """
    try:
        results = []
        
        for idx, signal_request in enumerate(signals):
            try:
                result = classify_ecg_signal(signal_request.signal)
                results.append({
                    "index": idx,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_signals": len(signals),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing error: {str(e)}"
        )

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)