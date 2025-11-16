from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from datetime import datetime
import traceback
import json
import os
import logging
from logging.handlers import RotatingFileHandler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
if not app.debug:
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler('logs/ecg_api.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('ECG API startup')

# Model configuration - use environment variables for production
MODEL_PATH = os.environ.get('MODEL_PATH', 'ecg_classifier_300hz.h5')
EXPECTED_SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', 300))
EXPECTED_SIGNAL_LENGTH = int(os.environ.get('SIGNAL_LENGTH', 3000))

# Class labels - ADJUST THESE TO MATCH YOUR MODEL!
CLASS_LABELS = os.environ.get('CLASS_LABELS', 'Normal,Abnormal,Atrial Fibrillation').split(',')

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info(f"✓ Model loaded successfully from {MODEL_PATH}")
        app.logger.info(f"✓ Model input shape: {model.input_shape}")
        app.logger.info(f"✓ Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        app.logger.error(f"✗ Error loading model: {e}")
        app.logger.error(traceback.format_exc())
        return False

def log_signal_stats(signal, stage=""):
    """Detailed logging of signal statistics"""
    signal_array = np.array(signal)
    
    stats = {
        'stage': stage,
        'length': len(signal_array),
        'mean': float(np.mean(signal_array)),
        'std': float(np.std(signal_array)),
        'min': float(np.min(signal_array)),
        'max': float(np.max(signal_array)),
        'median': float(np.median(signal_array)),
        'has_nan': bool(np.any(np.isnan(signal_array))),
        'has_inf': bool(np.any(np.isinf(signal_array))),
    }
    
    log_msg = (f"\n{'='*70}\n"
               f"SIGNAL STATISTICS - {stage}\n"
               f"{'='*70}\n"
               f"Length:     {stats['length']} samples\n"
               f"Mean:       {stats['mean']:.6f}\n"
               f"Std Dev:    {stats['std']:.6f}\n"
               f"Min:        {stats['min']:.6f}\n"
               f"Max:        {stats['max']:.6f}\n"
               f"Has NaN:    {stats['has_nan']}\n"
               f"Has Inf:    {stats['has_inf']}\n"
               f"{'='*70}")
    
    app.logger.info(log_msg)
    return stats

def preprocess_signal(raw_signal, expected_length=EXPECTED_SIGNAL_LENGTH):
    """
    Preprocess the incoming signal
    THIS MUST MATCH YOUR TRAINING PREPROCESSING EXACTLY
    """
    app.logger.info("Starting signal preprocessing pipeline")
    
    # Step 1: Convert to numpy array
    signal = np.array(raw_signal, dtype=np.float32)
    log_signal_stats(signal, "1. RAW INPUT")
    
    # Step 2: Check for invalid values
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Signal contains NaN or Inf values!")
    
    # Step 3: Normalize (Z-score normalization)
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    
    if signal_std < 1e-8:
        app.logger.warning("Signal has near-zero variance! Using mean centering only.")
        signal_normalized = signal - signal_mean
    else:
        signal_normalized = (signal - signal_mean) / signal_std
    
    log_signal_stats(signal_normalized, "2. AFTER NORMALIZATION")
    
    # Step 4: Ensure correct length
    current_length = len(signal_normalized)
    
    if current_length > expected_length:
        app.logger.warning(f"Signal too long ({current_length}). Truncating to {expected_length}")
        signal_normalized = signal_normalized[:expected_length]
    elif current_length < expected_length:
        app.logger.warning(f"Signal too short ({current_length}). Padding to {expected_length}")
        pad_length = expected_length - current_length
        signal_normalized = np.pad(signal_normalized, (0, pad_length), mode='constant')
    
    log_signal_stats(signal_normalized, "3. AFTER LENGTH ADJUSTMENT")
    
    # Step 5: Reshape for model input
    signal_reshaped = signal_normalized.reshape(1, expected_length, 1)
    
    app.logger.info(f"Final shape for model: {signal_reshaped.shape}")
    
    return signal_reshaped, {
        'original_length': len(raw_signal),
        'original_mean': float(signal_mean),
        'original_std': float(signal_std),
        'normalized_mean': float(np.mean(signal_normalized)),
        'normalized_std': float(np.std(signal_normalized))
    }

def log_model_output(predictions, class_labels):
    """Detailed logging of model predictions"""
    log_msg = f"\n{'='*70}\nMODEL OUTPUT ANALYSIS\n{'='*70}\n"
    
    for i, (label, prob) in enumerate(zip(class_labels, predictions[0])):
        log_msg += f"  {label:20s}: {prob:.6f}\n"
    
    predicted_idx = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    log_msg += f"\n✓ PREDICTION: {predicted_label}\n"
    log_msg += f"✓ CONFIDENCE: {confidence*100:.2f}%\n"
    
    sorted_probs = sorted(predictions[0], reverse=True)
    confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
    
    log_msg += f"\n📊 Confidence Analysis:\n"
    log_msg += f"   Top probability:    {sorted_probs[0]*100:.2f}%\n"
    if len(sorted_probs) > 1:
        log_msg += f"   2nd probability:    {sorted_probs[1]*100:.2f}%\n"
        log_msg += f"   Gap (certainty):    {confidence_gap*100:.2f}%\n"
    
    if confidence_gap < 0.2:
        log_msg += "   ⚠ LOW CERTAINTY - Model is unsure!\n"
    elif confidence_gap > 0.5:
        log_msg += "   ✓ HIGH CERTAINTY - Model is confident\n"
    
    log_msg += f"{'='*70}"
    app.logger.info(log_msg)
    
    return predicted_label, confidence

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'ECG Classification API',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'health': '/health',
            'predict': '/predict_signal',
            'test': '/test_with_sample'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'expected_input_length': EXPECTED_SIGNAL_LENGTH,
        'expected_sample_rate': EXPECTED_SAMPLE_RATE,
        'class_labels': CLASS_LABELS,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/predict_signal', methods=['POST'])
def predict_signal():
    """
    API endpoint for ECG signal classification
    Expects JSON: {'signal': [values], 'sample_rate': 300, ...}
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        app.logger.info(f"New prediction request - {timestamp}")
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The model failed to load at startup'
            }), 503
        
        # Parse request
        data = request.get_json()
        
        if not data or 'signal' not in data:
            return jsonify({
                'error': 'Missing signal data',
                'expected_format': {
                    'signal': [1, 2, 3],
                    'sample_rate': 300,
                    'duration': 10
                }
            }), 400
        
        # Extract data
        raw_signal = data['signal']
        sample_rate = data.get('sample_rate', EXPECTED_SAMPLE_RATE)
        duration = data.get('duration', None)
        source = data.get('source', 'unknown')
        
        app.logger.info(f"Request Info - Source: {source}, Sample Rate: {sample_rate}Hz, "
                       f"Duration: {duration}s, Length: {len(raw_signal)}")
        
        # Validate sample rate
        if sample_rate != EXPECTED_SAMPLE_RATE:
            app.logger.warning(f"Sample rate mismatch! Expected: {EXPECTED_SAMPLE_RATE}Hz, "
                             f"Received: {sample_rate}Hz")
        
        # Preprocess signal
        processed_signal, preprocessing_stats = preprocess_signal(
            raw_signal, 
            expected_length=EXPECTED_SIGNAL_LENGTH
        )
        
        # Make prediction
        app.logger.info("Running model inference...")
        predictions = model.predict(processed_signal, verbose=0)
        
        # Log and analyze output
        predicted_label, confidence = log_model_output(predictions, CLASS_LABELS)
        
        # Prepare response
        probabilities = {
            label: float(prob) 
            for label, prob in zip(CLASS_LABELS, predictions[0])
        }
        
        response = {
            'success': True,
            'timestamp': timestamp,
            'result': {
                'prediction': predicted_label,
                'confidence': float(confidence),
                'probabilities': probabilities
            },
            'debug_info': {
                'input_stats': preprocessing_stats,
                'model_input_shape': list(processed_signal.shape),
                'sample_rate': sample_rate,
                'signal_length': len(raw_signal)
            }
        }
        
        app.logger.info(f"Request completed successfully - Prediction: {predicted_label}")
        return jsonify(response), 200
        
    except ValueError as e:
        app.logger.error(f"Validation Error: {e}")
        return jsonify({
            'error': 'Invalid input data',
            'message': str(e)
        }), 400
        
    except Exception as e:
        app.logger.error(f"Internal Error: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/test_with_sample', methods=['POST', 'GET'])
def test_with_sample():
    """
    Test endpoint - generates a synthetic normal ECG signal
    Useful for debugging without hardware
    """
    try:
        app.logger.info("Generating synthetic ECG signal for testing...")
        
        # Generate synthetic ECG
        t = np.linspace(0, 10, EXPECTED_SIGNAL_LENGTH)
        heart_rate = 75  # bpm
        frequency = heart_rate / 60
        
        signal = np.zeros_like(t)
        for beat_time in np.arange(0, 10, 1/frequency):
            peak_idx = int((beat_time / 10) * EXPECTED_SIGNAL_LENGTH)
            if peak_idx < len(signal):
                signal[max(0, peak_idx-5):min(len(signal), peak_idx+5)] += \
                    np.exp(-((np.arange(-5, 5)**2) / 2)) * 100
        
        signal += np.sin(2 * np.pi * 0.1 * t) * 20
        signal += np.random.normal(0, 5, len(signal))
        
        test_data = {
            'signal': signal.tolist(),
            'sample_rate': EXPECTED_SAMPLE_RATE,
            'duration': 10,
            'source': 'synthetic_test'
        }
        
        # Call prediction endpoint
        with app.test_request_context('/predict_signal', method='POST', json=test_data):
            return predict_signal()
            
    except Exception as e:
        app.logger.error(f"Test error: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize model on startup
with app.app_context():
    app.logger.info("="*70)
    app.logger.info("ECG CLASSIFICATION API STARTING")
    app.logger.info("="*70)
    app.logger.info(f"Expected Signal Length:  {EXPECTED_SIGNAL_LENGTH} samples")
    app.logger.info(f"Expected Sample Rate:    {EXPECTED_SAMPLE_RATE} Hz")
    app.logger.info(f"Class Labels:            {CLASS_LABELS}")
    app.logger.info(f"Model Path:              {MODEL_PATH}")
    app.logger.info("="*70)
    
    if not load_model():
        app.logger.error("Failed to load model. API will return 503 errors.")

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)