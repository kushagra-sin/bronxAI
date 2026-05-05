import serial
import time
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy import signal
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import interp1d
import requests
import json

# --- Configuration ---
SERIAL_PORT = '/dev/cu.SLAB_USBtoUART'  # Change this to your ESP32's COM port!
BAUD_RATE = 115200
SAMPLE_RATE = 500      # Hz (Your AD8232 sensor output)
PLOT_DURATION_SECONDS = 10 # How many seconds of ECG data to plot per image
OUTPUT_FOLDER = 'ecg_reports'

# Cloud API Configuration
API_ENDPOINT = "https://bronxai-1.onrender.com/predict_signal" # <--- CORRECTED!
AUTO_UPLOAD = True  # Set to True to automatically upload to cloud

# Calculate number of samples needed for one plot
NUM_SAMPLES_PER_PLOT = int(SAMPLE_RATE * PLOT_DURATION_SECONDS)

# Initialize data buffer
ecg_data = []

# Ensure output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- Signal Processing Functions ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(data, fs, freq=50.0, quality=30.0):
    """
    Apply notch filter to remove powerline interference (50/60 Hz)
    """
    nyquist = 0.5 * fs
    freq = freq / nyquist
    b, a = signal.iirnotch(freq, quality)
    return filtfilt(b, a, data)

def denoise_ecg_signal(raw_signal, fs=500):
    """
    Comprehensive ECG denoising pipeline
    """
    signal_array = np.array(raw_signal, dtype=float)
    
    # High-pass filter (baseline wander removal)
    b_high, a_high = butter(4, 0.5, btype='highpass', fs=fs)
    signal_baseline_removed = filtfilt(b_high, a_high, signal_array)
    
    # Bandpass filter (0.5-40 Hz)
    b_band, a_band = butter_bandpass(0.5, 40, fs, order=4)
    signal_bandpass = filtfilt(b_band, a_band, signal_array)
    
    # Notch filters (50Hz and 60Hz)
    signal_notch = notch_filter(signal_bandpass, fs, freq=60.0, quality=30)
    signal_notch = notch_filter(signal_notch, fs, freq=50.0, quality=30)
    
    # Median filter
    signal_denoised = medfilt(signal_notch, kernel_size=3)
    
    # Moving average
    window_size = 3
    signal_smoothed = np.convolve(signal_denoised, np.ones(window_size)/window_size, mode='same')
    
    return signal_smoothed

def resample_to_300hz(signal_data, original_fs=500, target_fs=300):
    """
    Resample from 500 Hz (sensor) to 300 Hz (model expects)
    This matches the PhysioNet dataset frequency
    """
    duration = len(signal_data) / original_fs
    target_length = int(duration * target_fs)
    
    # Resample using interpolation
    x_old = np.linspace(0, 1, len(signal_data))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, signal_data, kind='cubic')
    resampled = f(x_new)
    
    return resampled

# --- Plotting Function ---
def generate_and_save_plot(data, filename="ecg_report.png"):
    if not data or len(data) < 100:
        print("Insufficient data to plot.")
        return None

    # Apply denoising
    print("Applying signal denoising filters...")
    denoised_data = denoise_ecg_signal(data, fs=SAMPLE_RATE)
    
    # Create time axis
    time_axis = np.arange(len(data)) / SAMPLE_RATE

    # Scale denoised signal back to ADC range for consistency
    denoised_scaled = (denoised_data - np.min(denoised_data)) / (np.max(denoised_data) - np.min(denoised_data)) * 4095
    
    # Calculate signal range with extra padding
    signal_min = np.min(denoised_scaled)
    signal_max = np.max(denoised_scaled)
    signal_range = signal_max - signal_min
    y_padding = signal_range * 0.3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot denoised signal
    ax.plot(time_axis, denoised_scaled, color='#DC143C', linewidth=1.2)
    
    # Styling
    ax.set_title(f"ECG Signal ({PLOT_DURATION_SECONDS}s)", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Amplitude (mV)", fontsize=13, fontweight='bold')
    
    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=1.2, color='#D32F2F', alpha=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='-', linewidth=0.6, color='#EF5350', alpha=0.4)
    
    # Y-axis with extra space
    ax.set_ylim(signal_min - y_padding, signal_max + y_padding)
    
    # Background
    ax.set_facecolor('#FFF5F5')
    fig.patch.set_facecolor('white')
    
    # Tick settings
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    
    y_range = (signal_max + y_padding) - (signal_min - y_padding)
    y_major_interval = y_range / 8
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_major_interval))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(y_major_interval / 5))
    
    plt.tight_layout()
    
    # Save
    full_path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved denoised ECG report to {full_path}")
    
    return full_path, denoised_data

def upload_to_cloud(signal_data, api_endpoint, image_path=None):
    """
    Upload denoised ECG signal to cloud API for classification
    Sends RAW SIGNAL DATA (not PNG) for better accuracy
    """
    try:
        print(f"Uploading signal data to cloud API...")
        
        # Resample from 500Hz to 300Hz (match training data)
        signal_300hz = resample_to_300hz(signal_data, original_fs=SAMPLE_RATE, target_fs=300)
        
        # Normalize (same as training)
        signal_normalized = (signal_300hz - np.mean(signal_300hz)) / (np.std(signal_300hz) + 1e-8)
        
        # Prepare JSON payload
        payload = {
        'signal': signal_normalized.tolist(), # Send the resampled but not yet normalized signal
        'sample_rate': 300,
        'duration': PLOT_DURATION_SECONDS,
        'source': 'AD8232_sensor'
    }
        
        # Send to API
        response = requests.post(
            api_endpoint,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("AI CLASSIFICATION RESULT")
            print("="*60)
            print(f"Prediction: {result['result']['prediction']}")
            print(f"Confidence: {result['result']['confidence']*100:.2f}%")
            print("\nAll Probabilities:")
            for label, prob in result['result']['probabilities'].items():
                print(f"  {label}: {prob*100:.2f}%")
            print("="*60 + "\n")
            return result
        else:
            print(f"✗ API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("✗ Upload timeout. Check your internet connection.")
        return None
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Check if the API is running.")
        return None
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return None

# --- Serial Port Setup ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"✓ Connected to serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
    print(f"✓ Collecting {PLOT_DURATION_SECONDS}s segments at {SAMPLE_RATE}Hz")
    print(f"✓ Output folder: {OUTPUT_FOLDER}/")
    print("="*60)
except serial.SerialException as e:
    print(f"✗ Error: Could not open serial port {SERIAL_PORT}. {e}")
    print("\nTroubleshooting:")
    print("1. Ensure ESP32 is connected")
    print("2. Close other programs using the port")
    print("3. Check the correct COM port is selected")
    exit()

# --- Main Loop ---
print("Collecting ECG data... Press Ctrl+C to stop.\n")
try:
    sample_count = 0
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        # Look for ">ECG:VALUE" pattern
        match = re.match(r'>ECG:(\d+)', line)
        if match:
            try:
                ecg_value = int(match.group(1))
                ecg_data.append(ecg_value)
                sample_count += 1
                
                # Progress indicator
                if sample_count % 500 == 0:
                    progress = (len(ecg_data) / NUM_SAMPLES_PER_PLOT) * 100
                    print(f"Progress: {progress:.1f}% ({len(ecg_data)}/{NUM_SAMPLES_PER_PLOT} samples)", end='\r')

                # If enough data is collected
                if len(ecg_data) >= NUM_SAMPLES_PER_PLOT:
                    print("\n" + "="*60)
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = f"ecg_report_{timestamp_str}.png"
                    
                    # Generate PNG and get denoised signal
                    generated_filepath, denoised_signal = generate_and_save_plot(ecg_data, output_filename)
                    
                    print("="*60)
                    print(f"✓ File saved: {generated_filepath}")
                    
                    # Upload to cloud if enabled
                    if AUTO_UPLOAD and generated_filepath:
                        classification_result = upload_to_cloud(
                            denoised_signal,
                            API_ENDPOINT,
                            image_path=generated_filepath
                        )
                        
                        # Save result to JSON file for GUI
                        if classification_result:
                            result_filename = output_filename.replace('.png', '_result.json')
                            result_path = os.path.join(OUTPUT_FOLDER, result_filename)
                            with open(result_path, 'w') as f:
                                json.dump(classification_result, f, indent=2)
                            print(f"✓ Result saved to: {result_path}")
                    
                    print("="*60 + "\n")
                    
                    # Clear buffer
                    ecg_data = []
                    sample_count = 0
                    
            except ValueError:
                print(f"Could not parse ECG value from line: {line}")
        elif line and not line.startswith('>'):
            if "Initialized" in line or "---" not in line:
                print(f"ESP32: {line}")

except KeyboardInterrupt:
    print("\n\n✓ Stopping data collection.")
except Exception as e:
    print(f"\n✗ An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if ser.is_open:
        ser.close()
        print("✓ Serial port closed.")

print("\nSession complete!")
print(f"ECG reports saved in: {OUTPUT_FOLDER}/")
print("Upload the PNG files to your AI model or check JSON results.")