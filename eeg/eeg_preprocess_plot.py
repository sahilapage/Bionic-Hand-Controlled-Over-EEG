import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from collections import deque
import time
import sys

PORT = "COM4"
BAUD = 115200
FS = 250                    # Sampling rate (Hz)
N = 250                     # Samples per frame (1 second)
TIMEOUT = 2                 # Serial timeout (seconds)

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40)
}

# Visualization settings
HISTORY_LENGTH = 10         
TIME_YLIM = (-0.0002, 0.0002)
FREQ_XLIM = (0, 45)

def band_power(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])

def detect_state(alpha, beta):
    if alpha < 1e-10 and beta < 1e-10:
        return "Noise/No Signal"
    
    ratio = alpha / (beta + 1e-10)
    if ratio > 1.5:
        return "Relaxed 😌"
    elif ratio > 0.8:
        return "Calm 🧘"
    elif ratio > 0.4:
        return "Alert 👀"
    else:
        return "Focused 🎯"

def validate_eeg_data(eeg):
    """Check if EEG data is reasonable."""
    if np.all(eeg == 0):
        return False, "All zeros"
    if np.any(np.isnan(eeg)) or np.any(np.isinf(eeg)):
        return False, "Contains NaN/Inf"
    if np.std(eeg) < 1e-8:
        return False, "No variation"
    return True, "OK"

def connect_serial(port, baud, timeout):
    """Establish serial connection with error handling."""
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"✓ Connected to {port} at {baud} baud")
        time.sleep(2)  
        ser.reset_input_buffer()  
        return ser
    except serial.SerialException as e:
        print(f"✗ Failed to connect to {port}: {e}")
        print("\nAvailable ports:")
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"  - {p.device}: {p.description}")
        sys.exit(1)

def main():
    # Connect to ESP32
    ser = connect_serial(PORT, BAUD, TIMEOUT)
    
    # Setup plot
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax_time = fig.add_subplot(gs[0, :])      # Time domain (full width)
    ax_psd = fig.add_subplot(gs[1, :])       # PSD (full width)
    ax_bands = fig.add_subplot(gs[2, 0])     # Band powers
    ax_state = fig.add_subplot(gs[2, 1])     # State indicator
    ax_state.axis('off')
    
    time_history = deque(maxlen=HISTORY_LENGTH * N)
    
    # Statistics
    frame_count = 0
    error_count = 0
    start_time = time.time()
    last_fps_update = start_time
    fps = 0.0
    
    print("\n" + "="*50)
    print("EEG ANALYZER RUNNING")
    print("="*50)
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Read line from serial
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except serial.SerialException as e:
                print(f"Serial read error: {e}")
                error_count += 1
                continue
            
            # Check for EEG data marker
            if not line.startswith("EEG:"):
                continue
            
            # Parse data
            try:
                data = line.replace("EEG:", "").split(",")
                if len(data) != N:
                    print(f"⚠ Expected {N} samples, got {len(data)}")
                    error_count += 1
                    continue
                
                eeg = np.array(data, dtype=np.float32)
            except (ValueError, IndexError) as e:
                print(f"⚠ Parse error: {e}")
                error_count += 1
                continue
            
            # Validate data
            valid, msg = validate_eeg_data(eeg)
            if not valid:
                print(f"⚠ Invalid data: {msg}")
                error_count += 1
                continue
            
            frame_count += 1
            time_history.extend(eeg)
            
            ax_time.clear()
            time_data = np.array(time_history)
            time_axis = np.arange(len(time_data)) / FS
            ax_time.plot(time_axis, time_data, 'b-', linewidth=0.8)
            ax_time.set_title(f"EEG Signal (Time Domain) - Frame #{frame_count}", 
                            fontsize=12, fontweight='bold')
            ax_time.set_xlabel("Time (s)")
            ax_time.set_ylabel("Amplitude (V)")
            ax_time.set_ylim(TIME_YLIM)
            ax_time.grid(True, alpha=0.3)
            
            freqs, psd = welch(
                eeg,
                fs=FS,
                nperseg=N,
                window='hann',
                scaling='density',
                detrend='constant'
            )
            
            ax_psd.clear()
            ax_psd.semilogy(freqs, psd, 'r-', linewidth=1.5)
            ax_psd.set_xlim(FREQ_XLIM)
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("PSD (V²/Hz)")
            ax_psd.set_title("Power Spectral Density (Welch Method)", 
                           fontsize=12, fontweight='bold')
            ax_psd.grid(True, alpha=0.3, which='both')
            
            # Shade frequency bands
            colors = ['purple', 'blue', 'green', 'orange', 'red']
            for (band_name, (fmin, fmax)), color in zip(BANDS.items(), colors):
                ax_psd.axvspan(fmin, fmax, alpha=0.1, color=color)
            
            band_powers = {}
            for band_name, (fmin, fmax) in BANDS.items():
                band_powers[band_name] = band_power(freqs, psd, fmin, fmax)
            
            ax_bands.clear()
            bands_list = list(band_powers.keys())
            powers_list = list(band_powers.values())
            
            bars = ax_bands.bar(bands_list, powers_list, color=colors, alpha=0.7, 
                               edgecolor='black', linewidth=1.5)
            ax_bands.set_ylabel("Power (V²)")
            ax_bands.set_title("Band Powers", fontsize=11, fontweight='bold')
            ax_bands.set_yscale('log')
            ax_bands.grid(True, alpha=0.3, axis='y')
            
            # Add power values on bars
            for bar, power in zip(bars, powers_list):
                height = bar.get_height()
                ax_bands.text(bar.get_x() + bar.get_width()/2., height,
                            f'{power:.1e}',
                            ha='center', va='bottom', fontsize=8)
            
            ax_state.clear()
            ax_state.axis('off')
            
            state = detect_state(band_powers['Alpha'], band_powers['Beta'])
            alpha_beta_ratio = band_powers['Alpha'] / (band_powers['Beta'] + 1e-10)
            
            # Display state
            ax_state.text(0.5, 0.7, "Mental State", 
                         ha='center', va='center', fontsize=14, fontweight='bold')
            ax_state.text(0.5, 0.45, state, 
                         ha='center', va='center', fontsize=18, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Display metrics
            metrics_text = (
                f"α/β Ratio: {alpha_beta_ratio:.2f}\n"
                f"FPS: {fps:.1f}\n"
                f"Errors: {error_count}"
            )
            ax_state.text(0.5, 0.15, metrics_text,
                         ha='center', va='center', fontsize=9,
                         family='monospace')
            
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                last_fps_update = current_time
            
            plt.pause(0.001)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print("SHUTTING DOWN")
        print("="*50)
        elapsed = time.time() - start_time
        print(f"Total frames: {frame_count}")
        print(f"Total errors: {error_count}")
        print(f"Average FPS: {frame_count/elapsed:.2f}")
        print(f"Runtime: {elapsed:.1f} seconds")
        
    finally:
        ser.close()
        plt.close('all')
        print("\n✓ Serial port closed")
        print("✓ Goodbye!\n")

if __name__ == "__main__":
    main()