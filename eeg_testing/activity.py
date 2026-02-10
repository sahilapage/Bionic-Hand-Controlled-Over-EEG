import pylsl
import numpy as np
from collections import deque
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
import time

# Configuration
FFT_WINDOW_SIZE = 512  # ~1 second of data at 500 Hz
UPDATE_INTERVAL = 1.0   # Check activity every second

class ActivityDetector:
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate

        # Filters
        self.b_notch, self.a_notch = iirnotch(50, 30, sampling_rate)
        self.b_band, self.a_band = butter(4, [0.5 / (sampling_rate / 2), 45.0 / (sampling_rate / 2)], btype='band')
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch) * 0
        self.zi_band = lfilter_zi(self.b_band, self.a_band) * 0
        
        # Data buffer
        self.data_buffer = deque(maxlen=FFT_WINDOW_SIZE)
        
        # FFT setup
        self.freqs = np.fft.rfftfreq(FFT_WINDOW_SIZE, d=1.0/sampling_rate)
        self.fft_window = np.hanning(FFT_WINDOW_SIZE)
        self.window_correction = np.sum(self.fft_window)
        
    def process_sample(self, value):
        """Filter and store incoming sample"""
        # Apply filters
        notch_filtered, self.zi_notch = lfilter(self.b_notch, self.a_notch, [value], zi=self.zi_notch)
        band_filtered, self.zi_band = lfilter(self.b_band, self.a_band, notch_filtered, zi=self.zi_band)
        
        # Store in buffer
        self.data_buffer.append(band_filtered[0])
    
    def calculate_band_power(self, fft_mag, freq_range):
        """Calculate power in a frequency band"""
        low, high = freq_range
        mask = (self.freqs[1:] >= low) & (self.freqs[1:] <= high)
        return np.sum(fft_mag[mask] ** 2)
    
    def get_activity_state(self):
        """Determine if active or inactive based on brainwave bands"""
        if len(self.data_buffer) < FFT_WINDOW_SIZE:
            return None, None
        
        # Compute FFT
        signal = np.array(self.data_buffer, dtype=np.float64)
        windowed = signal * self.fft_window
        fft_result = np.fft.rfft(windowed)
        fft_mag = np.abs(fft_result[1:]) * (2.0 / self.window_correction)
        
        # Calculate band powers
        alpha = self.calculate_band_power(fft_mag, (8, 12))
        beta = self.calculate_band_power(fft_mag, (12, 30))
        
        total = alpha + beta
        if total == 0:
            return None, None
        
        alpha_pct = (alpha / total) * 100
        beta_pct = (beta / total) * 100
        
        # Determine state: Beta dominant = Active, Alpha dominant = Inactive
        if beta_pct > alpha_pct:
            state = "ACTIVE"
        else:
            state = "INACTIVE"
        
        return state, (alpha_pct, beta_pct)

def main():
    print("=== EEG Activity Detector ===")
    print("Searching for LSL stream 'BioAmp_EXG'...\n")
    
    # Connect to LSL stream
    streams = pylsl.resolve_byprop('name', 'BioAmp_EXG', timeout=5)
    if not streams:
        print("❌ No LSL stream found! Make sure the bridge is running.")
        return
    
    inlet = pylsl.StreamInlet(streams[0])
    sampling_rate = int(inlet.info().nominal_srate())
    
    print(f"✓ Connected to LSL stream")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print("\nWaiting for data...\n")
    print("=" * 60)
    
    detector = ActivityDetector(sampling_rate)
    last_check = time.time()
    
    try:
        while True:
            # Pull samples
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample:
                detector.process_sample(sample[0])
            
            # Check activity state every UPDATE_INTERVAL seconds
            if time.time() - last_check >= UPDATE_INTERVAL:
                state, bands = detector.get_activity_state()
                
                if state:
                    alpha_pct, beta_pct = bands
                    
                    # Color-coded output
                    if state == "ACTIVE":
                        status_icon = "🔴"
                        bar_beta = "█" * int(beta_pct / 5)
                        bar_alpha = "░" * int(alpha_pct / 5)
                    else:
                        status_icon = "🟢"
                        bar_beta = "░" * int(beta_pct / 5)
                        bar_alpha = "█" * int(alpha_pct / 5)
                    
                    print(f"{status_icon} {state:8} | Alpha: {alpha_pct:5.1f}% {bar_alpha:20} | Beta: {beta_pct:5.1f}% {bar_beta:20}")
                
                last_check = time.time()
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Stopped.")

if __name__ == "__main__":
    main()