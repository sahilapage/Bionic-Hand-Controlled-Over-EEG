import pylsl
import numpy as np
from collections import deque
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
import time

FFT_WINDOW_SIZE = 1024  # Increased for better accuracy
UPDATE_INTERVAL = 2.0    # Check every 2 seconds (more stable)

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
        
        # Smoothing for stability
        self.engagement_history = deque(maxlen=5)  # Smooth over 5 readings
        
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
        """Determine activity using engagement ratio"""
        if len(self.data_buffer) < FFT_WINDOW_SIZE:
            return None, None
        
        # Compute FFT
        signal = np.array(self.data_buffer, dtype=np.float64)
        
        # Reject if signal has too much artifact (very large values)
        if np.max(np.abs(signal)) > 1000:  # Artifact threshold
            return "ARTIFACT", None
        
        windowed = signal * self.fft_window
        fft_result = np.fft.rfft(windowed)
        fft_mag = np.abs(fft_result[1:]) * (2.0 / self.window_correction)
        
        # Calculate band powers
        theta = self.calculate_band_power(fft_mag, (4, 8))
        alpha = self.calculate_band_power(fft_mag, (8, 13))   # Extended alpha range
        beta = self.calculate_band_power(fft_mag, (13, 30))
        
        # Engagement ratio method
        engagement = beta / (alpha + theta + 0.0001)
        
        # Smooth the engagement value
        self.engagement_history.append(engagement)
        smoothed_engagement = np.mean(self.engagement_history)
        
        # Adjusted threshold - TUNE THIS VALUE
        # Lower = more sensitive to detecting ACTIVE
        # Higher = more likely to show INACTIVE
        THRESHOLD = 0.9  # Try 1.2 - 2.0 to find what works for you
        
        if smoothed_engagement > THRESHOLD:
            state = "ACTIVE"
        else:
            state = "INACTIVE"
        
        return state, (smoothed_engagement, alpha, beta, theta)

def main():
    print("=== EEG Activity Detector v2 (Engagement Ratio) ===")
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
    print(f"\n⚠️  Make sure electrodes are firmly attached!")
    print(f"⚠️  Threshold: 1.5 (adjust if needed)\n")
    print("Waiting for data...\n")
    print("=" * 70)
    
    detector = ActivityDetector(sampling_rate)
    last_check = time.time()
    
    try:
        while True:
            # Pull samples
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample:
                detector.process_sample(sample[0])
            
            # Check activity state
            if time.time() - last_check >= UPDATE_INTERVAL:
                state, data = detector.get_activity_state()
                
                if state == "ARTIFACT":
                    print("⚠️  ARTIFACT   | Movement or poor electrode contact detected")
                elif state and data:
                    engagement, alpha, beta, theta = data
                    
                    if state == "ACTIVE":
                        icon = "🔴"
                        bar = "█" * int(min(engagement * 8, 25))
                    else:
                        icon = "🟢"
                        bar = "░" * int(min(engagement * 8, 25))
                    
                    print(f"{icon} {state:8} | Engagement: {engagement:.2f} {bar:25} | A:{alpha:.0f} B:{beta:.0f} T:{theta:.0f}")
                
                last_check = time.time()
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Stopped.")
        print("\n💡 Tip: If results seem wrong, try adjusting THRESHOLD")
        print("   - Increase (1.8-2.0) if showing ACTIVE too often")
        print("   - Decrease (1.0-1.2) if showing INACTIVE too often")

if __name__ == "__main__":
    main()