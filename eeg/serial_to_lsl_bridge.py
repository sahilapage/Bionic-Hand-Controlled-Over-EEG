import serial
import pylsl
import time
import sys

# Configuration
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAMPLING_RATE = 500
NUM_CHANNELS = 1  

def find_arduino_port():
    """Helper function to list available ports"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    print("\nAvailable serial ports:")
    for port in ports:
        print(f"  {port.device} - {port.description}")
    return None

def main():
    print("=== Serial to LSL Bridge ===")
    print(f"Attempting to connect to {SERIAL_PORT} at {BAUD_RATE} baud...")
    
    # Try to open serial connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"✓ Connected to {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"✗ Failed to open {SERIAL_PORT}: {e}")
        find_arduino_port()
        sys.exit(1)
    
    # Create LSL stream
    info = pylsl.StreamInfo(
        name='BioAmp_EXG',
        type='EEG',
        channel_count=NUM_CHANNELS,
        nominal_srate=SAMPLING_RATE,
        channel_format='float32',
        source_id='bioamp_r3'
    )
    
    # Add channel metadata
    channels = info.desc().append_child("channels")
    for i in range(NUM_CHANNELS):
        ch = channels.append_child("channel")
        ch.append_child_value("label", f"Ch{i+1}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    
    outlet = pylsl.StreamOutlet(info)
    print(f"✓ LSL stream created: 'BioAmp_EXG'")
    print(f"  Channels: {NUM_CHANNELS}")
    print(f"  Sample rate: {SAMPLING_RATE} Hz")
    print("\nStreaming data... Press Ctrl+C to stop\n")
    
    sample_count = 0
    start_time = time.time()
    
    try:
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        value = float(line)
                        outlet.push_sample([value])
                        sample_count += 1
                        
                        # Print status every second
                        if sample_count % SAMPLING_RATE == 0:
                            elapsed = time.time() - start_time
                            actual_rate = sample_count / elapsed
                            print(f"Samples: {sample_count} | Rate: {actual_rate:.1f} Hz | Value: {value:.2f}")
                    
                    except ValueError:
                        # Skip non-numeric lines
                        pass
                        
            except UnicodeDecodeError:
                # Skip malformed data
                pass
                
    except KeyboardInterrupt:
        print("\n\nStopping stream...")
        elapsed = time.time() - start_time
        print(f"Total samples: {sample_count}")
        print(f"Average rate: {sample_count/elapsed:.1f} Hz")
    
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
