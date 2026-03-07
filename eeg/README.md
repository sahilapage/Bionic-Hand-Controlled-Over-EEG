# EEG Pipeline

Reads EEG data from an Arduino Uno R3 and the BioAmp EXG Pill, streams it over LSL, and then classifie brain state as active or inactive.

## How to run

### Flash the firmware into arduino:

Open `firmware/arduino_data.ino` in Arduino IDE and upload it to the Uno R3

### Start the LSL bridge

```bash
python serial_to_lsl_bridge.py
```

this reads the serial data from the Arduino and creates an LSL stream called `BioAmp_EXG`.

### For signal detection:

Visualizer (EEG waveform + FFT + brainwave bands):
```bash
python visualizer.py
```

Active/inactive state classifier between alpha and beta power:
```bash
python activity.py
```

Active/inactive state classifier using engagement ratio between beta and alpha:
```bash
python engagement_ratio.py
```

## Dependencies

running might require some dependencies:
```bash
pip install pyserial pylsl numpy scipy pyqtgraph PyQt5
```
