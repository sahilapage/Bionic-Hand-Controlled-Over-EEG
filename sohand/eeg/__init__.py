"""EEG acquisition and brain-state classification.

    firmware/bioamp_eeg/   Arduino sketch, 500 Hz single channel over serial
    sohand.eeg.bridge      serial -> LSL stream `BioAmp_EXG`
    sohand.eeg.classify    engaged / relaxed from the band powers
    sohand.eeg.visualizer  live waveform, spectrum and band bars (PyQt5)

The signal chain is one BioAmp EXG Pill into an Arduino Uno R3 ADC. That is a
single-channel, unreferenced, consumer-grade setup: it separates broad states
like eyes-closed alpha from active concentration, and nothing finer. Treat the
output as a gate, not as a decoded intention.
"""

from sohand.eeg.bands import BANDS, Spectrum, StreamFilter, engagement_ratio

__all__ = ["BANDS", "Spectrum", "StreamFilter", "engagement_ratio"]
