# EEG

Single-channel EEG acquisition, LSL streaming, and a two-state engaged/relaxed
classifier.

```
BioAmp EXG Pill → Arduino Uno R3 (500 Hz) → serial → LSL "BioAmp_EXG"
                                                        ├── classify
                                                        └── visualizer
```

## What this can and cannot do

One unreferenced electrode into a 10-bit ADC separates broad states — eyes-closed
alpha, active concentration — and nothing finer. It does not decode intent, or
which finger you are thinking about, or anything with more than a couple of bits
of information in it. Treat the output as a **gate**, not as a command.

## Hardware

| part | note |
|---|---|
| BioAmp EXG Pill | ships at 1000× gain; if you re-solder the gain resistor, update `BIOAMP_GAIN` in the sketch |
| Arduino Uno R3 | signal on A0, 10-bit ADC against the 5 V rail |
| electrodes | forehead / mastoid works; contact quality dominates everything downstream |

## Run it

**1. Flash the firmware.** Open `firmware/bioamp_eeg/bioamp_eeg.ino` in the
Arduino IDE and upload it to the Uno. It prints one microvolt reading per line
at 500 Hz.

**2. Bridge serial to LSL.**

```bash
python -m sohand.eeg.bridge                 # /dev/ttyACM0 at 115200
python -m sohand.eeg.bridge --list          # if you are not sure of the port
```

Watch the *measured* rate it prints. If it drifts below 500 Hz the FFT bin
centres are wrong and every band power shifts with them.

**3. Look at the signal, or classify it.**

```bash
python -m sohand.eeg.visualizer             # waveform + spectrum + band bars
python -m sohand.eeg.classify               # engaged / relaxed
python -m sohand.eeg.classify --method alpha-beta
```

## Signal chain

Every consumer shares `sohand.eeg.bands`, so the visualizer shows exactly what
the classifier decides on. Previously each module carried its own copy of the
DSP with slightly different constants — one used 8–12 Hz for alpha and another
8–13 Hz — and the two disagreed on identical input.

1. **Notch at 50 Hz** (Q = 30) to kill mains hum. Set `MAINS_HZ = 60.0` in
   North America.
2. **Bandpass 0.5–45 Hz**, 4th-order Butterworth: removes electrode DC drift
   below and everything above the gamma band.
3. **Hann-windowed rFFT** over a sliding window. 512 samples at 500 Hz is ~1 s
   and 0.98 Hz per bin, which is enough to separate alpha from beta.
4. **Band power** is the summed squared magnitude within each band.

Filtering is causal and sample-at-a-time, carrying `zi` forward, so the output
stays continuous across calls. Initial conditions are zero rather than
steady-state — the stream starts at rest, and scaling `lfilter_zi` by the first
sample would inject a step.

| band | Hz | rises when |
|---|---|---|
| delta | 0.5–4 | deep sleep — mostly an artifact indicator while awake |
| theta | 4–8 | drowsiness, low attentional load |
| alpha | 8–13 | eyes closed, relaxed wakefulness |
| beta | 13–30 | active concentration |
| gamma | 30–45 | largely muscle at this electrode count |

## Classification

**Engagement ratio** (default), from Pope et al. (1995):

```
engagement = beta / (alpha + theta)
```

Smoothed over five windows and thresholded. More stable than a bare alpha/beta
comparison because a drop in attention raises theta as well as alpha, so both
move the denominator the same way.

**Alpha–beta** (`--method alpha-beta`): whichever of the two holds more of the
combined power. No threshold to tune, so nothing to calibrate — but it flips on
noise whenever the two are close.

### The threshold is not universal

Resting engagement varies severalfold between people, and between electrode
placements on the same person. The 0.9 default is a starting point, not a
constant. Sit still with your eyes closed for a minute, read the printed ratio,
and set `--threshold` above it.

Windows whose peak-to-peak exceeds 1000 µV are reported as `ARTIFACT` rather
than classified — that is a jaw clench or a loose electrode, and classifying it
produces confident nonsense.

## Install

```bash
pip install -e ".[eeg]"
```

## Reference

Pope, Bogart & Bartolome, **Biocybernetic system evaluates indices of operator
engagement in automated task**, Biological Psychology 40(1–2), 1995.
