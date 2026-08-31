"""Classify brain state as engaged or relaxed from a live LSL EEG stream.

    python -m sohand.eeg.classify                      # engagement ratio
    python -m sohand.eeg.classify --method alpha-beta  # simpler comparison
    python -m sohand.eeg.classify --threshold 1.2      # tune to the wearer

Two methods, because they behave differently on a single unreferenced channel:

  engagement  beta / (alpha + theta), the Pope et al. (1995) index. Smoothed
              over several windows and thresholded. More stable, but the
              threshold is per-person -- calibrate it, do not trust the default.
  alpha-beta  whichever of alpha or beta holds more of the combined power.
              Threshold-free and so needs no calibration, but it flips on noise
              whenever the two are close.

THRESHOLD IS NOT UNIVERSAL. Resting engagement varies severalfold between
people and between electrode placements on the same person. Sit still with eyes
closed for a minute, read the printed ratio, and set the threshold above it.
"""

import argparse
import time
from collections import deque

import numpy as np

from sohand.eeg.bands import Spectrum, StreamFilter, engagement_ratio

STREAM_NAME = "BioAmp_EXG"

# Beyond this the window is muscle or a loose electrode, not EEG. Classifying it
# produces confident nonsense, so it is reported as an artifact instead.
ARTIFACT_UV = 1000.0


class StateDetector:
    """Sliding-window band-power classifier over one channel."""

    def __init__(self, sampling_rate, method="engagement", threshold=0.9,
                 window_size=1024, smoothing=5):
        self.method = method
        self.threshold = threshold
        self.filter = StreamFilter(sampling_rate, num_channels=1)
        self.spectrum = Spectrum(sampling_rate, window_size)
        self.buffer = deque(maxlen=window_size)
        self.history = deque(maxlen=smoothing)

    def push(self, value):
        self.buffer.append(self.filter(value))

    def state(self):
        """(label, detail dict), or (None, None) until the window fills."""
        if len(self.buffer) < self.spectrum.window_size:
            return None, None

        signal = np.asarray(self.buffer, dtype=np.float64)
        if np.max(np.abs(signal)) > ARTIFACT_UV:
            return "ARTIFACT", None

        powers = self.spectrum.band_powers(self.buffer)
        if powers is None:
            return None, None

        if self.method == "alpha-beta":
            total = powers["alpha"] + powers["beta"]
            if total <= 0:
                return None, None
            share = powers["beta"] / total
            label = "ENGAGED" if share > 0.5 else "RELAXED"
            return label, {"score": share, "scale": 1.0, **powers}

        ratio = engagement_ratio(powers)
        self.history.append(ratio)
        smoothed = float(np.mean(self.history))
        label = "ENGAGED" if smoothed > self.threshold else "RELAXED"
        return label, {"score": smoothed, "scale": self.threshold, **powers}


def connect(timeout=5.0):
    # Imported here, not at module scope, so `StateDetector` can be used (and
    # tested) on recorded data without the LSL runtime installed.
    import pylsl

    streams = pylsl.resolve_byprop("name", STREAM_NAME, timeout=timeout)
    if not streams:
        raise SystemExit(
            f"No LSL stream named '{STREAM_NAME}'. Start the bridge first:\n"
            "    python -m sohand.eeg.bridge")
    return pylsl.StreamInlet(streams[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("engagement", "alpha-beta"),
                   default="engagement")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="engagement ratio above which the state reads ENGAGED")
    p.add_argument("--window", type=int, default=1024,
                   help="FFT window in samples; 1024 at 500 Hz is ~2 s")
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between classifications")
    args = p.parse_args()

    inlet = connect()
    rate = int(inlet.info().nominal_srate())
    detector = StateDetector(rate, args.method, args.threshold, args.window)

    print(f"Connected: {rate} Hz, method={args.method}, "
          f"window={args.window} samples ({args.window / rate:.1f} s)")
    if args.method == "engagement":
        print(f"Threshold {args.threshold} -- calibrate this on yourself; see "
              "the module docstring.")
    print("-" * 72)

    last = time.time()
    try:
        while True:
            sample, _ = inlet.pull_sample(timeout=0.1)
            if sample:
                detector.push(sample[0])

            if time.time() - last < args.interval:
                continue
            last = time.time()

            label, detail = detector.state()
            if label == "ARTIFACT":
                print("ARTIFACT  | movement or poor electrode contact")
            elif label:
                bar = "#" * int(min(detail["score"] / detail["scale"] * 12, 30))
                print(f"{label:<8}  | score {detail['score']:6.2f} {bar:<30} "
                      f"a={detail['alpha']:.0f} b={detail['beta']:.0f} "
                      f"t={detail['theta']:.0f}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
