"""Filtering and band-power analysis, shared by every EEG consumer here.

Three modules used to carry their own copy of this code with slightly different
constants -- one used an 8-12 Hz alpha band and another 8-13 Hz, so the two
classifiers disagreed on identical input. There is one definition now.
"""

from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

# Standard clinical bands, Hz. Alpha rises when the eyes close or attention
# drops; beta tracks active concentration. The ratio between them is the only
# thing a single unreferenced electrode measures reliably.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

MAINS_HZ = 50.0          # set to 60.0 in North America
MAINS_Q = 30.0
PASSBAND_HZ = (0.5, 45.0)


class StreamFilter:
    """Causal notch + bandpass, one independent filter state per channel.

    Sample-at-a-time rather than block-at-a-time because the display and the
    classifier both consume the stream live; carrying `zi` forward is what
    keeps the output continuous across calls.
    """

    def __init__(self, sampling_rate, num_channels=1,
                 mains_hz=MAINS_HZ, passband=PASSBAND_HZ):
        nyquist = sampling_rate / 2.0
        self.b_notch, self.a_notch = iirnotch(mains_hz, MAINS_Q, sampling_rate)
        self.b_band, self.a_band = butter(
            4, [passband[0] / nyquist, passband[1] / nyquist], btype="band")
        # Zero initial conditions, not steady-state: the stream starts at rest,
        # and `lfilter_zi` scaled by the first sample would inject a step.
        self.zi_notch = [lfilter_zi(self.b_notch, self.a_notch) * 0
                         for _ in range(num_channels)]
        self.zi_band = [lfilter_zi(self.b_band, self.a_band) * 0
                        for _ in range(num_channels)]

    def __call__(self, value, channel=0):
        notched, self.zi_notch[channel] = lfilter(
            self.b_notch, self.a_notch, [value], zi=self.zi_notch[channel])
        banded, self.zi_band[channel] = lfilter(
            self.b_band, self.a_band, notched, zi=self.zi_band[channel])
        return float(banded[-1])


class Spectrum:
    """Hann-windowed rFFT over a sliding window, plus band powers.

    Window length trades resolution against latency: 512 samples at 500 Hz is
    ~1 s and just under 1 Hz per bin, enough to separate alpha from beta.
    """

    def __init__(self, sampling_rate, window_size=512, smoothing=1):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)[1:]
        self.window = np.hanning(window_size)
        self._correction = float(np.sum(self.window))
        self._history = deque(maxlen=max(smoothing, 1))

    @property
    def resolution_hz(self):
        return self.sampling_rate / self.window_size

    def magnitude(self, samples):
        """Amplitude spectrum of the newest `window_size` samples, DC dropped.

        Returns None until the buffer has filled -- a short window would be
        zero-padded and read as a spurious low-frequency component.
        """
        if len(samples) < self.window_size:
            return None
        chunk = np.asarray(samples, dtype=np.float64)[-self.window_size:]
        spectrum = np.fft.rfft(chunk * self.window)
        mag = np.abs(spectrum[1:]) * (2.0 / self._correction)
        self._history.append(mag)
        return np.mean(self._history, axis=0)

    def band_power(self, magnitude, band):
        low, high = BANDS[band] if isinstance(band, str) else band
        mask = (self.freqs >= low) & (self.freqs <= high)
        return float(np.sum(magnitude[mask] ** 2))

    def band_powers(self, samples, relative=False):
        """Power in each named band, or None if the window is not full yet."""
        mag = self.magnitude(samples)
        if mag is None:
            return None
        powers = {name: self.band_power(mag, name) for name in BANDS}
        if relative:
            total = sum(powers.values())
            if total <= 0:
                return None
            powers = {k: v / total for k, v in powers.items()}
        return powers

    def peak_frequency(self, magnitude, above_hz=2.0):
        """Strongest frequency above `above_hz`, skipping drift and DC leakage."""
        mask = self.freqs >= above_hz
        if not mask.any():
            return float("nan")
        return float(self.freqs[mask][int(np.argmax(magnitude[mask]))])


def engagement_ratio(powers):
    """beta / (alpha + theta) -- the standard EEG engagement index.

    Pope et al. (1995) proposed it for adaptive automation; it is more stable
    than a bare alpha/beta comparison because a drop in attention raises theta
    as well as alpha, so both move the denominator the same way.
    """
    return powers["beta"] / (powers["alpha"] + powers["theta"] + 1e-9)
