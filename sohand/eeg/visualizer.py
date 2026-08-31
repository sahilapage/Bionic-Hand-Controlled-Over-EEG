"""Live EEG monitor: waveform, spectrum and relative band powers.

    python -m sohand.eeg.visualizer

Subscribes to the LSL stream published by `sohand.eeg.bridge` and draws, per
selected channel, the filtered waveform over the last few seconds and its
amplitude spectrum, plus a bar chart of relative band power for one channel.

Filtering and FFT come from `sohand.eeg.bands`, so what you see here is exactly
what `sohand.eeg.classify` decides on -- when the two disagreed it was because
each carried its own copy of the DSP.

Needs PyQt5 and pyqtgraph.
"""

import sys
import time
from collections import deque

import numpy as np
import pylsl
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFrame, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
                             QWidget)
from pyqtgraph import PlotWidget

from sohand.eeg.bands import BANDS, Spectrum, StreamFilter

# ~1 s of data at 500 Hz: just under 1 Hz per bin, which separates alpha from
# beta. A longer window sharpens the spectrum but lags the display.
FFT_WINDOW_SIZE = 512
# Averaging this many spectra steadies the plot at the cost of responsiveness.
SMOOTHING_WINDOW_SIZE = 10
DISPLAY_SECONDS = 4
REFRESH_MS = 20
STALE_STREAM_S = 2.0

CHANNEL_COLORS = ["#FF0054", "#00FF8C", "#AA42FF", "#00FF47",
                  "#FF8C19", "#FF00FF", "#00FFFF", "#FFFF00"]


class ChannelBuffers:
    """Per-channel filtering plus the two ring buffers the plots read from."""

    def __init__(self, num_channels, sampling_rate):
        self.num_channels = num_channels
        self.filter = StreamFilter(sampling_rate, num_channels)
        size = DISPLAY_SECONDS * sampling_rate
        self.waveform = [np.zeros(size) for _ in range(num_channels)]
        self.write_idx = [0] * num_channels
        self.fft_window = [deque(maxlen=FFT_WINDOW_SIZE)
                           for _ in range(num_channels)]

    def push(self, sample):
        for ch in range(self.num_channels):
            value = self.filter(sample[ch], channel=ch)
            self.waveform[ch][self.write_idx[ch]] = value
            self.write_idx[ch] = (self.write_idx[ch] + 1) % len(self.waveform[ch])
            self.fft_window[ch].append(value)

    def display(self, channel):
        """The ring buffer unrolled so the oldest sample is first."""
        idx = self.write_idx[channel]
        buf = self.waveform[channel]
        return np.concatenate([buf[idx:], buf[:idx]])


class ChannelSettings(QDialog):
    """Which channels to plot, and which one drives the band-power bars."""

    def __init__(self, num_channels, selected, band_channel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channels")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Waveform and spectrum channels:"))
        self.checkboxes = []
        for i in range(num_channels):
            box = QCheckBox(f"Channel {i + 1}")
            box.setChecked(i in selected)
            self.checkboxes.append(box)
            layout.addWidget(box)

        layout.addWidget(QLabel("\nBand-power channel:"))
        self.combo = QComboBox()
        for i in range(num_channels):
            self.combo.addItem(f"Channel {i + 1}")
        self.combo.setCurrentIndex(band_channel)
        layout.addWidget(self.combo)

        ok = QPushButton("OK")
        ok.clicked.connect(self._accept)
        layout.addWidget(ok)
        self.setLayout(layout)

    def _accept(self):
        # Deselecting every channel would leave an empty window with no way
        # back, so keep at least one.
        if not any(box.isChecked() for box in self.checkboxes):
            self.checkboxes[0].setChecked(True)
        self.accept()

    def selection(self):
        return ([i for i, b in enumerate(self.checkboxes) if b.isChecked()],
                self.combo.currentIndex())


class EEGMonitor(QMainWindow):
    def __init__(self, inlet):
        super().__init__()
        self.setWindowTitle("EEG monitor -- waveform, spectrum, band power")
        self.setGeometry(100, 100, 1200, 800)

        self.inlet = inlet
        info = inlet.info()
        self.sampling_rate = int(info.nominal_srate())
        self.num_channels = info.channel_count()

        self.buffers = ChannelBuffers(self.num_channels, self.sampling_rate)
        self.spectra = [Spectrum(self.sampling_rate, FFT_WINDOW_SIZE,
                                 smoothing=SMOOTHING_WINDOW_SIZE)
                        for _ in range(self.num_channels)]
        self.selected = list(range(self.num_channels))
        self.band_channel = 0
        self.last_sample_time = None

        self._build_ui()
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(REFRESH_MS)

    # -- layout -------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.wave_container = QWidget()
        self.wave_layout = QVBoxLayout(self.wave_container)
        self.wave_layout.setSpacing(0)
        scroll.setWidget(self.wave_container)
        left_layout.addWidget(scroll)

        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: rgba(50,50,50,150);"
                            " border: 1px solid #888; border-radius: 5px; }")
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(5, 5, 5, 5)
        button = QPushButton("Channels")
        button.setStyleSheet(
            "QPushButton { background-color:#2c3e50; color:#ecf0f1; border:none;"
            " border-radius:3px; padding:4px 10px; }"
            "QPushButton:hover { background-color:#34495e; }")
        button.clicked.connect(self.show_settings)
        frame_layout.addWidget(button, alignment=Qt.AlignRight)
        left_layout.addWidget(frame, alignment=Qt.AlignRight)
        root.addWidget(left, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.fft_plot = PlotWidget()
        self.fft_plot.setBackground("black")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_plot.setLabel("bottom", "Frequency (Hz)")
        self.fft_plot.setLabel("left", "Amplitude (uV)")
        self.fft_plot.setXRange(0, 50, padding=0)
        right_layout.addWidget(self.fft_plot, stretch=1)

        self.band_plot = pg.PlotWidget()
        self.band_plot.setBackground("black")
        self.band_plot.showGrid(x=True, y=True, alpha=0.3)
        self.band_plot.setLabel("left", "Relative power")
        self.band_plot.setXRange(-0.5, len(BANDS) - 0.5)
        self.band_plot.setYRange(0, 1)
        self.band_plot.setMouseEnabled(x=False, y=False)
        self.band_names = list(BANDS)
        self.band_bars = pg.BarGraphItem(
            x=list(range(len(BANDS))), height=[0] * len(BANDS), width=0.5,
            brushes=[pg.mkBrush(c) for c in CHANNEL_COLORS[:len(BANDS)]])
        self.band_plot.addItem(self.band_bars)
        self.band_plot.getAxis("bottom").setTicks(
            [[(i, n.capitalize()) for i, n in enumerate(self.band_names)]])
        right_layout.addWidget(self.band_plot, stretch=1)

        root.addWidget(right, stretch=1)
        self.setCentralWidget(central)
        self._build_curves()

    def _build_curves(self):
        self.wave_plots, self.wave_curves, self.fft_curves = [], [], []
        for ch in range(self.num_channels):
            plot = PlotWidget()
            plot.setBackground("black")
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setLabel("left", f"Ch {ch + 1}", color="white")
            plot.getAxis("left").setTextPen("white")
            plot.getAxis("bottom").setTextPen("white")
            plot.setYRange(-5000, 5000, padding=0)
            plot.setXRange(0, DISPLAY_SECONDS, padding=0)
            plot.setMouseEnabled(x=False, y=True)
            plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            pen = pg.mkPen(color=CHANNEL_COLORS[ch % len(CHANNEL_COLORS)], width=2)
            self.wave_curves.append(plot.plot(pen=pen))
            self.wave_plots.append(plot)
            self.wave_layout.addWidget(plot)
            self.fft_curves.append(self.fft_plot.plot(pen=pen))
        self._apply_visibility()

    def _apply_visibility(self):
        for ch in range(self.num_channels):
            visible = ch in self.selected
            self.wave_plots[ch].setVisible(visible)
            self.fft_curves[ch].setVisible(visible)
            if visible:
                idx = self.wave_layout.indexOf(self.wave_plots[ch])
                self.wave_layout.setStretch(idx, 1)

    def show_settings(self):
        dialog = ChannelSettings(self.num_channels, self.selected,
                                 self.band_channel, self)
        if dialog.exec_():
            self.selected, self.band_channel = dialog.selection()
            self._apply_visibility()

    # -- update loop --------------------------------------------------------
    def update_plots(self):
        samples, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=50)
        if not samples:
            stale = (self.last_sample_time is not None
                     and time.time() - self.last_sample_time > STALE_STREAM_S)
            if stale:
                print("LSL stream went silent -- closing.")
                self.timer.stop()
                self.close()
            return

        self.last_sample_time = time.time()
        for sample in samples:
            self.buffers.push(sample)

        time_axis = np.linspace(0, DISPLAY_SECONDS,
                                len(self.buffers.waveform[0]))
        for ch in self.selected:
            self.wave_curves[ch].setData(time_axis, self.buffers.display(ch))
            mag = self.spectra[ch].magnitude(self.buffers.fft_window[ch])
            if mag is not None:
                self.fft_curves[ch].setData(self.spectra[ch].freqs, mag)

        ch = self.band_channel
        powers = self.spectra[ch].band_powers(self.buffers.fft_window[ch],
                                              relative=True)
        if powers is not None:
            self.band_bars.setOpts(height=[powers[n] for n in self.band_names])


def main():
    streams = pylsl.resolve_streams()
    if not streams:
        raise SystemExit("No LSL streams found. Start the bridge first:\n"
                         "    python -m sohand.eeg.bridge")
    inlet = pylsl.StreamInlet(streams[0])
    info = inlet.info()
    print(f"Connected to '{info.name()}': {info.channel_count()} ch at "
          f"{info.nominal_srate()} Hz")

    app = QApplication(sys.argv)
    window = EEGMonitor(inlet)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
