import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import matplotlib.pyplot as plt
from sigtool.transform import FFT, FFTResult


class Signal:
    def __init__(self, data: np.array, sampling_rate: float):
        self.data = data
        self.sampling_rate = sampling_rate
        self.time = np.arange(len(data)) / sampling_rate
        self.window_type = None
        self.windowed_data = None

    @classmethod
    def from_csv(cls, path: str, sampling_rate: float):
        data = np.loadtxt(path, delimiter=",")
        return cls(data, sampling_rate)

    @classmethod
    def from_wav(cls, path: str):
        sampling_rate, data = wavfile.read(path)
        return cls(data.astype(np.float32), sampling_rate)

    def to_csv(self, path: str):
        np.savetxt(path, self.data, delimiter=",")

    def to_wave(self, path: str):
        wavfile.write(path, int(self.sampling_rate),
                      self.data.astype(np.int16))

    def plot(self, title="Signal"):
        plt.plot(self.time, self.data)
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def apply_filter(self, filter_obj):
        self.data = filter_obj.apply(self.data, self.sampling_rate)

    def fft(self, nbins, use_window: bool = True) -> FFTResult:
        fft = FFT(self.sampling_rate, nbins)
        data = self.windowed_data if use_window and self.windowed_data is not None else self.data
        freqs, fft_vals = fft.compute_fft(data)
        return FFTResult(freqs, fft_vals, fft, self.window_type)

    def apply_window(self, window_type: str = "hann"):
        """Apply a window function to the signal data."""
        if self.windowed_data is not None:
            raise ValueError("Window already applied")

        if len(self.data) < 2:
            raise ValueError("Signal too short for windowing")

        try:
            window = get_window(window_type, len(self.data))
        except ValueError:
            raise ValueError(f"Invalid window type: '{window_type}'")

        self.windowed_data = self.data * window
        self.window_type = window_type
        return self.windowed_data
