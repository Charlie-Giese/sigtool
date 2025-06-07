import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sigtool.transform import FFT, FFTResult


class Signal:
    def __init__(self, data: np.array, sampling_rate: float):
        self.data = data
        self.sampling_rate = sampling_rate
        self.time = np.arange(len(data)) / sampling_rate

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

    def fft(self, nbins):
        fft = FFT(self.sampling_rate, nbins)
        freqs, fft_vals = fft.compute_fft(self.data)
        return FFTResult(freqs, fft_vals, fft)
