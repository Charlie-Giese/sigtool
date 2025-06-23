import numpy as np
from enum import Enum, auto
import matplotlib.pyplot as plt


class FFT:
    def __init__(self, sample_rate, nbins=None):
        self.sample_rate = sample_rate
        if nbins is None:
            self.nbins = sample_rate
        else:
            self.nbins = nbins

    def compute_fft(self, signal):
        """Return the FFT of the input signal and the corresponding frequency bins"""
        N = self.nbins
        fft_vals = np.fft.fft(signal, n=N)
        freqs = np.fft.fftfreq(N, d=1/self.sample_rate)
        return freqs, fft_vals

    def compute_magnitude(self, fft_vals):
        """Return the magnitude spectrum"""
        return np.abs(fft_vals)

    def compute_power(self, fft_vals):
        """Return the power spectrum"""
        return np.abs(fft_vals) ** 2

    def compute_phase(self, fft_vals):
        """Return the phase spectrum"""
        return np.angle(fft_vals)


class FFTPlotMode(Enum):
    COMPLEX = auto()
    MAGNITUDE = auto()
    POWER = auto()
    PHASE = auto()


class FFTResult:
    def __init__(self, freqs, fft_vals, fft_impl, window_type=None):
        self.freqs = freqs
        self.fft_vals = fft_vals
        self._fft = fft_impl
        self.window_type = window_type

    def power(self):
        return self._fft.compute_power(self.fft_vals)

    def magnitude(self):
        return self._fft.compute_magnitude(self.fft_vals)

    def phase(self):
        return self._fft.compute_phase(self.fft_vals)

    def plot(self,
             mode: FFTPlotMode = FFTPlotMode.MAGNITUDE,
             xlabel="Frequency (Hz)",
             db: bool = False,
             **kwargs):
        if isinstance(mode, str):
            mode = FFTPlotMode[mode.upper()]

        freqs = np.fft.fftshift(self.freqs)
        data = np.fft.fftshift(self.fft_vals)

        if mode == FFTPlotMode.COMPLEX:
            y = data
            ylabel = "Real / Imag"
            plt.plot(self.freqs, y.real, label="Real", c='k', **kwargs)
            plt.plot(self.freqs, y.imag, label="Imag", c='r', **kwargs)
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(ylabel)
            plt.title("FFT: Complex Components")
            plt.grid(True)
            plt.tight_layout()
            return
        elif mode == FFTPlotMode.MAGNITUDE:
            y = self._fft.compute_magnitude(data)
            ylabel = "Magnitude"
            if db:
                y = 20 * np.log10(y + 1e-12)
                ylabel = "Magnitude (dB)"
        elif mode == FFTPlotMode.PHASE:
            y = self._fft.compute_phase(data)
            ylabel = "Phase (radians)"
        elif mode == FFTPlotMode.POWER:
            y = self._fft.compute_power(data)
            ylabel = "Power"
            if db:
                y = 10 * np.log10(y + 1e-12)
                ylabel = "Power (dB)"
        else:
            raise ValueError(f"Unsupported plot mode: {mode}")

        plt.plot(freqs, y, c='k', **kwargs)
        plt.title(f"FFT {mode.name.capitalize()} Spectrum" +
                  (f" (window: {self.window_type})" if self.window_type else ""))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
