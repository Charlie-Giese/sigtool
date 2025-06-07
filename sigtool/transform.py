import numpy as np


class FFT:
    def __init__(self, sample_rate, nbins):
        self.sample_rate = sample_rate
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
