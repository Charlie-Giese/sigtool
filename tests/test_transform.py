import numpy as np
from sigtool.transform import FFT


def test_fft_not_crash_and_properties():

    fs = 1000
    f0 = 50
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2 * np.pi * f0 * t)

    fft = FFT(fs)
    freqs, fft_vals = fft.compute_fft(signal)

    assert len(freqs) == fs
    assert len(fft_vals) == fs


def test_fft_magnitude_peak():
    fs = 1000
    f0 = 50
    t = np.linspace(0, 1.0, fs, endpoint=False)
    signal = np.sin(2 * np.pi * f0 * t)

    fft = FFT(fs)
    freqs, fft_vals = fft.compute_fft(signal)
    magnitude = fft.compute_magnitude(fft_vals)

    peak_idx = np.argmax(magnitude)
    peak_freq = freqs[peak_idx]

    assert abs(peak_freq - f0) < 1.0
