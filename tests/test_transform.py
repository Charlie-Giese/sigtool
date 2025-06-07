import numpy as np
from sigtool.transform import FFT
from sigtool.signal import Signal


def test_fft_not_crash_and_properties():
    sample_rate = 1000
    data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, sample_rate))
    sig = Signal(data, sample_rate)

    N = 128

    freq, fft_vals = sig.fft(N)

    assert len(freq) == N
    assert len(fft_vals) == N
