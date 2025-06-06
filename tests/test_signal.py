import numpy as np
from sigtool.signal import Signal


def test_signal_from_array_and_properties():
    sampling_rate = 1000
    data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, sampling_rate))
    sig = Signal(data, sampling_rate)

    assert len(sig.data) == sampling_rate
    assert sig.sampling_rate == sampling_rate
    assert np.isclose(sig.time[-1], (sampling_rate - 1) / sampling_rate)


def test_signal_apply_filter_does_not_crash():
    from sigtool.filter import LowPassFilter
    data = np.random.randn(1000)
    sig = Signal(data, 1000)
    filt = LowPassFilter(cutoff=100)
    sig.apply_filter(filt)
    assert len(sig.data) == 1000
