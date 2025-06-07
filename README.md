# sigtool

**sigtool** is a lightweight Python 3 library for basic digital signal processing (DSP), built with simplicity and clean structure in mind. It is intended as a demonstration of good object-oriented programming practices in Python, particularly for use cases where Python is an auxiliary tool in a systems-level environment.

## Features

- 🧰 FIR-based low-pass, high-pass, and band-pass filters using `scipy.signal.firwin`
- ⚡ Fast Fourier Transform (FFT) with support for magnitude, power, and phase spectra
- 🧪 Pytest-based testing setup with optional coverage
- 📦 Clean Python package layout using `pyproject.toml`

## Installation

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd sigtool_project
pip install -e .[dev]
```

## Usage

Example usage of filtering and FFT:

```python
from sigtool.signal import Signal
from sigtool.filter import LowPassFilter
from sigtool.transform import FFT

# Create a signal
samples = ...  # e.g., from np.sin(...)
sampling_rate = 1000
sig = Signal(samples, sampling_rate)

# Apply filter
lpf = LowPassFilter(cutoff=100)
filtered = lpf.apply(sig.samples, sig.sampling_rate)

# Compute FFT
fft = FFT(sig.sampling_rate)
freqs, spectrum = fft.compute_fft(filtered)
```

## License

MIT License

