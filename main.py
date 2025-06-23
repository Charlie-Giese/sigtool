from sigtool.signal import Signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

sampling_rate = 1000
data = np.sin(2 * np.pi * 200 * np.linspace(0, 1, sampling_rate))
sig = Signal(data, sampling_rate)
sig.apply_window()

fft_result = sig.fft(sampling_rate)

fft_result.plot("power", db=True)
plt.show()
