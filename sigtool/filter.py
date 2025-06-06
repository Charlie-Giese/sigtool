from scipy.signal import butter, filtfilt


class BaseFilter:
    def apply(self, data, sampling_rate):
        raise NotImplementedError


class LowPassFilter(BaseFilter):
    def __init__(self, cutoff, order=5):
        self.cutoff = cutoff
        self.order = order

    def apply(self, data, sampling_rate):
        nyq = 0.5 * sampling_rate
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
