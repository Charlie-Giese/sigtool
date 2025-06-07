from scipy.signal import firwin, lfilter


class BaseFilter:
    def apply(self, data, sampling_rate):
        raise NotImplementedError


class FIRFilter(BaseFilter):
    def __init__(self, cutoff, numtaps=101, window='hamming'):
        self.cutoff = cutoff
        self.numtaps = numtaps
        self.window = window

    def _design_filter(self, sampling_rate):
        raise NotImplementedError

    def apply(self, data, sampling_rate):
        taps = self._design_filter(sampling_rate)
        return lfilter(taps, 1.0, data)


class LowPassFilter(FIRFilter):
    def _design_filter(self, sampling_rate):
        nyq = 0.5 * sampling_rate
        normal_cutoff = self.cutoff / nyq
        return firwin(self.numtaps, normal_cutoff, window=self.window, pass_zero=True)


class HighPassFilter(FIRFilter):
    def _design_filter(self, sampling_rate):
        nyq = 0.5 * sampling_rate
        normal_cutoff = self.cutoff / nyq
        return firwin(self.numtaps, normal_cutoff, window=self.window, pass_zero=False)


class BandPassFilter(FIRFilter):
    def __init__(self, low_cutoff, high_cutoff, numtaps=101, window='hamming'):
        super().__init__((low_cutoff, high_cutoff), numtaps, window)

    def _design_filter(self, sampling_rate):
        nyq = 0.5 * sampling_rate
        low = self.cutoff[0] / nyq
        high = self.cutoff[1] / nyq
        return firwin(self.numtaps, [low, high], window=self.window, pass_zero=False)
