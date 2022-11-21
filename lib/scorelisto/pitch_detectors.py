import numpy as np
import numpy.typing as npt
import logging
from abc import ABC, abstractmethod
from typing import Tuple

# Get Logger
logger = logging.getLogger(__name__)

# Type hints
FloatNDArray = npt.NDArray[np.float_]


class PitchDetector(ABC):
    def __init__(self, sample_rate_hz: float):
        """
        Recover the pitch from an array.

        :param sample_rate_hz: Sample Rate of the input buffers [Hz]
        """
        self.sample_rate_hz = sample_rate_hz

    @abstractmethod
    def perform(self, audio_buffer: FloatNDArray) -> float:
        """
        Detect the pitch from an input audio buffer.

        :param audio_buffer: Audio buffer (sampled at self.sample_rate_hz)
        :returns: Pitch [Hz], NaN if the pitch is not detected.
        """
        pass


class McLeodDetector(PitchDetector):
    def __init__(self,  sample_rate_hz: float,
                        cutoff: float = 0.97,
                        small_cutoff: float = 0.5,
                        lower_pitch_cutoff: float = 50.0,
                        higher_pitch_cutoff: float = 5000.0):
        """
        Recover the pitch from an array.

        :param sample_rate_hz: Sample Rate of the input buffers [Hz]
        :param cutoff: Sample Rate of the input buffers [Hz]
        :param small_cutoff: Sample Rate of the input buffers [Hz]
        :param lower_pitch_cutoff: Sample Rate of the input buffers [Hz]
        :param higher_pitch_cutoff: Sample Rate of the input buffers [Hz]
        """
        super().__init__(sample_rate_hz)
        self.cutoff = cutoff
        self.small_cutoff = small_cutoff
        self.lower_pitch_cutoff = lower_pitch_cutoff
        self.higher_pitch_cutoff = higher_pitch_cutoff

    def _peak_picking(self, nsdf: FloatNDArray):
        pos = 0
        cur_max_pos = 0
        max_positions = []
        length_nsdf = len(nsdf)
        while (pos < (length_nsdf - 1) / 3) and (nsdf[pos] > 0):
            pos += 1
        while (pos < length_nsdf - 1) and (nsdf[pos] <= 0.0):
            pos += 1
        if pos == 0:
            pos = 1
        while pos < length_nsdf - 1:
            if (nsdf[pos] > nsdf[pos - 1]) and (
                    nsdf[pos] >= nsdf[pos + 1]):
                if cur_max_pos == 0 or\
                   nsdf[pos] > nsdf[cur_max_pos]:
                    cur_max_pos = pos
                elif nsdf[pos] > nsdf[cur_max_pos]:
                    cur_max_pos = pos
            pos += 1
            if pos < length_nsdf - 1 and nsdf[pos] <= 0:
                if cur_max_pos > 0:
                    max_positions.append(cur_max_pos)
                    cur_max_pos = 0
                while pos < length_nsdf - 1 and nsdf[pos] <= 0:
                    pos += 1
        if cur_max_pos > 0:
            max_positions.append(cur_max_pos)
        return max_positions

    def _nsdf(self, audio_buffer: FloatNDArray) -> FloatNDArray:
        audio_buffer -= np.mean(audio_buffer)
        autocorr_f = np.correlate(audio_buffer, audio_buffer, mode='full')
        nsdf = None
        with np.errstate(divide='ignore', invalid='ignore'):
            nsdf = np.true_divide(autocorr_f[int(autocorr_f.size / 2):],
                                    autocorr_f[int(autocorr_f.size / 2)])
            nsdf[nsdf == np.inf] = 0
            nsdf = np.nan_to_num(nsdf)
        return nsdf

    def _parabolic_interpolation(self, nsdf: FloatNDArray, tau: int) -> Tuple[float, float]:
        nsdfa, nsdfb, nsdfc = nsdf[tau - 1 : tau + 2]
        b_value = float(tau)
        bottom = nsdfc + nsdfa - 2 * nsdfb
        if bottom == 0.0:
            turning_point_x = b_value
            turning_point_y = nsdfb
        else:
            delta = nsdfa - nsdfc
            turning_point_x = b_value + delta / (2 * bottom)
            turning_point_y = nsdfb - delta * delta / (8 * bottom)
        return turning_point_x, turning_point_y

    def perform(self, audio_buffer: FloatNDArray) -> float:
        period_estimates = []
        amp_estimates = []
        nsdf = self._nsdf(np.copy(audio_buffer))
        max_positions = self._peak_picking(nsdf)
        highest_amplitude = float('-inf')
        for tau in max_positions:
            highest_amplitude = max(highest_amplitude, nsdf[tau])
            if nsdf[tau] > self.small_cutoff:
                turning_point_x, turning_point_y = self._parabolic_interpolation(nsdf, tau)
                amp_estimates.append(turning_point_y)
                period_estimates.append(turning_point_x)
                highest_amplitude = max(highest_amplitude, turning_point_y)
        if not period_estimates:
            pitch = np.nan
        else:
            actual_cutoff = self.cutoff * highest_amplitude
            period_index = 0
            for i in range(0, len(amp_estimates)):
                if amp_estimates[i] >= actual_cutoff:
                    period_index = i
                    break
            period = period_estimates[period_index]
            pitch_estimate = self.sample_rate_hz / period
            if (pitch_estimate > self.lower_pitch_cutoff) and (pitch_estimate < self.higher_pitch_cutoff):
                pitch = pitch_estimate
            else:
                pitch = np.nan
        return pitch
