from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from scipy import signal
from .tools import  extract_peak_indexes, \
                    histogram_gaussian_kde, \
                    HysteresisThreshold
import logging
from typing import List
from collections import Counter
from dataclasses import dataclass

# Get Logger
logger = logging.getLogger(__name__)

# Type hints
FloatNDArray = npt.NDArray[np.float_]



class HistogramSplitter:
    def __init__(self, te: float, fwhm: float, histo_step: float):
        self.te = te
        self.fwhm = fwhm
        self.histo_step = histo_step
        self.std_gaussian = (fwhm / histo_step) / 2.355
        nb_points = int(np.ceil(self.std_gaussian * 3 * 2))
        if(not (nb_points % 2)):
            nb_points += 1
        self.gaussian = signal.windows.gaussian(nb_points, self.std_gaussian)
    
    def _getHistogram(self, tonegram: FloatNDArray):
        nb_step_min = int(np.floor(np.min(tonegram) / self.histo_step))
        nb_step_max = int(np.ceil(np.max(tonegram) / self.histo_step))
        MinST = nb_step_min * self.histo_step
        MaxST = nb_step_max * self.histo_step
        histo, x_range = np.histogram(tonegram, bins=nb_step_max - nb_step_min, range=(MinST, MaxST))
        return(histo, x_range)
    
    def _smoothHistogram(self, histogram):
        histo_smooth = signal.convolve(histogram, self.gaussian, mode='same', method='direct')
        return(histo_smooth)

    def _deletePeaksTooLow(self, note_list, nb_samples_min:int):
        new_list = list(note_list)
        new_list.sort(reverse=True)
        while((len(new_list) > 1) and ((new_list[-1][0]) < nb_samples_min)):
            new_list.pop(-1)
        return(new_list)

    def _extractPeaks(self, tonegram: FloatNDArray):
        histo, x_histo = self._getHistogram(tonegram)
        histo_smooth = self._smoothHistogram(histo)
        peak_indexes = extract_peak_indexes(histo_smooth)
        note_heights = x_histo[peak_indexes] + self.histo_step / 2.0
        note_lengths = histo_smooth[peak_indexes]
        note_list = list(zip(note_lengths, note_heights))
        return(note_list)

    def _deletePeaksTooClosed(self, note_list, min_tone_gap:float):
        new_list = list(note_list)
        ind_note = 0
        list_length = len(new_list)
        while(ind_note < (list_length - 1)):
            list_length = len(new_list)
            for ind_note_test in range(list_length - 1, ind_note, -1):
                if(np.abs(new_list[ind_note_test][1]- new_list[ind_note][1]) < min_tone_gap):
                    new_list.pop(ind_note_test)
            ind_note += 1
            list_length = len(new_list)
        return(new_list)
    
    def perform(self, tonegram: FloatNDArray, min_note_length_s:float, min_tone_gap:float):
        note_list = self._extractPeaks(tonegram)
        nb_samples_min = int(np.round(min_note_length_s / self.te))
        note_list = self._deletePeaksTooLow(note_list, nb_samples_min)
        note_list = self._deletePeaksTooClosed(note_list, min_tone_gap)
        return(note_list)


@dataclass
class Group():
    """
    Informations about a group of linked notes
    :param ind_start: Start Index of the group in the tonegram
    :param ind_stop: Stop Index of the group in the tonegram
    :param fit: Signal of size (ind_stop - ind_start) after quantification (recovering the note height).
    """
    ind_start:int
    ind_stop:int # Not included
    fit:FloatNDArray=None

    def getLength(self) -> int:
        return self.ind_stop - self.ind_start


@dataclass
class AnalogNote():
    """
    Class representing an "analog" note.

    Analog note = unormalized height and lengths
    (Height and length correspond to the values measured on the analog signal without any interpretation)

    :param length_s: Length of the note or rest [seconds]
    :param height_tone: Height of the note (None for a rest) [tone]
    :param energy_db: Energy of the note (None for a rest) [dB]
    :param linked_b: Is the note linked to the note before ?
    :param is_a_rest: Is the note actually just a rest ?
    """
    length_s:float
    height_tone:float=None
    energy_db:float=None
    linked_b:bool=False
    is_a_rest:bool=False


class NoteDetector(ABC):

    @abstractmethod
    def perform(self, tonegram: FloatNDArray, time_step_s:float, energygram: FloatNDArray) -> List[AnalogNote]:
        """
        Recover the notes in the signal by finding the right indexes where to split the input signal.

        :param tonegram: Array of tones [tone]
        :param time_step_s: Time step of the input data [seconds].
        :param energygram: Array of energy values.
        :returns: List of :class:`AnalogNote`
        """
        pass


class HistogramNoteDetector(NoteDetector):
    def __init__(self,
            ws_medfilter_s:float=20e-3,
            threshold_energy_ON:float=25.0,
            threshold_energy_OFF:float=30.0,
            min_group_length_s:float=50e-3,
            min_note_length_s:float=100e-3,
            min_tone_gap:float=0.5,
            fwhm_tone_gaussian:float=0.5,
            ):
        """
        :param ws_medfilter_s: Size of the windows of the median filter applied on the tonegram [seconds]
        :param threshold_energy_ON: (Mean Energy - threshold_energy_ON) = value of the threshold (ON) used for the hysteris thresholding on the energy [dB]
        :param threshold_energy_OFF: (Mean Energy - threshold_energy_OFF) = value of the threshold (OFF) used for the hysteris thresholding on the energy [dB]
        :param min_group_length_s: Minimum group length, group lengths under that value will be ignored [seconds]
        :param min_note_length_s: Minimum note length, note lengths under that value will be ignored [seconds]
        :param min_tone_gap: Minimum gap between 2 notes (note change detection) [tone]
        :param fwhm_tone_gaussian: Full width at half maximum of the gaussian used to smooth the histogram [tone]
        :raises ValueError: if threshold_energy_ON >= threshold_energy_OFF
        """
        self.ws_medfilter_s = ws_medfilter_s
        self.threshold_energy_ON = threshold_energy_ON
        self.threshold_energy_OFF = threshold_energy_OFF
        self.min_group_length_s = min_group_length_s
        self.min_note_length_s = min_note_length_s
        self.min_tone_gap = min_tone_gap
        self.fwhm_tone_gaussian = fwhm_tone_gaussian

    def _A_ApplyMedianFilter(self, tonegram: FloatNDArray, time_step_s:float) -> FloatNDArray:
        kernel_size = int(round(self.ws_medfilter_s / time_step_s))
        if(not kernel_size % 2):
            kernel_size += 1
        if(kernel_size < 3):
            logger.warning(f"Ignoring Median filter: the size of the windows is too low (ws_medfilter_s={self.ws_medfilter_s * 1000:.1f} ms)")

        temp = np.where(np.isnan(tonegram), -np.inf, tonegram)
        temp = signal.medfilt(temp, kernel_size=kernel_size)
        temp = np.where(np.isinf(temp), np.nan, temp)
        return temp

    def _B_RemoveLowEnergy(self, tonegram: FloatNDArray, energygram: FloatNDArray) -> FloatNDArray:
        X_histo, histogram_Energy = histogram_gaussian_kde(energygram, bw_method='scott', bins=1000, xmin=np.min(energygram), xmax=np.max(energygram))
        mean_energy = X_histo[np.argmax(histogram_Energy)]
        threshold_ON = mean_energy - self.threshold_energy_ON
        threshold_OFF = mean_energy - self.threshold_energy_OFF
        thresholder = HysteresisThreshold(threshold_ON, threshold_OFF)
        mask = np.logical_not(thresholder.perform(energygram))
        temp = np.where(mask, np.nan, tonegram)
        return temp, threshold_ON, threshold_OFF

    def _C_RemoveGroupsTooShort(self, tonegram: FloatNDArray, time_step_s:float) -> FloatNDArray:
        nb_samples_min = int(np.round(self.min_group_length_s / time_step_s))
        mask = np.isnan(tonegram)
        nb_samples = len(mask)
        val_mask_init = mask[0]
        kInit = 0
        nb_same_value = 1
        for k in range(1, nb_samples):
            if(mask[k] == val_mask_init):
                nb_same_value = nb_same_value + 1
            else:
                if((val_mask_init is False) and (nb_same_value < nb_samples_min)):
                    mask[kInit : k] = True
                nb_same_value = 1
                val_mask_init = mask[k]
                kInit = k
        if((val_mask_init is False) and (nb_same_value < nb_samples_min)):
            mask[kInit : nb_samples] = True
        temp = np.where(mask, np.nan, tonegram)
        return temp

    def _D_DetectGroupsOfNotes(self, tonegram: FloatNDArray) -> List[Group]:
        groups = []
        nb_samples = len(tonegram)
        gate_detected = False
        if(not np.isnan(tonegram[0])):
            gate_detected = True
            kStart = 0
        for k in range(1, nb_samples):
            if((not np.isnan(tonegram[k])) and not gate_detected):
                gate_detected = True
                kStart = k
            if(np.isnan(tonegram[k]) and gate_detected):
                groups.append(Group(kStart, k))
                gate_detected = False
        return groups

    def _E_Detectnotes(self, tonegram: FloatNDArray, time_step_s: float, groups: List[Group]):
        tone_step = 0.01
        splitter = HistogramSplitter(time_step_s, self.fwhm_tone_gaussian, tone_step)
        for grp in groups:
            tonegroup = tonegram[grp.ind_start : grp.ind_stop]
            note_list = splitter.perform(tonegroup, self.min_note_length_s, self.min_tone_gap)
            # Numerized signal
            temp_fit = np.zeros(len(tonegroup))
            note_height_temp = np.array([h for _, h in note_list])
            for k in range(0, len(tonegroup)):
                ind_min = np.argmin(np.abs(tonegroup[k] - note_height_temp))
                temp_fit[k] = note_list[ind_min][1]
            grp.fit = temp_fit

    def _F_FilterFittedHeight(self, groups: List[Group], time_step_s: float):
        ws = int(np.round(self.min_note_length_s / time_step_s))
        for grp in groups:
            N = grp.getLength()
            new_fit = np.zeros(N)
            if(N <= (2 * ws + 1)):
                new_fit[:] = Counter(grp.fit).most_common(1)[0][0]
            else:
                modification = True
                new_fit = np.copy(grp.fit)
                while(modification):
                    modification = False
                    # Beginning of the group
                    occurences = Counter(grp.fit[0 : 2 * ws + 1])
                    most_common = occurences.most_common(1)[0][0]
                    for k in range(0, ws):
                        if(new_fit[k] != most_common):
                            new_fit[k] = most_common
                            modification = True
                    # Middle of the group
                    for k in range(ws, N - ws - 1):
                        occurences = Counter(grp.fit[k - ws : k + ws + 1])
                        most_common = occurences.most_common(1)[0][0]
                        if(new_fit[k] != most_common):
                            new_fit[k] = most_common
                            modification = True
                    # End of the group
                    occurences = Counter(grp.fit[N - 2 * ws - 1 : N])
                    most_common = occurences.most_common(1)[0][0]
                    for k in range(N - ws - 1, N):
                        if(new_fit[k] != most_common):
                            new_fit[k] = most_common
                            modification = True
            grp.fit = new_fit

    def _G_GenerateAnalogPartition(self, groups:List[Group], time_step_s:float, energygram: FloatNDArray = None) -> List[AnalogNote]:
        list_notes = []
        for ind_grp, grp in enumerate(groups):
            if(ind_grp > 0):
                IndGroupBeforeStop = groups[ind_grp - 1].ind_stop
                length_s = (grp.ind_start - IndGroupBeforeStop) * time_step_s
                # Add A rest
                list_notes.append(AnalogNote(length_s, is_a_rest=True))
            N = grp.getLength()
            ind_note_start = 0
            nb_notes = 0
            for ind in range(1, N):
                note_height_before = grp.fit[ind - 1]
                note_height_current = grp.fit[ind]
                if((note_height_current != note_height_before) or (ind == N - 1)):
                    nb_samples_note = ind - ind_note_start
                    if(ind == N - 1):
                        nb_samples_note += 1
                    energy_note = None if (energygram is None) else np.max(energygram[grp.ind_start + ind_note_start : grp.ind_start + ind])
                    length_s = nb_samples_note * time_step_s
                    list_notes.append(AnalogNote(length_s, note_height_before, energy_note, nb_notes > 0))
                    ind_note_start = ind
                    nb_notes += 1
        return(list_notes)

    def perform(self, tonegram: FloatNDArray, time_step_s:float, energygram: FloatNDArray = None) -> List[AnalogNote]:
        temp = self._A_ApplyMedianFilter(tonegram, time_step_s)
        tOn, tOff = None, None
        if energygram is not None:
            temp, tOn, tOff = self._B_RemoveLowEnergy(temp, energygram)
        temp = self._C_RemoveGroupsTooShort(temp, time_step_s)
        groups = self._D_DetectGroupsOfNotes(temp)
        self._E_Detectnotes(temp, time_step_s, groups)
        self._F_FilterFittedHeight(groups, time_step_s)
        list_notes = self._G_GenerateAnalogPartition(groups, time_step_s, energygram)
        # Parameters for debugging
        params = {
            "offset": groups[0].ind_start * time_step_s,
            "tOn": tOn,
            "tOff": tOff
        }
        return list_notes, params
