from abc import ABC, abstractmethod
import logging
import numpy as np
import numpy.typing as npt
from typing import Tuple
import librosa

# Get Logger
logger = logging.getLogger(__name__)

# Type hints
FloatNDArray = npt.NDArray[np.float_]


class AudioReader(ABC):
    def __init__(self):
        """
        Open an audio file and extract the data samples and parameters (sample rate, number of samples).

        Audio is automatically converted to mono.
        """
        self.filepath: str = None
        self.data = None
        self.nb_samples: int = None
        self.fe: float = None
        self.te: float = None
        
    @abstractmethod
    def read(self, filepath: str, sample_rate_hz: int = None) -> None:
        """
        Extract the audio frames and its parameters to the class attributes.
        The audio frames must be converted to mono.
        (self.data, self.nb_samples, self.fe, self.te)

        :param filepath: Path to the audio file.
        :param sample_rate_hz: Sample rate of the audio to acquire (if None, it keeps the original audio sample rate)
        """
        pass

    def _getClosestIndex(self, time_s: float) -> int:
        """
        Get the closest index of the data corresponding to the input time.

        :param time_s: Time in seconds
        :returns: Index
        """
        ind = int(round(time_s / self.te))
        if(ind < 0):
            ind = 0
        elif(ind > self.nb_samples - 1):
            ind = self.nb_samples - 1
        return(ind)

    def _getStartStopIndexes(self, time_start_s: float = None, time_stop_s: float = None) -> Tuple[int, int]:
        """
        Get the indexes of the data corresponding to the start time and stop time provided.
        If time_start_s is None, index (0) is returned
        If time_stop_s is None, index (self.nb_samples - 1) is returned.

        :param time_start_s: Start Time [seconds]
        :param time_stop_s: Stop Time [seconds]
        :returns: (index_start, index_stop)
        """
        if(time_stop_s is None):
            ind_stop = self.nb_samples - 1
        else:
            ind_stop = self._getClosestIndex(time_stop_s)
        if(time_start_s is None):
            ind_start = 0
        else:
            ind_start = self._getClosestIndex(time_start_s)
        if((ind_stop > self.nb_samples) or (ind_start > ind_stop) or (ind_stop < 1) or (ind_start < 0)):
            raise ValueError("Invalid Indexes ==> Start Index:%.1f ms  Stop Index:%.1f ms" % (time_start_s * 1e3, time_stop_s * 1e3))
        return((ind_start, ind_stop))

    def iterate(self, windows_size_s: float, time_step_s: float, time_start_s: float = None, time_stop_s: float=None) -> FloatNDArray:
        """
        Iterate through the audio data with the given windows size and time step.

        The time where to start and stop can also be provided.

        :param windows_size_s: Windows size [seconds]
        :param time_step_s: Time step [seconds]
        :param time_start_s: (Optional) Start time [seconds]
        :param time_stop_s: (Optional) Stop time [seconds]
        :returns: Data chunk
        """
        ind_start, ind_stop = self._getStartStopIndexes(time_start_s, time_stop_s)
        windows_size_spl = int(np.round(windows_size_s / self.te))
        nb_chunks = int((ind_stop - ind_start - windows_size_spl) * (self.te / time_step_s))
        for ind_chunk in range(nb_chunks):
            ind_inf = round((ind_chunk * time_step_s) / self.te) + ind_start
            ind_sup = ind_inf + windows_size_spl
            yield self.data[ind_inf : ind_sup]

    def getSampleRate(self) -> float:
        """
        Get the sample rate of the data in Hertz.
        """
        return self.fe

    def getData(self) -> FloatNDArray:
        """
        Get the data audio array (always mono).
        """
        return self.data


class LibrosaReader(AudioReader):
    def read(self, filepath: str, sample_rate_hz: int = None) -> None:
        self.data, fs = librosa.load(filepath, sr=sample_rate_hz, mono=True)
        self.fe = float(fs)
        self.te = 1 / self.fe
        self.nb_samples = len(self.data)

