import numpy as np
from scipy.stats import gaussian_kde

F0_STANDARD = 32.7032

# Convert Tone & Frequency
def tone2freq(tone, f0=F0_STANDARD):
    return(f0 * np.power(2.0, tone / 12.0))

def freq2tone(freq, f0=F0_STANDARD):
    return(np.log10(freq / f0) * 12.0 / np.log10(2.0))


# Convert Tone & Note+Octave
def tone2note(semitone):
    height = semitone % 12
    octave = int(semitone / 12.0)
    return((height, octave))

def note2tone(height, octave):
    semitone = octave * 12 + height
    return(semitone)


def compute_energy(signal):
    energy = np.sum(np.power(signal, 2, dtype='float64')) / len(signal)
    return(energy)

def extract_peak_indexes(signal):
    peak_indexes = []
    StatutNoteAV = "DOWN"
    kStart = None
    for k in range(1, len(signal)):
        if(signal[k] > signal[k - 1]):
            StatutNoteAP = "UP"
        elif(signal[k] == signal[k - 1]):
            StatutNoteAP = "CONST"
        else:
            StatutNoteAP = "DOWN"
        
        if((StatutNoteAV=="UP") and (StatutNoteAP=="DOWN")):
            peak_indexes.append(k - 1)
        
        if((StatutNoteAV=="UP") and (StatutNoteAP=="CONST")):
            kStart = k - 1
        
        if((StatutNoteAV=="CONST") and (StatutNoteAP=="UP")):
            kStart = None
        
        if((StatutNoteAV=="CONST") and (StatutNoteAP=="DOWN") and (kStart != None)):
            peak_indexes.append(int(round(((kStart + k - 1) / 2.0))))
        
        StatutNoteAV = StatutNoteAP
    
    return(peak_indexes)


def histogram_gaussian_kde(data, bw_method, bins, xmin=None, xmax=None):
    if(xmin==None):
        xmin = np.min(data)
    if(xmax==None):
        xmax = np.max(data)
    x = np.linspace(xmin, xmax, bins)
    density = gaussian_kde(data, bw_method)
    histogram = density(x)    
    return([x, histogram])


class HysteresisThreshold:
    def __init__(self, ActivationThreshold, DeactivationThreshold):
        if(ActivationThreshold < DeactivationThreshold):
            raise ValueError("The activation threshold cannot be lower than the deactivation threshold" +
                             "ActivationThreshold:%.1f < DeactivationThreshold:%.1f" % (ActivationThreshold, DeactivationThreshold))
        
        self.Activation = ActivationThreshold
        self.Deactivation = DeactivationThreshold
    
    def _stateNumber(self, value):
        if(value > self.Activation):
            # Above the activation threshold
            return(2) 
        elif(value > self.Deactivation):
            # Between the both thresholds
            return(1)
        else:
            # Below the deactivation threshold
            return(0)
    
    def perform(self, signal):
        N = len(signal)
        ResultVector = np.zeros(N, dtype='bool')
        
        # Initial condition
        PastState = self._stateNumber(signal[0])
        if(PastState == 0):
            ResultVector[0] = False
        elif(PastState == 2):
            ResultVector[0] = True
        else:
            # Beginning of a between zone
            kStartZone1 = 0
        
        for k in range(1, N):
            CurrentState = self._stateNumber(signal[k])
            
            if(CurrentState == 0):
                if(PastState == 1):
                    ResultVector[kStartZone1 : k + 1] = False
                    PastState = 0
                else:
                    ResultVector[k] = False
                    PastState = 0
            elif(CurrentState == 1):
                if(PastState == 0):
                    kStartZone1 = k
                    PastState = 1
                elif(PastState == 2):
                    ResultVector[k] = True
                    PastState = 2
            elif(CurrentState == 2):
                if(PastState == 1):
                    ResultVector[kStartZone1 : k + 1] = True
                    PastState = 2
                else:
                    ResultVector[k] = True
                    PastState = 2
        
        return(ResultVector)
