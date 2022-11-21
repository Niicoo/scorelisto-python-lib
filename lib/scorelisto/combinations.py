import numpy as np
from typing import Final, Literal, List, Sequence, Tuple, Dict
import numpy.typing as npt
from dataclasses import dataclass, field
from scorelisto.musicxml import Note, NoteType, NotePitch, PitchStep

COMBINATIONS_DIVISION: Final[int] = 24

if sum(COMBINATIONS_DIVISION % np.array([2, 3, 4, 6, 8, 12], dtype=int)):
    raise ValueError("WRONG VALUE OF 'COMBINATIONS_DIVISION' SET IN THE SOURCE FILE !")

LENGTH_WHOLE: Final[int] = COMBINATIONS_DIVISION * 4
LENGTH_HALF: Final[int] = COMBINATIONS_DIVISION * 2
LENGTH_QUARTER: Final[int] = COMBINATIONS_DIVISION
LENGTH_EIGHTH: Final[int] = int(COMBINATIONS_DIVISION / 2)
LENGTH_16TH: Final[int] = int(COMBINATIONS_DIVISION / 4)
LENGTH_32ND: Final[int] = int(COMBINATIONS_DIVISION / 8)
LENGTH_T_EIGHTH: Final[int] = int(COMBINATIONS_DIVISION / 3)
LENGTH_T_16TH: Final[int] = int(COMBINATIONS_DIVISION / 6)
LENGTH_T_32ND: Final[int] = int(COMBINATIONS_DIVISION / 12)

# Type hints
FloatNDArray = npt.NDArray[np.float_]
RestOrPitch = Literal["rest", "pitch"]

@dataclass
class Combination():
    lengths: Sequence[int]
    natures: Sequence[RestOrPitch]
    nbnotes: int = field(init=False)
    nbbeats: int = field(init=False)
    triplet: bool = field(init=False)

    def __post_init__(self):
        self.lengths = np.array(self.lengths)
        self.nbnotes = len(self.lengths)
        self.nbbeats = int(np.sum(self.lengths) / LENGTH_QUARTER)
        self.triplet = (LENGTH_T_16TH in self.lengths) or (LENGTH_T_EIGHTH in self.lengths)
        if np.sum(self.lengths) % LENGTH_QUARTER:
            raise ValueError("Wrong Combination value, there isn't an integer number of beat")
        if len(self.lengths) != len(self.natures):
            raise ValueError("The length of the note length and note nature does not match")

    def getRatio(self) -> FloatNDArray:
        ratio = self.lengths / np.sum(self.lengths)
        return(ratio)

    def getError(self, note_lengths: FloatNDArray) -> float:
        ideal_lengths = self.getRatio() * np.sum(note_lengths)
        error_lengths = np.abs(ideal_lengths - note_lengths) * self.nbbeats
        error = np.sum(np.sqrt(np.power(ideal_lengths, 2.0) + np.power(error_lengths, 2.0)))
        return(error)

    def split(self, beat_start: int = 0, beat_stop: int = None) -> Tuple["Combination", int]:
        """
        Get a fraction of the combination from beat_start to beat_end.
        A new combination is generated and returned along with the number
        of notes that have been removed at the beginning of the original combination.

        :param beat_start: Beat start (from 0 to self.nbbeats) [default is 0]
        :param beat_stop: Beat stop (from 0 to self.nbbeats) [if None, it is set to self.nbbeats]
        :returns: (New splitted combination, Number of note removed)
        """
        beat_stop = self.nbbeats if beat_stop is None else beat_stop

        new_notelengths = list(self.lengths)
        new_notenatures = list(self.natures)

        ind_offset = 0
        # Remove beats to ignore at the beginning
        nb_ticks_to_delete = beat_start * COMBINATIONS_DIVISION
        for length in self.lengths:
            if(length <= nb_ticks_to_delete):
                new_notelengths.pop(0)
                new_notenatures.pop(0)
                ind_offset += 1
                nb_ticks_to_delete -= length
                if(nb_ticks_to_delete == 0): break
            else:
                new_notelengths[0] -= nb_ticks_to_delete
                break
        # Remove beats to ignore at the end
        nb_ticks_to_delete = np.sum(new_notelengths) - (beat_stop - beat_start) * COMBINATIONS_DIVISION
        for ind in range(len(new_notelengths) - 1, -1, -1):
            if(new_notelengths[ind] == nb_ticks_to_delete):
                new_notelengths.pop(ind)
                new_notenatures.pop(ind)
                break
            elif(new_notelengths[ind] < nb_ticks_to_delete):
                nb_ticks_to_delete -= new_notelengths[ind]
                new_notelengths.pop(ind)
                new_notenatures.pop(ind)
            else:
                new_notelengths[ind] -= nb_ticks_to_delete
                break
        return (Combination(new_notelengths, new_notenatures), ind_offset)

    def toNotes(self) -> List[List[Note]]:
        notes: List[List[Note]] = []
        for notelength, notenature in zip(self.lengths, self.natures):
            pitcharg = {"pitch": None if notenature == "rest" else NotePitch(PitchStep.C, -10)}
            notes_in_length: List[Note] = []
            duration_sum = 0
            while(duration_sum < notelength):
                if(notelength == (LENGTH_WHOLE + LENGTH_EIGHTH)):
                    # blanche pointée -> noire pointée :    4,5 beats
                    notes_in_length.append(Note(LENGTH_HALF + LENGTH_QUARTER, NoteType.HALF, dot=True, triplet=False, **pitcharg))
                    notes_in_length.append(Note(LENGTH_QUARTER + LENGTH_EIGHTH, NoteType.QUARTER, dot=True, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_WHOLE):
                    # ronde :                               4 beats
                    notes_in_length.append(Note(LENGTH_WHOLE, NoteType.WHOLE, dot=False, triplet=False, **pitcharg))
                elif(notelength == LENGTH_HALF + LENGTH_QUARTER + LENGTH_EIGHTH):
                    # blanche -> noire pointée :            3,5 beats
                    notes_in_length.append(Note(LENGTH_HALF, NoteType.QUARTER, dot=False, triplet=False, **pitcharg))
                    notes_in_length.append(Note(LENGTH_QUARTER + LENGTH_EIGHTH, NoteType.HALF, dot=True, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_HALF + LENGTH_QUARTER):
                    # blanche pointée :                     3 beats
                    notes_in_length.append(Note(LENGTH_HALF + LENGTH_QUARTER, NoteType.HALF, dot=True, triplet=False, **pitcharg))
                elif(notelength == LENGTH_HALF + LENGTH_EIGHTH):
                    # noire -> noire pointée :              2,5 beats
                    notes_in_length.append(Note(LENGTH_QUARTER, NoteType.QUARTER, dot=False, triplet=False, **pitcharg))
                    notes_in_length.append(Note(LENGTH_QUARTER + LENGTH_EIGHTH, NoteType.QUARTER, dot=True, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_HALF):
                    # blanche :                             2 beats
                    notes_in_length.append(Note(LENGTH_HALF, NoteType.HALF, dot=False, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_QUARTER + LENGTH_EIGHTH):
                    # noire pointée :                       1,5 beats
                    notes_in_length.append(Note(LENGTH_QUARTER + LENGTH_EIGHTH, NoteType.QUARTER, dot=True, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_QUARTER):
                    # noire :                               1 beat
                    notes_in_length.append(Note(LENGTH_QUARTER, NoteType.QUARTER, dot=False, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_EIGHTH + LENGTH_16TH):
                    # croche pointée :                      0.75 beat
                    notes_in_length.append(Note(LENGTH_EIGHTH + LENGTH_16TH, NoteType.EIGHTH, dot=True, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_EIGHTH):
                    # croche (triolet: croche triolet pointée) :              0.5 beat
                    notes_in_length.append(Note(LENGTH_EIGHTH, NoteType.EIGHTH, dot=self.triplet, triplet=self.triplet, **pitcharg))
                elif(notelength >= LENGTH_T_EIGHTH):
                    # croche triolet :                      0.33 beat
                    notes_in_length.append(Note(LENGTH_T_EIGHTH, NoteType.EIGHTH, dot=False, triplet=True, **pitcharg))
                elif(notelength >= LENGTH_16TH):
                    # double croche :                       0.25 beat
                    notes_in_length.append(Note(LENGTH_16TH, NoteType.SIXTEENTH, dot=False, triplet=False, **pitcharg))
                elif(notelength >= LENGTH_T_16TH):
                    # double croche triolet :               0.1666 beat
                    notes_in_length.append(Note(LENGTH_T_16TH, NoteType.SIXTEENTH, dot=False, triplet=True, **pitcharg))
                else:
                    raise ValueError("Invalid input argument 'Lengths', the data value does not correspond to a known note rythm")
                duration_sum = sum(note.duration for note in notes_in_length)
            if(duration_sum != notelength):
                raise RuntimeError("Error calculating the best note to write, wrong input values or not implemented rythm")
            notes.append(notes_in_length)
        return(notes)


############################################################################################
############################################################################################
############################################################################################
############################################################################################
COMBINATIONS: Dict[str, Combination] = {}
# 1 NOTE
############################################################################################
COMBINATIONS['1NOTE_1BEAT'] = Combination([LENGTH_QUARTER], ["pitch"])
COMBINATIONS['1REST_1BEAT'] = Combination([LENGTH_QUARTER], ["rest"])
COMBINATIONS['1NOTE_2BEATS'] = Combination([LENGTH_HALF], ["pitch"])
COMBINATIONS['1REST_2BEATS'] = Combination([LENGTH_HALF], ["rest"])
COMBINATIONS['1NOTE_3BEATS'] = Combination([LENGTH_HALF + LENGTH_QUARTER], ["pitch"])
COMBINATIONS['1REST_3BEATS'] = Combination([LENGTH_HALF + LENGTH_QUARTER], ["rest"])
COMBINATIONS['1NOTE_4BEATS'] = Combination([LENGTH_WHOLE], ["pitch"])
COMBINATIONS['1REST_4BEATS'] = Combination([LENGTH_WHOLE], ["rest"])
COMBINATIONS['1NOTE_5BEATS'] = Combination([LENGTH_WHOLE + LENGTH_QUARTER], ["pitch"])
COMBINATIONS['1REST_5BEATS'] = Combination([LENGTH_WHOLE + LENGTH_QUARTER], ["rest"])
COMBINATIONS['1NOTE_6BEATS'] = Combination([LENGTH_WHOLE + LENGTH_HALF], ["pitch"])
COMBINATIONS['1REST_6BEATS'] = Combination([LENGTH_WHOLE + LENGTH_HALF], ["rest"])
COMBINATIONS['1NOTE_7BEATS'] = Combination([LENGTH_WHOLE + LENGTH_HALF + LENGTH_QUARTER], ["pitch"])
COMBINATIONS['1REST_7BEATS'] = Combination([LENGTH_WHOLE + LENGTH_HALF + LENGTH_QUARTER], ["rest"])
COMBINATIONS['1NOTE_8BEATS'] = Combination([LENGTH_WHOLE + LENGTH_WHOLE], ["pitch"])
COMBINATIONS['1REST_8BEATS'] = Combination([LENGTH_WHOLE + LENGTH_WHOLE], ["rest"])

# 2 NOTES
############################################################################################
# 1 BEAT
COMBINATIONS['EN_EN'] = Combination([LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "pitch"])
COMBINATIONS['ER_EN'] = Combination([LENGTH_EIGHTH, LENGTH_EIGHTH], ["rest", "pitch"])
COMBINATIONS['EN_ER'] = Combination([LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "rest"])
COMBINATIONS['DEN_SN'] = Combination([LENGTH_EIGHTH + LENGTH_16TH, LENGTH_16TH], ["pitch", "pitch"])
COMBINATIONS['SN_DEN'] = Combination([LENGTH_16TH, LENGTH_EIGHTH + LENGTH_16TH], ["pitch", "pitch"])
# 2 BEATS
COMBINATIONS['DQN_EN'] = Combination([LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "pitch"])
COMBINATIONS['QR-ER_EN'] = Combination([LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_EIGHTH], ["rest", "pitch"])
COMBINATIONS['DQN_ER'] = Combination([LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "rest"])
COMBINATIONS['EN_EN-QN'] = Combination([LENGTH_EIGHTH, LENGTH_QUARTER + LENGTH_EIGHTH], ["pitch", "pitch"])
# 3 BEATS
COMBINATIONS['QN-DQN_EN'] = Combination([LENGTH_HALF + LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "pitch"])
COMBINATIONS['QR-QR-ER_EN'] = Combination([LENGTH_HALF + LENGTH_EIGHTH, LENGTH_EIGHTH], ["rest", "pitch"])
COMBINATIONS['QN-DQN_ER'] = Combination([LENGTH_HALF + LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "rest"])
COMBINATIONS['EN_EN-HN'] = Combination([LENGTH_EIGHTH, LENGTH_HALF + LENGTH_EIGHTH], ["pitch", "pitch"])
# 4 BEATS
COMBINATIONS['HN-DQN_EN'] = Combination([LENGTH_HALF + LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "pitch"])
COMBINATIONS['QR-QR-QR-ER_EN'] = Combination([LENGTH_HALF + LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_EIGHTH], ["rest", "pitch"])
COMBINATIONS['HN-DQN_ER'] = Combination([LENGTH_HALF + LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_EIGHTH], ["pitch", "rest"])
COMBINATIONS['EN_EN-DHN'] = Combination([LENGTH_EIGHTH, LENGTH_HALF + LENGTH_QUARTER + LENGTH_EIGHTH], ["pitch", "pitch"])

# 3 NOTES
############################################################################################
# 1 BEAT
COMBINATIONS['EN_SN_SN'] = Combination([LENGTH_EIGHTH, LENGTH_16TH, LENGTH_16TH], ["pitch", "pitch", "pitch"])
COMBINATIONS['ER_SN_SN'] = Combination([LENGTH_EIGHTH, LENGTH_16TH, LENGTH_16TH], ["rest", "pitch", "pitch"])
COMBINATIONS['SN_SN_EN'] = Combination([LENGTH_16TH, LENGTH_16TH, LENGTH_EIGHTH], ["pitch", "pitch", "pitch"])
COMBINATIONS['SN_SN_ER'] = Combination([LENGTH_16TH, LENGTH_16TH, LENGTH_EIGHTH], ["pitch", "pitch", "rest"])
COMBINATIONS['SN_EN_SN'] = Combination([LENGTH_16TH, LENGTH_EIGHTH, LENGTH_16TH], ["pitch", "pitch", "pitch"])
COMBINATIONS['T_EN_EN_EN'] = Combination([LENGTH_T_EIGHTH, LENGTH_T_EIGHTH, LENGTH_T_EIGHTH], ["pitch", "pitch", "pitch"])
COMBINATIONS['T_EN_DEN_SN'] = Combination([LENGTH_T_EIGHTH, LENGTH_EIGHTH, LENGTH_T_16TH], ["pitch", "pitch", "pitch"])
COMBINATIONS['T_EN_SN_DEN'] = Combination([LENGTH_T_EIGHTH, LENGTH_T_16TH, LENGTH_EIGHTH], ["pitch", "pitch", "pitch"])
COMBINATIONS['T_SN_EN_DEN'] = Combination([LENGTH_T_16TH, LENGTH_T_EIGHTH, LENGTH_EIGHTH], ["pitch", "pitch", "pitch"])
COMBINATIONS['T_SN_DEN_EN'] = Combination([LENGTH_T_16TH, LENGTH_EIGHTH, LENGTH_T_EIGHTH], ["pitch", "pitch", "pitch"])
COMBINATIONS['T_DEN_EN_SN'] = Combination([LENGTH_EIGHTH, LENGTH_T_EIGHTH, LENGTH_T_16TH], ["pitch", "pitch", "pitch"])
COMBINATIONS['T_DEN_SN_EN'] = Combination([LENGTH_EIGHTH, LENGTH_T_16TH, LENGTH_T_EIGHTH], ["pitch", "pitch", "pitch"])
# 2 BEATS
COMBINATIONS['EN_QN_EN'] = Combination([LENGTH_EIGHTH, LENGTH_QUARTER, LENGTH_EIGHTH], ["pitch", "pitch", "pitch"])
COMBINATIONS['DQN_SN_SN'] = Combination([LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_16TH, LENGTH_16TH], ["pitch", "pitch", "pitch"])
# 3 BEATS
COMBINATIONS['QN-DQN_SN_SN'] = Combination([LENGTH_HALF + LENGTH_EIGHTH, LENGTH_16TH, LENGTH_16TH], ["pitch", "pitch", "pitch"])
# 4 BEATS
COMBINATIONS['HN-DQN_SN_SN'] = Combination([LENGTH_HALF + LENGTH_QUARTER + LENGTH_EIGHTH, LENGTH_16TH, LENGTH_16TH], ["pitch", "pitch", "pitch"])

# 4 NOTES
############################################################################################
# 1 BEAT
COMBINATIONS['SN_SN_SN_SN'] = Combination([LENGTH_16TH, LENGTH_16TH, LENGTH_16TH, LENGTH_16TH], ["pitch", "pitch", "pitch", "pitch"])
############################################################################################
############################################################################################
############################################################################################
############################################################################################


class MeasureCombinationsSplitter:

    def _findMinimumCutLength(self, length_cut: int) -> int:
        """
        When a note is splitted in 2, we want to find the smallest note that is going
        to be generating by this split.
        """
        if length_cut in [
                LENGTH_WHOLE, LENGTH_HALF, LENGTH_QUARTER, LENGTH_EIGHTH, LENGTH_T_EIGHTH, 
                LENGTH_16TH, LENGTH_T_16TH, LENGTH_32ND, LENGTH_T_32ND]:
            return length_cut

        options = []
        for term_1 in [LENGTH_WHOLE, LENGTH_HALF, LENGTH_QUARTER, LENGTH_EIGHTH, LENGTH_T_EIGHTH, LENGTH_16TH, LENGTH_T_16TH, LENGTH_32ND, LENGTH_T_32ND]:
            for term_2 in [LENGTH_WHOLE, LENGTH_WHOLE, LENGTH_HALF, LENGTH_QUARTER, LENGTH_EIGHTH, LENGTH_T_EIGHTH, LENGTH_16TH, LENGTH_T_16TH, LENGTH_32ND, LENGTH_T_32ND]:
                if (term_1 + term_2) == length_cut:
                    options.append(min(term_1, term_2))
        if not options:
            raise NotImplementedError
        return max(options)

    def _findBestBeatsPerMeasure(self, combinations: List[Combination]) -> Tuple[int, int]:
        """
        Find the best measure configuration (3:4, 4:4, 5:4)
        "best" = what makes the score the most beautiful (limiting splitting the notes)

        :param combinations: List of combinations to split
        :returns: Number of beat per measure, Number of rests to add at the beginning
        """

        list_lengths = [length for comb in combinations for length in comb.lengths]
        best_score = np.inf
        best_config = (-1, -1)
        for nbbeats_per_measure in [3, 4, 5]:
            for nbrests_added in range(0, nbbeats_per_measure - combinations[0].nbbeats + 1):
                list_lengths_temp = [COMBINATIONS_DIVISION] * nbrests_added + list_lengths
                cumsum_lengths = np.cumsum(np.array(list_lengths_temp, dtype=int))
                # Get the splits that would be created in this configuration
                splits = []
                nbdiv_search = nbbeats_per_measure * COMBINATIONS_DIVISION
                for nbdivisions in cumsum_lengths:
                    while nbdivisions >= nbdiv_search:
                        if nbdivisions == nbdiv_search:
                            # No Split
                            nbdiv_search += nbbeats_per_measure * COMBINATIONS_DIVISION
                        elif nbdivisions > nbdiv_search:
                            # Split
                            # notelength = list_lengths[ind_div - nbrests_added]
                            length_cut = (nbdivisions - nbdiv_search)
                            length_cut = self._findMinimumCutLength(length_cut)
                            splits.append(length_cut)
                            nbdiv_search += nbbeats_per_measure * COMBINATIONS_DIVISION
                # Calculate the score of this configuration
                score = np.sum(1 / (np.array(splits) / COMBINATIONS_DIVISION))
                if score < best_score:
                    best_score = score
                    best_config = (nbbeats_per_measure, nbrests_added)
        return best_config

    def _getNoteIndexes(self, combinations: List[Combination]) -> List[int]:
        """
        Get the indexes of the first notes of each combination

        :param combinations: List of combinations
        :returns: List of indexes
        """
        note_indexes = [0]
        for ind_comb, comb in enumerate(combinations):
            past_ind = note_indexes[-1]
            if(ind_comb != len(combinations) - 1):
                note_indexes.append(past_ind + comb.nbnotes)
        return(note_indexes)
        
    def perform(self, combinations: List[Combination]) -> Tuple[List[List[Tuple[Note, int]]], int]:
        """
        Find the best measure configuration (3:4, 4:4, 5:4)
        and split the combinations accordingly.
        Because it's important to save the original indexes of the notes before being split
        the indexes of the notes is also returned

        :param combinations: List of combinations to split
        :returns: List(measures) of List of notes and its index and the number of beats per measure
        """
        nb_beats_per_measure, nb_rests_to_add = self._findBestBeatsPerMeasure(combinations)
        notes_indexes = self._getNoteIndexes(combinations)
        # Initialization
        list_measures = []
        measure_tmp = []
        nb_beats_total = sum([comb.nbbeats for comb in combinations]) + nb_rests_to_add
        nb_beats_total_done = 0
        nb_beats_measure_remaining = nb_beats_per_measure
        ind_comb = 0
        ind_beat_start = 0
        # Adding the required rests at the beginning
        for _ in range(nb_rests_to_add):
            measure_tmp.append((Note(LENGTH_QUARTER, NoteType.QUARTER), -1))
            nb_beats_total_done += 1
            nb_beats_measure_remaining -= 1
        # Add (and split if necessary) the combinations
        while(nb_beats_total_done < nb_beats_total):
            # Number of beat remaining in the current combination
            nb_beats_comb_remaining = combinations[ind_comb].nbbeats - ind_beat_start
            # End Indice of the combination to consider
            if(nb_beats_comb_remaining < nb_beats_measure_remaining):
                ind_beat_end = combinations[ind_comb].nbbeats
            else:
                ind_beat_end = ind_beat_start + nb_beats_measure_remaining
            # Get the correspondant notes
            comb, ind_offset = combinations[ind_comb].split(ind_beat_start, ind_beat_end)
            comb_notes = comb.toNotes()
            ind_note = ind_offset + notes_indexes[ind_comb]
            for notes in comb_notes:
                for notetemp in notes:
                    measure_tmp.append((notetemp, ind_note))
                ind_note += 1
            # Adding the number ot beat done
            nb_beats_total_done += (ind_beat_end - ind_beat_start)
            # Increment on the combination if all the beat of the combination are processed
            if(nb_beats_comb_remaining <= nb_beats_measure_remaining):
                ind_comb += 1
                ind_beat_start = 0
            else:
                ind_beat_start += nb_beats_measure_remaining
            # Adding a new measure if the measure is finished
            if(nb_beats_comb_remaining >= nb_beats_measure_remaining):
                list_measures.append(measure_tmp)
                measure_tmp = []
                nb_beats_measure_remaining = nb_beats_per_measure
            else:
                nb_beats_measure_remaining -= nb_beats_comb_remaining
        # Complete the last measure of silence
        if(measure_tmp != []):
            for _ in range(0, nb_beats_measure_remaining):
                measure_tmp.append((Note(LENGTH_QUARTER, NoteType.QUARTER), -1))
            list_measures.append(measure_tmp)
        return(list_measures, nb_beats_per_measure)
