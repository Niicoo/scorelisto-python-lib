import numpy as np
import logging
import copy
import random
from collections.abc import Generator
from dataclasses import dataclass
from .combinations import COMBINATIONS, COMBINATIONS_DIVISION, RestOrPitch, MeasureCombinationsSplitter
from .tools import tone2note, note2tone
from .dijkstra import Graph
from .note_detectors import AnalogNote
from .musicxml import MusicXmlScore, PitchStep, NotePitch, ClefSign, KeyFifths, KeyMode, CreateAttributesElement
from .midi import MidiScore, MetaEvent, MidiEvent
from typing import List, Dict, Tuple, Union

# Get Logger
logger = logging.getLogger(__name__)

# Typing
NoteId = str
NoteIndex = Tuple[int, int, int]


@dataclass
class ConfCombination():
    key: str
    error: float
    masked: bool


class Configuration():
    def __init__(self, nbnotes: int, nbbeats: int, delay: float, id_notes: List[NoteId], ref: str = ''):
        self.delay: float = delay
        self.nbbeats: int = nbbeats
        self.nbnotes: int = nbnotes
        self.notes_ids: List[NoteId] = id_notes.copy()
        self.ref: str = ref
        self.combinations: List[ConfCombination] = []

    def getDelay(self) -> float:
        return(self.delay)

    def getNbbeats(self) -> int:
        return(self.nbbeats)

    def getNbnotes(self) -> int:
        return(self.nbnotes)

    def getRef(self) -> str:
        return(self.ref)

    def getCombinations(self) -> List[ConfCombination]:
        return self.combinations.copy()

    def setRef(self, ref: str) -> None:
        self.ref = ref

    def getNotesIds(self) -> List[NoteId]:
        return(self.notes_ids.copy())

    def getBestCombination(self) -> ConfCombination:
        for comb in self.combinations:
            if not comb.masked:
                return comb
        raise RuntimeError("No combination available")

    def hasAtLeastOneUnmaskComb(self) -> bool:
        for comb in self.combinations:
            if not comb.masked:
                return True
        return False

    def combinationAlreadyExists(self, key: str) -> bool:
        for comb in self.combinations:
            if(comb.key == key):
                return True
        return False

    def isValid(self, error_max: float, delay_min: float, delay_max: float) -> bool:
        """
            Check if at least one combination is valid according to the limits set by the input arguments.
        """
        if((self.delay > delay_max) or (self.delay < delay_min)):
            return False
        for comb in self.combinations:
            if((not comb.masked) and (comb.error < error_max)):
                return True
        return False

    def addCombination(self, key: str, error: float, mask: bool = False) -> None:
        if not (key in COMBINATIONS):
            raise ValueError("the combination you're trying to add does not exists: %s" % key)

        if(self.combinationAlreadyExists(key)):
            raise ValueError("Trying to add a combination that is already in there")

        self.combinations.append(ConfCombination(key, error, mask))
        self.combinations.sort(key=lambda param: param.error)

    def maskCombinations(self, keys: List[str]) -> None:
        for comb in self.combinations:
            if comb.key in keys:
                comb.masked = True


class DigitalNote:
    def __init__(self, notetype: RestOrPitch, analoglength: float, analogheight: float = None, octave: int = None):
        if((notetype != "rest") and (notetype != "pitch")):
            raise ValueError("Invalid input argument for 'notetype', can be 'pitch' or 'rest' , actual value: '%s'" % notetype)
        if((notetype == "rest") and ((analogheight is not None) or (octave is not None))):
            raise ValueError("Too many arguments for DigitalNote of type 'rest'")
        if((notetype == "note") and ((analogheight is None) or (octave is None))):
            raise ValueError("Not enought arguments for DigitalNote of type 'note', the height and the octave have to be specified")
        self.type: RestOrPitch = notetype
        self.analoglength: float = analoglength
        self.analogheight: float = analogheight
        self.octave: int = octave
        self.configurations: List[Configuration] = []

    def isARest(self) -> bool:
        return(self.type == "rest")

    def isANote(self) -> bool:
        return(self.type == "pitch")

    def getType(self) -> RestOrPitch:
        return(self.type)

    def getAnalogLength(self) -> float:
        return(self.analoglength)

    def getAnalogHeight(self) -> float:
        return(self.analogheight)

    def getConfigurations(self) -> List[Configuration]:
        return self.configurations

    def resetConfigurations(self) -> None:
        self.configurations = []

    def getRoundedHeightAndOctave(self) -> Tuple[int, int]:
        if(self.analogheight >= 11.5):
            return((0, self.octave + 1))
        else:
            int_height = int(np.round(self.analogheight))
            return((int_height, self.octave))

    def configurationAlreadyExists(self, new_conf: Configuration) -> bool:
        existing_refs = [conf.getRef() for conf in self.configurations]
        already_exists = new_conf.getRef() in existing_refs
        return(already_exists)

    def addConfiguration(self, conf: Configuration) -> None:
        if(self.configurationAlreadyExists(conf)):
            raise ValueError("Trying to add the configuration with a ref already existing: %s" % conf.getRef())
        self.configurations.append(conf)

    def applyOffset(self, offset: float) -> None:
        temp_tone = note2tone(self.analogheight, self.octave)
        temp_tone += offset
        self.analogheight, self.octave = tone2note(temp_tone)

    def increaseByOneOctave(self) -> None:
        if(self.isARest()):
            logger.warning("Changing the octave of a rest has no effect.")
        else:
            self.octave += 1

    def lowerByOneOctave(self) -> None:
        if(self.isARest()):
            logger.warning("Changing the octave of a rest has no effect.")
        else:
            self.octave -= 1

    def lowerByOneHalftone(self) -> None:
        if(self.isARest()):
            logger.warning("Changing the tone of a rest has no effect.")
        else:
            if((self.analogheight - 1.0) < 0.0):
                self.lowerByOneOctave()
            self.analogheight = (self.analogheight - 1.0) % 12

    def increaseByOneHalftone(self) -> None:
        if(self.isARest()):
            logger.warning("Changing the tone of a rest has no effect.")
        else:
            if((self.analogheight + 1.0) >= 12.0):
                self.increaseByOneOctave()
            self.analogheight = (self.analogheight + 1.0) % 12

    def maskCombinations(self, keys: List[str]) -> None:
        for conf in self.configurations:
            conf.maskCombinations(keys)

    def isHeightInFifths(self, fifths: KeyFifths) -> bool:
        if(self.isARest()):
            raise RuntimeError("Trying to get the parameter of the note height whereas it's actually just a 'rest'")
        int_height, _ = self.getRoundedHeightAndOctave()
        # DO / SI DIESE
        if((int_height == 0) and (((fifths.value >= -5) and (fifths.value <= 1)) or (fifths.value == 7))):
            return(True)
        # DO DIESE / RE BEMOL
        elif((int_height == 1) and ((fifths.value >= 2) or (fifths.value <= -4))):
            return(True)
        # RE
        elif((int_height == 2) and ((fifths.value >= -3) and (fifths.value <= 3))):
            return(True)
        # RE DIESE / MI BEMOL
        elif((int_height == 3) and ((fifths.value >= 4) or (fifths.value <= -2))):
            return(True)
        # MI / FA BEMOL
        elif((int_height == 4) and (((fifths.value >= -1) and (fifths.value <= 5)) or (fifths.value == -7))):
            return(True)
        # FA / MI DIESE
        elif((int_height == 5) and (((fifths.value >= -6) and (fifths.value <= 0)) or (fifths.value >= 6))):
            return(True)
        # FA DIESE / SOL BEMOL
        elif((int_height == 6) and ((fifths.value >= 1) or (fifths.value <= -5))):
            return(True)
        # SOL
        elif((int_height == 7) and ((fifths.value >= -4) and (fifths.value <= 2))):
            return(True)
        # SOL DIESE / LA BEMOL
        elif((int_height == 8) and ((fifths.value >= 3) or (fifths.value <= -3))):
            return(True)
        # LA
        elif((int_height == 9) and ((fifths.value >= -2) and (fifths.value <= 4))):
            return(True)
        # LA DIESE / SI BEMOL
        elif((int_height == 10) and ((fifths.value >= 5) or (fifths.value <= -1))):
            return(True)
        # SI / DO BEMOL
        elif((int_height == 11) and (((fifths.value >= 0) and (fifths.value <= 6)) or (fifths.value <= -6))):
            return(True)
        return(False)

    def getStepAlterOctave(self, fifths: KeyFifths) -> NotePitch:
        if(self.isARest()):
            raise RuntimeError("Trying to get the parameter of the note height whereas it's actually just a 'rest'")
        height_int, octave = self.getRoundedHeightAndOctave()
        if(height_int == 0):
            if(fifths.value >= 7):
                # SI DIESE
                return(NotePitch(PitchStep.B, octave - 1, 1))
            else:
                # DO
                return(NotePitch(PitchStep.C, octave))
        elif(height_int == 1):
            if(fifths.value >= -3):
                # DO DIESE
                return(NotePitch(PitchStep.C, octave, 1))
            else:
                # RE BEMOL
                return(NotePitch(PitchStep.D, octave, -1))
        elif(height_int == 2):
            # RE
            return(NotePitch(PitchStep.D, octave))
        elif(height_int == 3):
            if(fifths.value >= 4):
                # RE DIESE
                return(NotePitch(PitchStep.D, octave, 1))
            else:
                # MI BEMOL
                return(NotePitch(PitchStep.E, octave, -1))
        elif(height_int == 4):
            if(fifths.value <= -7):
                # FA BEMOL
                return(NotePitch(PitchStep.F, octave, 1))
            else:
                # MI
                return(NotePitch(PitchStep.E, octave))
        elif(height_int == 5):
            if(fifths.value >= 6):
                # MI DIESE
                return(NotePitch(PitchStep.E, octave, 1))
            else:
                # FA
                return(NotePitch(PitchStep.F, octave))
        elif(height_int == 6):
            if(fifths.value >= -4):
                # FA DIESE
                return(NotePitch(PitchStep.F, octave, 1))
            else:
                # SOL BEMOL
                return(NotePitch(PitchStep.G, octave, -1))
        elif(height_int == 7):
            # SOL
            return(NotePitch(PitchStep.G, octave))
        elif(height_int == 8):
            if(fifths.value > 0):
                # SOL DIESE
                return(NotePitch(PitchStep.G, octave, 1))
            else:
                # LA BEMOL
                return(NotePitch(PitchStep.A, octave, -1))
        elif(height_int == 9):
            # LA
            return(NotePitch(PitchStep.A, octave))
        elif(height_int == 10):
            if(fifths.value >= 5):
                # LA DIESE
                return(NotePitch(PitchStep.A, octave, 1))
            else:
                # SI BEMOL
                return(NotePitch(PitchStep.B, octave, -1))
        elif(height_int == 11):
            if(fifths.value >= -5):
                # SI
                return(NotePitch(PitchStep.B, octave))
            else:
                # DO BEMOL
                return(NotePitch(PitchStep.C, octave + 1, -1))
        else:
            raise ValueError('Invalid note height: %d  (must be an integer between 0 and 11)' % height_int)


class DigitalPartition:
    def __init__(self,
            delaymin: float = 0.3,
            delaymax: float = 1.5,
            maxdelayvar: float = 0.5,
            errormax: float = 10.0,
            combs_to_mask: List[str] = None):
        """
        self.chrono_notes = [
            [[ id_note1 ]],
            [[ id_note2 ]],
            [[ id_note3, id_rest1], [id_note_simu1]],
            [[ id_note4 ]],
            [[ id_note5, id_rest2], [id_note_simu2]],
            [[ id_note6, id_rest3], [id_note_simu3]],
            ...
        ]
        """
        # Attributes
        self.delaymin: float = delaymin
        self.delaymax: float = delaymax
        self.maxdelayvar: float = maxdelayvar
        self.errormax: float = errormax
        self.combs_to_mask: float = [] if combs_to_mask is None else combs_to_mask.copy()

        # Temporary parameters
        self.dico_notes: Dict[NoteId, DigitalNote] = {}
        self.chrono_notes: List[List[List[NoteId]]] = []
        self.clef: ClefSign = ClefSign.G
        self.fifths: KeyFifths = KeyFifths.ZERO
        self.graph: Graph = None
        self.best_path_found: List[str] = []
        self.best_path_infos: List[Dict[str, Union[str, List[NoteId]]]] = []

    def _getUniqueId(self, prefix: str) -> NoteId:
        id = prefix + "_%06d" % random.sample(range(0, 1000000), 1)[0]
        while id in self.dico_notes:
            id = self._getUniqueId(prefix)
        return id

    def addNote(self, is_a_note: bool, note_length: float, note_height: float) -> None:
        if(is_a_note):
            height, octave = tone2note(note_height)
            dignote = DigitalNote("pitch", note_length, height, octave)
            # Adding a new note
            id_note = self._getUniqueId("NOTE")
            self.dico_notes[id_note] = dignote
            self.chrono_notes.append([[id_note]])
            logger.debug(f"Note added (ID: {id_note}) [length={dignote.analoglength}, height={dignote.analogheight}]")
        else:
            if(len(self.chrono_notes) == 0):
                raise ValueError("Cannot Add a rest at the beginning of a partition")
            elif(len(self.chrono_notes[-1]) > 2):
                raise ValueError("Problem during building digital partition")
            elif(len(self.chrono_notes[-1]) == 2):
                logger.warning("Adding a rest after a rest... something's weird")
                # Adding length to the past rest
                RefPastRest = self.chrono_notes[-1][0][1]
                PastLengthRest = self.dico_notes[RefPastRest].analoglength
                self.dico_notes[RefPastRest].analoglength = PastLengthRest + note_length
                # Adding length to the past simulated note
                RefPastNote = self.chrono_notes[-1][1][0]
                PastLengthNote = self.dico_notes[RefPastNote].analoglength
                self.dico_notes[RefPastNote].analoglength = PastLengthNote + note_length
            else:
                # Adding a new rest
                dignote = DigitalNote("rest", note_length)
                id_note = self._getUniqueId("REST")
                self.dico_notes[id_note] = dignote
                self.chrono_notes[-1][0].append(id_note)
                logger.debug(f"Rest added (ID: {id_note}) [length={dignote.analoglength}]")
                # Simulate a note having length of the past note + this rest
                RefPastNote = self.chrono_notes[-1][0][0]
                dignote = copy.deepcopy(self.dico_notes[RefPastNote])
                dignote.analoglength = dignote.analoglength + note_length
                id_note = self._getUniqueId("SNOTE")
                self.dico_notes[id_note] = dignote
                self.chrono_notes[-1].append([id_note])
                logger.debug(f"Simulated Note added (ID: {id_note}) [length={dignote.analoglength}, height={dignote.analogheight}]")

    def _iterateOverAllNotes(self) -> Generator[NoteIndex]:
        for a in range(len(self.chrono_notes)):
            for b in range(len(self.chrono_notes[a])):
                for c in range(len(self.chrono_notes[a][b])):
                    yield((a, b, c))

    def _getIndexesFromIdNote(self, id_note: NoteId) -> NoteIndex:
        for inds in self._iterateOverAllNotes():
            if(self.chrono_notes[inds[0]][inds[1]][inds[2]] == id_note):
                return(inds)

    def _getIdNoteFromIndexes(self, inds: NoteIndex) -> NoteId:
        return(self.chrono_notes[inds[0]][inds[1]][inds[2]])

    def _isTheLastNote(self, id_note: NoteId) -> bool:
        return(self.chrono_notes[-1][0][0] == id_note)

    def _getIndsNextNotes(self, inds: NoteIndex) -> List[NoteIndex]:
        inds_next = []
        if(len(self.chrono_notes[inds[0]][inds[1]]) > (inds[2] + 1)):
            inds_next.append((inds[0], inds[1], inds[2] + 1))
        elif(len(self.chrono_notes) > (inds[0] + 1)):
            if(len(self.chrono_notes[inds[0] + 1]) == 2):
                inds_next.append((inds[0] + 1, 0, 0))
                inds_next.append((inds[0] + 1, 1, 0))
            else:
                inds_next.append((inds[0] + 1, 0, 0))
        return(inds_next)

    def _getOptimumOffset(self, precision: float = 0.005) -> float:
        error_min = np.inf
        best_offset = np.nan
        for offset in np.arange(-0.5, 0.5, precision):
            error = 0
            for ind_note in range(len(self.chrono_notes)):
                id_note = self.chrono_notes[ind_note][0][0]
                if(self.dico_notes[id_note].isANote()):
                    tone = self.dico_notes[id_note].getAnalogHeight() + offset
                    error += np.abs(tone - np.round(tone))
            if error < error_min:
                best_offset = offset
        return(best_offset)

    def _applyOffsetToAllNotes(self, offset: float):
        for id_note in self.dico_notes:
            if(self.dico_notes[id_note].isANote()):
                self.dico_notes[id_note].applyOffset(offset)

    def _setSimulatedRestsBeginning(self) -> None:
        """
            The beat may not start on the first note, it can be off beat. Because the method
            used here to recover is based on recovering the position of the beat, we add rests
            at the beginning to consider the possibility of this off beat case.
        """
        # Deleting older simulated rests
        if(self.chrono_notes[0][0][0][0:4] == "SRES"):
            self.dico_notes.pop(self.chrono_notes[0][0][0])
            if(len(self.chrono_notes[0]) == 2):
                self.dico_notes.pop(self.chrono_notes[0][1][0])
            self.chrono_notes.pop(0)
        # More than 1 note at the beginning ?
        nb_notes_beginning = len(self.chrono_notes[0])
        # Length of the first note
        id_first_note = self.chrono_notes[0][0][0]
        rest_length = self.dico_notes[id_first_note].getAnalogLength()
        # Adding a rest of the same length
        dignote1 = DigitalNote("rest", rest_length)
        id_new_note: NoteId = self._getUniqueId("SREST")
        self.dico_notes[id_new_note] = dignote1
        self.chrono_notes.insert(0, [[id_new_note]])
        # Adding a second simulated rest if there is a simulated note
        if(nb_notes_beginning == 2):
            id_first_note = self.chrono_notes[1][1][0]
            rest_length = self.dico_notes[id_first_note].getAnalogLength()
            dignote2 = DigitalNote("rest", rest_length)
            id_new_note: NoteId = self._getUniqueId("SREST")
            self.dico_notes[id_new_note] = dignote2
            self.chrono_notes[0].append([id_new_note])

    def increaseOneOctaveEveryNotes(self) -> None:
        for id_note in self.dico_notes:
            if self.dico_notes[id_note].isANote():
                self.dico_notes[id_note].increaseByOneOctave()

    def lowerOneOctaveEveryNotes(self) -> None:
        for id_note in self.dico_notes:
            if self.dico_notes[id_note].isANote():
                self.dico_notes[id_note].lowerByOneOctave()

    def increaseOneHalftoneEveryNotes(self) -> None:
        for id_note in self.dico_notes:
            if self.dico_notes[id_note].isANote():
                self.dico_notes[id_note].increaseByOneHalftone()

    def lowerOneHalftoneEveryNotes(self) -> None:
        for id_note in self.dico_notes:
            if self.dico_notes[id_note].isANote():
                self.dico_notes[id_note].lowerByOneHalftone()

    def _getMeanHeight(self) -> float:
        mean_height = 0
        nb_notes = 0
        for id_note in self.dico_notes:
            if(id_note[0:4] == "NOTE"):
                Height = self.dico_notes[id_note].analogheight
                Octave = self.dico_notes[id_note].octave
                mean_height += note2tone(Height, Octave)
                nb_notes += 1
        mean_height = mean_height / nb_notes
        return(mean_height)

    def _listNotesNext(self, ind_start: NoteIndex, nb_notes: int) -> List[List[NoteId]]:
        next_notes = []
        next_notes.append([ind_start])
        for _ in range(nb_notes - 1):
            list_notes = copy.deepcopy(next_notes)
            next_notes = []
            for num in range(len(list_notes)):
                # Last note of the current list
                ind_last = list_notes[num][-1]
                inds_next = self._getIndsNextNotes(ind_last)
                for ind_note in inds_next:
                    TEMP = copy.deepcopy(list_notes[num])
                    TEMP.append(ind_note)
                    next_notes.append(TEMP)
        for k in range(0, len(next_notes)):
            for h in range(0, len(next_notes[k])):
                next_notes[k][h] = self._getIdNoteFromIndexes(next_notes[k][h])
        return(next_notes)

    def _listBestCombinationsForOneConf(self, path: List[NoteId], nb_beats: int) -> Configuration:
        nb_notes = len(path)
        natures: List[RestOrPitch] = []
        note_lengths: List[float] = []
        for id_note in path:
            note_lengths.append(self.dico_notes[id_note].getAnalogLength())
            natures.append(self.dico_notes[id_note].getType())
        delay = np.sum(note_lengths) / nb_beats
        conf = Configuration(nb_notes, nb_beats, delay, path)
        for key, comb in COMBINATIONS.items():
            if(comb.nbnotes != nb_notes):
                continue
            if(comb.nbbeats != nb_beats):
                continue
            if(comb.natures != natures):
                continue
            error = comb.getError(note_lengths)
            conf.addCombination(key, error)
        return conf

    def _getConfFromConfRef(self, ref: str) -> Configuration:
        for id_note in self.dico_notes:
            for conf in self.dico_notes[id_note].getConfigurations():
                if(conf.getRef() == ref):
                    return(conf)
        raise ValueError(f"The configuration requested (ref='{ref}') does not exits.")

    def _A_MinimizeHeightError(self) -> None:
        offset = self._getOptimumOffset()
        self._applyOffsetToAllNotes(offset)

    def _B_AutoSetFifths(self) -> None:
        best_fifths = KeyFifths.ZERO
        nb_signs_best = np.inf
        for fifths in KeyFifths:
            nb_signs_tmp = 0
            for id_note in self.dico_notes:
                # not selecting simulated notes !!
                if(id_note[0:4] == 'NOTE'):
                    if not self.dico_notes[id_note].isHeightInFifths(fifths):
                        nb_signs_tmp += 1
            if(nb_signs_tmp < nb_signs_best):
                best_fifths = fifths
                nb_signs_best = nb_signs_tmp
        self.fifths = best_fifths

    def _C_AutoSetClef(self) -> None:
        mean_height = self._getMeanHeight()
        if(mean_height / 12.0 >= 4):
            self.clef = ClefSign.G
        else:
            self.clef = ClefSign.F

    def _D_AutoTranslateOctave(self) -> None:
        mean_height = self._getMeanHeight()
        if(self.clef == ClefSign.G):
            nb_octaves = int(np.round(((5 * 12 - 1) - mean_height) / 12.0))
        elif(self.clef == ClefSign.F):
            nb_octaves = int(np.round(((3 * 12 + 2) - mean_height) / 12.0))
        else:
            raise ValueError("Invalid attributes 'clef', can be 'G' or 'F', currently it's: %s" % self.clef.value)
        if(nb_octaves > 0):
            for _ in range(0, abs(nb_octaves)):
                self.increaseOneOctaveEveryNotes()
        elif(nb_octaves < 0):
            for _ in range(0, abs(nb_octaves)):
                self.lowerOneOctaveEveryNotes()

    def _E_FindConfigurationsForAllNotes(self) -> None:
        self._setSimulatedRestsBeginning()
        no_conf = 0
        for index_note in self._iterateOverAllNotes():
            id_note = self._getIdNoteFromIndexes(index_note)
            self.dico_notes[id_note].resetConfigurations()
            for nb_notes in range(1, 5):
                paths = self._listNotesNext(index_note, nb_notes)
                for nb_beats in range(1, 9):
                    for path in paths:
                        conf = self._listBestCombinationsForOneConf(path, nb_beats)
                        if(len(conf.getCombinations()) > 0):
                            conf.setRef(f"CONF_{no_conf:06d}_{id_note}")
                            self.dico_notes[id_note].addConfiguration(conf)
                            no_conf += 1
        logger.debug(f"{no_conf} configurations added")

    def _F_MaskCombinations(self) -> None:
        for id_note in self.dico_notes:
            self.dico_notes[id_note].maskCombinations(self.combs_to_mask)

    def _G_BuildGraph(self) -> None:
        weight_adjust = 0.5
        self.graph = Graph()
        self.graph.addVertex("StartPoint")
        self.graph.addVertex("EndPoint")
        # Creating vertex
        for id_note in self.dico_notes:
            for conf in self.dico_notes[id_note].getConfigurations():
                if(conf.hasAtLeastOneUnmaskComb()):
                    self.graph.addVertex(conf.getRef())
        # Creating edges
        for k in range(0, len(self.chrono_notes[0])):
            id_note = self.chrono_notes[0][k][0]
            length_simulated_rest = self.dico_notes[id_note].getAnalogLength()
            for conf in self.dico_notes[id_note].getConfigurations():
                if(conf.hasAtLeastOneUnmaskComb()):
                    self.graph.addEdge("StartPoint", conf.getRef(), -length_simulated_rest)

        for k in range(0, len(self.chrono_notes[1])):
            id_note = self.chrono_notes[1][k][0]
            for conf in self.dico_notes[id_note].getConfigurations():
                if(conf.hasAtLeastOneUnmaskComb()):
                    self.graph.addEdge("StartPoint", conf.getRef(), 0)

        for index_note in self._iterateOverAllNotes():
            id_note = self._getIdNoteFromIndexes(index_note)
            for conf in self.dico_notes[id_note].getConfigurations():
                if(not conf.isValid(self.errormax, self.delaymin, self.delaymax)):
                    logger.debug(f"Configuration {conf.getRef()} not valid ")
                    continue
                current_ref = conf.getRef()
                current_delay = conf.getDelay()
                comb_error = conf.getBestCombination().error
                comb_length = conf.getDelay() * conf.getNbbeats()
                id_last_note = conf.getNotesIds()[-1]
                if(self._isTheLastNote(id_last_note)):
                    self.graph.addEdge(current_ref, "EndPoint", comb_length)
                else:
                    index_last_note = self._getIndexesFromIdNote(id_last_note)
                    indexes_next_notes = self._getIndsNextNotes(index_last_note)
                    for index_note_next in indexes_next_notes:
                        id_note_next = self._getIdNoteFromIndexes(index_note_next)
                        for conf_next in self.dico_notes[id_note_next].getConfigurations():
                            next_ref = conf_next.getRef()
                            next_delay = conf_next.getDelay()
                            delay_var_valid = np.abs(current_delay - next_delay) / current_delay < self.maxdelayvar
                            if(conf_next.isValid(self.errormax, self.delaymin, self.delaymax) and delay_var_valid):
                                delay_error = np.sqrt(np.power(10.0 * np.log10(next_delay) - 10.0 * np.log10(current_delay), 2.0) + np.power(comb_length, 2.0))
                                weight = (1.0 - weight_adjust) * delay_error + weight_adjust * comb_error
                                self.graph.addEdge(current_ref, next_ref, weight)

    def _H_GetOptimalPath(self) -> None:
        self.best_path_found = self.graph.performDijkstraShortestPath("StartPoint", "EndPoint")[1:-1]
        logger.debug(f"Best path found: {self.best_path_found}")
        self.best_path_infos = []
        for conf_ref in self.best_path_found:
            conf = self._getConfFromConfRef(conf_ref)
            best_comb = conf.getBestCombination().key
            id_notes = conf.getNotesIds()
            logger.debug(f"conf {conf_ref}: combination {best_comb}, notes: {id_notes}")
            self.best_path_infos.append({"combination": best_comb, "id_notes": id_notes})

    def _I_generateMusicXMLScore(self, author: str, title: str) -> bytes:
        if(self.best_path_infos == []):
            return(None)
        combinations = [COMBINATIONS[info["combination"]] for info in self.best_path_infos]
        notes_ids = []
        for info in self.best_path_infos:
            notes_ids = notes_ids + info["id_notes"]
        measure_splitter = MeasureCombinationsSplitter()
        list_measures, nb_beats_per_measure = measure_splitter.perform(combinations)
        # Create a partition
        scoreXML = MusicXmlScore(Title=title, Composer=author)
        # Create the parts
        id_part = "P1"
        scoreXML.addPart(id_part, "Solo")
        # Attributes
        clef_line = 2 if (self.clef == ClefSign.G) else 4
        e_attributes = CreateAttributesElement(
            divisions=COMBINATIONS_DIVISION,
            key_fifths=self.fifths,
            key_mode=KeyMode.MAJOR,
            time_beats=nb_beats_per_measure,
            time_beattype=4,
            clef_sign=self.clef,
            clef_line=clef_line)
        # Part 1
        for ind_measure, measure in enumerate(list_measures):
            scoreXML.addMeasure(id_part)
            if(ind_measure == 0):
                scoreXML.addAttributes(id_part, e_attributes)
            for note, index in measure:
                if(note.pitch is not None):
                    id_note = notes_ids[index]
                    pitch = self.dico_notes[id_note].getStepAlterOctave(self.fifths)
                    note.pitch = pitch
                scoreXML.addNote(id_part, note)
        # Writting the file
        data_string = scoreXML.toString()
        return(data_string)

    def _getMeanDelayPath(self) -> float:
        mean_delay = 0.0
        nbtotbeats = 0
        for conf_ref in self.best_path_found:
            conf = self._getConfFromConfRef(conf_ref)
            delay = conf.getDelay()
            nbbeats = conf.getNbbeats()
            mean_delay += delay * nbbeats
            nbtotbeats += nbbeats
        mean_delay = mean_delay / nbtotbeats
        return(mean_delay)

    def _I_generateMidiScore(self) -> bytes:
        if(self.best_path_infos == []):
            return(None)
        combinations = [COMBINATIONS[info["combination"]] for info in self.best_path_infos]
        notes_ids = []
        for info in self.best_path_infos:
            notes_ids = notes_ids + info["id_notes"]
        measure_splitter = MeasureCombinationsSplitter()
        list_measures, nb_beats_per_measure = measure_splitter.perform(combinations)
        mean_delay = self._getMeanDelayPath()
        # microseconds per quarter note
        tempo = int(round(mean_delay * 1e6))
        midi_score = MidiScore(PPQ = 480)
        track0 = midi_score.addTrack()
        track0.addMetaEvent(MetaEvent.SET_TEMPO, data=tempo)
        track0.addMetaEvent(MetaEvent.TIME_SIGNATURE, data=(nb_beats_per_measure, 4, COMBINATIONS_DIVISION, 8))
        track0.addMetaEvent(MetaEvent.END_OF_TRACK)
        track1 = midi_score.addTrack()
        track1.addMetaEvent(MetaEvent.SEQUENCE_TRACK_NAME, data="Main track")
        track1.addDeltaTime(0)
        track1.addMidiEvent(MidiEvent.PROGRAM_CHANGE, 1)
        delta_time_ticks = 0
        for measure in list_measures:
            for note, index in measure:
                duration_ticks = midi_score.lengthToTicks(note.duration * 1.0 / COMBINATIONS_DIVISION)
                if(note.pitch is None):
                    delta_time_ticks += duration_ticks
                else:
                    id_note = notes_ids[index]
                    height_int, octave = self.dico_notes[id_note].getRoundedHeightAndOctave()
                    key = int(note2tone(height_int, octave))
                    track1.addNote(delta_time_ticks, duration_ticks, key, volume=120)
                    delta_time_ticks = 0
        track1.addMetaEvent(MetaEvent.END_OF_TRACK)
        midi_bytes = midi_score.toBytes()
        return(midi_bytes)

    def _I_generateMidiScoreNoRhythm(self) -> bytes:
        # microseconds per quarter note
        tempo = 1000000
        midi_score = MidiScore(PPQ = 480)
        track0 = midi_score.addTrack()
        track0.addMetaEvent(MetaEvent.SET_TEMPO, data=tempo)
        track0.addMetaEvent(MetaEvent.END_OF_TRACK)
        track1 = midi_score.addTrack()
        track1.addMetaEvent(MetaEvent.SEQUENCE_TRACK_NAME, data="Main track")
        track1.addDeltaTime(0)
        track1.addMidiEvent(MidiEvent.PROGRAM_CHANGE, 1)
        delta_time_ticks = 0
        for Knote in range(len(self.chrono_notes)):
            for Pnote in range(len(self.chrono_notes[Knote][0])):
                id_note = self.chrono_notes[Knote][0][Pnote]
                length_s = self.dico_notes[id_note].getAnalogLength()
                duration_ticks = midi_score.lengthToTicks(length_s)
                if(self.dico_notes[id_note].isARest()):
                    delta_time_ticks += duration_ticks
                else:
                    int_height, octave = self.dico_notes[id_note].getRoundedHeightAndOctave()
                    key = int(note2tone(int_height, octave))
                    track1.addNote(delta_time_ticks, duration_ticks, key, volume=120)
                    delta_time_ticks = 0
        track1.addMetaEvent(MetaEvent.END_OF_TRACK)
        midi_bytes = midi_score.toBytes()
        return(midi_bytes)

    def performWithoutRhythm(self, notes: List[AnalogNote]) -> bytes:
        """
            Generate the midi without any rhythm interpolation.
            Just correcting the height of the note if necessary.
        """
        for note in notes:
            self.addNote(not note.is_a_rest, note.length_s, note.height_tone)
        self._A_MinimizeHeightError()
        self._B_AutoSetFifths()
        self._C_AutoSetClef()
        self._D_AutoTranslateOctave()
        return self._I_generateMidiScoreNoRhythm()

    def perform(self,
            notes: List[AnalogNote],
            format: str = "musicxml",
            author: str = "scorelisto",
            title: str = "Untitled") -> bytes:
        for note in notes:
            self.addNote(not note.is_a_rest, note.length_s, note.height_tone)
        self._A_MinimizeHeightError()
        self._B_AutoSetFifths()
        self._C_AutoSetClef()
        self._D_AutoTranslateOctave()
        self._E_FindConfigurationsForAllNotes()
        self._F_MaskCombinations()
        self._G_BuildGraph()
        self._H_GetOptimalPath()
        if(format.lower() == "musicxml"):
            return self._I_generateMusicXMLScore(author, title)
        elif(format.lower() == "midi"):
            return self._I_generateMidiScore()
        else:
            raise ValueError("Invalid input argument")
