import lxml.etree as ET
from enum import Enum
from functools import total_ordering
from dataclasses import dataclass, field
from typing import Dict, Tuple
import copy


@total_ordering
class NoteType(Enum):
    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"
    SIXTEENTH = "16th"
    THIRSTYSECOND = "32nd"

    def __eq__(self, other: "NoteType"):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other: "NoteType"):
        if self.__class__ is other.__class__:
            if(self.value == "whole"):
                return False
            elif(self.value == "half"):
                return other.value in ["whole"]
            elif(self.value == "quarter"):
                return other.value in ["whole", "half"]
            elif(self.value == "eighth"):
                return other.value in ["whole", "half", "quarter"]
            elif(self.value == "16th"):
                return other.value in ["whole", "half", "quarter", "eighth"]
            elif(self.value == "32nd"):
                return other.value in ["whole", "half", "quarter", "eighth", "16th"]
            else:
                return NotImplemented
        return NotImplemented


class ClefSign(Enum):
    G = "G"
    F = "F"
    C = "C"
    PERCUSSION = "PERCUSSION"
    TAB = "TAB"
    JIANPU = "JIANPU"
    NONE = "NONE"

class KeyFifths(Enum):
    ZERO = 0
    # Sharp
    ONE_SHARP = 1
    TWO_SHARPS = 2
    THREE_SHARPS = 3
    FOUR_SHARPS = 4
    FIVE_SHARPS = 5
    SIX_SHARPS = 6
    SEVEN_SHARPS = 7
    # Flat
    ONE_FLAT = -1
    TWO_FLATS = -2
    THREE_FLATS = -3
    FOUR_FLATS = -4
    FIVE_FLATS = -5
    SIX_FLATS = -6
    SEVEN_FLATS = -7

class KeyMode(Enum):
    MINOR = "minor"
    MAJOR = "major"

class Beam(Enum):
    BEGIN = "begin"
    CONTINUE = "continue"
    END = "end"

class SlurPlacement(Enum):
    BELOW = "below"
    ABOVE = "above"

class SlurType(Enum):
    START = "start"
    CONTINUE = "continue"
    STOP = "stop"

class PitchStep(Enum):
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    A = "A"
    B = "B"
    DO = "C"
    RE = "D"
    MI = "E"
    FA = "F"
    SOL = "G"
    LA = "A"
    SI = "B"

@dataclass
class NotePitch():
    step: PitchStep
    octave: int
    alter: int = None

@dataclass
class Slur():
    placement: SlurPlacement
    type: SlurType

@dataclass
class Note():
    duration: int = None
    type: NoteType = None
    pitch: NotePitch = None
    dot: bool = False
    triplet: bool = False
    beam: Dict[int, Beam] = field(default_factory=dict)
    slur: Dict[int, Slur] = field(default_factory=dict)


def CreateAttributesElement(divisions: int = None,
                            key_fifths: KeyFifths = None,
                            key_mode: KeyMode = None,
                            time_beats: int = None,
                            time_beattype: int = None,
                            clef_sign: ClefSign = None,
                            clef_line: int = None):
    if( (divisions is None) and 
        (key_fifths is None) and
        (key_mode is None) and
        (time_beats is None) and
        (time_beattype is None) and
        (clef_sign is None) and
        (clef_line is None)):
        raise RuntimeError("At Least one attributes parameter has to be set")
    
    if((key_mode is not None) and (key_fifths is None)):
        raise RuntimeError("you need to defined at least the 'fifths' key parameters")
    
    if(((time_beats is None) and (time_beattype is not None)) or 
       ((time_beats is not None) and (time_beattype is None))):
        raise RuntimeError("you need to defined any or both of the time parameters: 'beats' and 'beat-type'")
    
    if((clef_line is not None) and (clef_sign is None)):
        raise RuntimeError("you need to defined at least the 'sign' time parameters")
    
    e_attributes = ET.Element("attributes")
    ## divisions
    if(divisions is not None):
        e_divisions = ET.SubElement(e_attributes, "divisions")
        e_divisions.text = f"{divisions:d}"
    ## key
    if(key_fifths is not None):
        e_key = ET.SubElement(e_attributes, "key")
        e_fifths = ET.SubElement(e_key, "fifths")
        e_fifths.text = f"{key_fifths.value:d}"
        if(key_mode is not None):
            e_mode = ET.SubElement(e_key, "mode")
            e_mode.text = key_mode.value
    ## time
    if((time_beats is not None) and (time_beattype is not None)):
        e_time = ET.SubElement(e_attributes, "time")
        e_beats = ET.SubElement(e_time, "beats")
        e_beats.text = f"{time_beats:d}"
        e_beattype = ET.SubElement(e_time, "beat-type")
        e_beattype.text = f"{time_beattype:d}"
    ## clef
    if(clef_sign is not None):
        e_clef = ET.SubElement(e_attributes, "clef")
        e_sign = ET.SubElement(e_clef, "sign")
        e_sign.text = clef_sign.value
        if(clef_line is not None):
            e_line = ET.SubElement(e_clef, "line")
            e_line.text = f"{clef_line:d}"
    return(e_attributes)


class MusicXmlScore:
    def __init__(self, Title="Untitled", Composer="Unknown"):
        self.header1 = b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        self.header2 = b'<!DOCTYPE score-partwise PUBLIC\n\t"-//Recordare//DTD MusicXML 4.0 Partwise//EN"\n\t"http://www.musicxml.org/dtds/partwise.dtd">\n'
        # score-partwise
        self.e_scorepartwise = ET.Element("score-partwise", attrib={"version": "4.0"})
        # Title
        self.e_work = ET.SubElement(self.e_scorepartwise, "work")
        e_worktitle = ET.SubElement(self.e_work, "work-title")
        e_worktitle.text = Title
        # Composer
        self.e_identification = ET.SubElement(self.e_scorepartwise, "identification")
        e_creator = ET.SubElement(self.e_identification, "creator", attrib={"type": "composer"})
        e_creator.text = Composer
        e_rights = ET.SubElement(self.e_identification, "rights")
        e_rights.text = "Copyright © 2018 ScoreListo, Inc."
        # part-list
        self.e_partlist = ET.SubElement(self.e_scorepartwise, "part-list")
        
    def getPartFromID(self, id_part: str) -> ET._Element:
        for e_part in self.e_scorepartwise.findall("part"):
            if(e_part.attrib['id'] == id_part):
                return(e_part)
        raise ValueError(f"The part with id:{id:d} does not exists")
    
    def partIDAlreadyExists(self, id_part: str) -> bool:
        for e_scorepart in self.e_partlist.iterchildren("score-part"):
            if(e_scorepart.attrib['id'] == id_part):
                return(True)
        return(False)
    
    def updateTitle(self, title :str):
        self.e_work.find("work-title").text = title
    
    def _getNumberOfMeasures(self, part: ET._Element) -> int:
        return(len(part.findall("measure")))
    
    def _getTimeElement(self, e_measure: ET._Element) -> ET._Element:
        if(e_measure is None):
            return(None)
        e_attributes = e_measure.find("attributes")
        if(e_attributes is None):
            return(None)
        e_time = e_attributes.find("time")
        if(e_time is None):
            return(None)
        return(e_time)
    
    def _getDivisionsElement(self, e_measure: ET._Element) -> ET._Element:
        if(e_measure is None):
            return(None)
        e_attributes = e_measure.find("attributes")
        if(e_attributes is None):
            return(None)
        e_divisions = e_attributes.find("divisions")
        if(e_divisions is None):
            return(None)
        return(e_divisions)
    
    def _getRelatedDivisionsElement(self, e_measure: ET._Element) -> ET._Element:
        while(e_measure is not None):
            e_divisions = self._getDivisionsElement(e_measure)
            if(e_divisions is not None):
                return(e_divisions)
            e_measure = e_measure.getprevious()
        return(None)
    
    def _getRelatedTimeElement(self, e_measure: ET._Element) -> ET._Element:
        while(e_measure is not None):
            e_time = self._getTimeElement(e_measure)
            if(e_time is not None):
                return(e_time)
            e_measure = e_measure.getprevious()
        return(None)
    
    def _measureSumNoteDuration(self, e_measure: ET._Element) -> int:
        duration_sum = 0
        for e_note in e_measure.findall("note"):
            e_duration = e_note.find("duration")
            if(e_duration is None):
                raise RuntimeError("A note has been defined without setting the duration parameter")
            duration_sum += int(e_duration.text)
        return(duration_sum)
    
    def _calculateMaximumDuration(self, e_measure: ET._Element) -> int:
        e_divisions = self._getRelatedDivisionsElement(e_measure)
        e_time = self._getRelatedTimeElement(e_measure)
        if((e_divisions is None) or (e_time is None)):
            raise RuntimeError("The parameters chosen for 'divisions' and 'time' are not set")
        divisions = int(e_divisions.text)
        beats = int(e_time.find("beats").text)
        beattype = int(e_time.find("beat-type").text)
        duration_per_measure = divisions * beats * (4.0 / beattype)
        if(not duration_per_measure.is_integer()):
            raise RuntimeError("The parameters chosen for 'divisions' and 'time' are incorrects")
        return(int(duration_per_measure))
    
    def _isTheMeasureCompleted(self, e_measure: ET._Element) -> bool:
        duration_per_measure = self._calculateMaximumDuration(e_measure)
        duration_last_measure = self._measureSumNoteDuration(e_measure)
        return(duration_per_measure == duration_last_measure)
    
    def isReady(self) -> bool:
        parts = self.e_scorepartwise.findall("part")
        list_nb_measures = [self._getNumberOfMeasures(part) for part in parts]
        if(len(set(list_nb_measures)) != 1):
            # 0   => number of parts = 0
            # > 1 => different number of measures in each parts
            return(False)
        if(list_nb_measures[0]==0):
            return(False)
        for part in parts:
            for measure in part.iterchildren("measure"):
                if(not self._isTheMeasureCompleted(measure)):
                    return(False)
        return(True)
    
    def addPart(self, id_part: str, partname: str, instrumentname: str = "Piano"):
        # Check if the Id does not already exists
        if self.partIDAlreadyExists(id_part):
            raise ValueError(f"The id already exists => {id_part}")
        e_scorepart = ET.SubElement(self.e_partlist, "score-part", attrib={"id": id_part})
        e_partname = ET.SubElement(e_scorepart, "part-name")
        e_partname.text = partname
        e_scoreinstrument = ET.SubElement(e_scorepart, "score-instrument", attrib={"id": f"{id_part}-I1"})
        e_instrumentname = ET.SubElement(e_scoreinstrument, "instrument-name")
        e_instrumentname.text = instrumentname
        ET.SubElement(self.e_scorepartwise, "part", attrib={"id": id_part})
    
    def addMeasure(self, id_part: str) -> ET._Element:
        e_part = self.getPartFromID(id_part)
        if(e_part.getchildren() != []):
            e_lastmeasure = e_part.getchildren()[-1]
            nextnumber = int(e_lastmeasure.attrib['number']) + 1
            if(not self._isTheMeasureCompleted(e_lastmeasure)):
                raise RuntimeError("Trying to add a new measure whereas the last measure is not yet completed")
        else:
            nextnumber = 1
        e_newmeasure = ET.SubElement(e_part, "measure", attrib={"number": f"{nextnumber:d}"})
        return(e_newmeasure)
    
    def _checkTimeBetweenMeasures(self, id_part: str, ind_measure: int, E_time: ET._Element):
        for e_part in self.e_scorepartwise.iterchildren("part"):
            if(e_part.attrib['id'] != id_part):
                measures = e_part.getchildren()
                if(len(measures) > ind_measure):
                    e_measure = measures[ind_measure]
                    e_attributes = e_measure.find("attributes")
                    if(e_attributes is None):
                        if(e_measure.find("note") is not None):
                            raise RuntimeError("Trying to add a time signature different than orther parts")
                    else:
                        e_time = e_attributes.find("time")
                        if(e_time is None):
                            raise RuntimeError("Trying to add a time signature different than orther parts")
                        beats_A = int(e_time.find('beats').text)
                        beattype_A = int(e_time.find('beat-type').text)
                        beats_B = int(E_time.find('beats').text)
                        beattype_B = int(E_time.find('beat-type').text)
                        if((beats_A != beats_B) or (beattype_A != beattype_B)):
                            raise RuntimeError("Trying to add a time signature different than orther parts")
    
    def _getLastMeasure(self, id_part: str) -> Tuple[int, ET._Element]:
        e_part = self.getPartFromID(id_part)
        e_measures = e_part.getchildren()
        if(e_measures == []):
            raise RuntimeError(f"Any measure created in part having id: {id_part}")
        return((len(e_measures) - 1, e_measures[-1]))
    
    def addAttributes(self, id_part: str, e_attributes: ET._Element):
        ind_measure, e_measure = self._getLastMeasure(id_part)
        if(e_measure.find("note") is not None):
            raise RuntimeError("Trying to add a attributes whereas a note has already been created... too late")
        if(e_measure.find("attributes") is not None):
            raise RuntimeError("Attributes have already been defined")
        if(e_attributes.tag != 'attributes'):
            raise RuntimeError(f"Wrong input tag: {e_attributes.tag} instead of 'attributes'")
        if(ind_measure == 0):
            if((e_attributes.find("divisions") is None) or 
               (e_attributes.find("key") is None) or 
               (e_attributes.find("time") is None) or 
               (e_attributes.find("clef") is None)):
                raise RuntimeError("All attributes have to be set for the first measure")
        else:
            if((e_attributes.find("divisions") is None) and 
               (e_attributes.find("key") is None) and 
               (e_attributes.find("time") is None) and 
               (e_attributes.find("clef") is None)):
                raise RuntimeError("You need to defined at least one attribute parameter")
        if(e_attributes.find("time") is not None):
            self._checkTimeBetweenMeasures(id_part, ind_measure, e_attributes.find("time"))
        e_measure.append(copy.deepcopy(e_attributes))
    
    def _createNoteElement(self, note: Note) -> ET._Element:
        e_note = ET.Element("note")
        if(note.pitch is None):
            ET.SubElement(e_note, "rest")
        else:
            e_pitch = ET.SubElement(e_note, "pitch")
            e_step = ET.SubElement(e_pitch, "step")
            e_step.text = note.pitch.step.value
            if(note.pitch.alter is not None):
                e_alter = ET.SubElement(e_pitch, "alter")
                e_alter.text = "%d" % note.pitch.alter
            e_octave = ET.SubElement(e_pitch, "octave")
            e_octave.text = "%d" % note.pitch.octave
        
        e_duration = ET.SubElement(e_note, "duration")
        e_duration.text = "%d" % note.duration
        e_type = ET.SubElement(e_note, "type")
        e_type.text = note.type.value
        if(note.dot):
            ET.SubElement(e_note, "dot")
        if(note.triplet):
            e_timemod = ET.SubElement(e_note, "time-modification")
            e_actualnotes = ET.SubElement(e_timemod, "actual-notes")
            e_actualnotes.text = "3"
            e_normalnotes = ET.SubElement(e_timemod, "normal-notes")
            e_normalnotes.text = "2"
        for number, action in note.beam.items():
            ET.SubElement(e_note, "beam", attrib={'number': f"{number:d}"}).text = action.value
        for number, params in note.slur.items():
            e_notations = ET.SubElement(e_note, "notations")
            attributes = {
                "number": f"{number:d}",
                "placement": params.placement.value,
                "type": params.type.value
            }
            ET.SubElement(e_notations, "slur", attrib=attributes)
        return(e_note)

    def addNote(self, id_part: str, note: Note):
        _, e_measure = self._getLastMeasure(id_part)
        current_duration = self._measureSumNoteDuration(e_measure)
        max_duration = self._calculateMaximumDuration(e_measure)
        duration = note.duration
        if((current_duration + duration) > max_duration):
            raise RuntimeError("With this note duration, the maximum duration of the measure is exceeded")
        e_note = self._createNoteElement(note)
        e_measure.append(copy.deepcopy(e_note))
    
    def _beamBeautifier(self):
        for e_part in self.e_scorepartwise.findall("part"):
            for e_measure in e_part.findall("measure"):
                divisions = int(self._getRelatedDivisionsElement(e_measure).text)
                e_notes = e_measure.findall("note")
                # Delete existing beams information
                for e_note in e_notes:
                    e_beams = e_note.findall("beam")
                    for e_beam in e_beams:
                        e_note.remove(e_beam)
                beams_start = {1: False, 2: False, 3: False}
                duration_sum = 0
                for ind in range(0, len(e_notes)):
                    current_type = NoteType(e_notes[ind].find("type").text)
                    duration_sum += int(e_notes[ind].find("duration").text)
                    # Search the index where to insert beams (has to be before the 'notations')
                    insert_ind = len(e_notes[ind].getchildren())
                    for ind_child, e_child in enumerate(e_notes[ind].getchildren()):
                        if e_child.tag == "notations":
                            insert_ind = ind_child
                            break
                    # Conditions to stop the beam
                    if (ind == (len(e_notes) - 1)) or ((duration_sum % divisions) == 0):
                        next_type = NoteType.WHOLE
                        next_ispitch = False
                    else:
                        next_type = NoteType(e_notes[ind + 1].find("type").text)
                        next_ispitch = e_notes[ind + 1].find("rest") is None
                    # Iterate over the 3 beams level possible
                    for no_beam, note_type in [(1, NoteType.EIGHTH), (2, NoteType.SIXTEENTH), (3, NoteType.THIRSTYSECOND)]:
                        if (current_type <= note_type) and (next_type <= note_type) and next_ispitch:
                            beam = Beam.CONTINUE if beams_start[no_beam] else Beam.BEGIN
                            e_beam = ET.Element("beam", attrib={'number': f"{no_beam}"})
                            e_beam.text = beam.value
                            e_notes[ind].insert(insert_ind, e_beam)
                            insert_ind += 1
                            beams_start[no_beam] = True
                        elif beams_start[no_beam]:
                            e_beam = ET.Element("beam", attrib={'number': f"{no_beam}"})
                            e_beam.text = Beam.END.value
                            e_notes[ind].insert(insert_ind, e_beam)
                            insert_ind += 1
                            beams_start[no_beam] = False
                    
    def toString(self, beautify_beams: bool = True) -> bytes:
        if(self.isReady()):
            if beautify_beams:
                self._beamBeautifier()
            return(self.header1 + self.header2 + ET.tostring(self.e_scorepartwise, pretty_print=True))
        else:
            raise RuntimeError("The partition is not ready to be written")

    def toFile(self, filepath: str, beautify_beams: bool = True):
        with open(filepath, mode='wb') as f:
            f.write(self.toString(beautify_beams))


if __name__ == "__main__":
    # Create a partition
    ScoreXML = MusicXmlScore(Title="Unexpected Response", Composer="Fidelano Mizzilani")
    # Create the parts
    IdPart1 = "P1"
    IdPart2 = "P2"
    part1 = ScoreXML.addPart(IdPart1, "Trumpet")
    part2 = ScoreXML.addPart(IdPart2, "Saxophone")
    
    # Attributes
    E_attributes = CreateAttributesElement(
                            divisions=24,
                            key_fifths=KeyFifths.THREE_FLATS,
                            key_mode="major",
                            time_beats=4,
                            time_beattype=4,
                            clef_sign=ClefSign.G,
                            clef_line=2)
    
    # Part 1
    measure = ScoreXML.addMeasure(IdPart1)
    ScoreXML.addAttributes(IdPart1, E_attributes)
    ScoreXML.addNote(IdPart1, Note(24, NoteType.QUARTER, NotePitch(PitchStep.C, 4)))
    ScoreXML.addNote(IdPart1, Note(24, NoteType.QUARTER, NotePitch(PitchStep.C, 4)))
    ScoreXML.addNote(IdPart1, Note(12, NoteType.EIGHTH, NotePitch(PitchStep.B, 4), beam={1: Beam.BEGIN}))
    ScoreXML.addNote(IdPart1, Note(12, NoteType.EIGHTH, NotePitch(PitchStep.A, 4), beam={1: Beam.END}))
    ScoreXML.addNote(IdPart1, Note(6, NoteType.SIXTEENTH, NotePitch(PitchStep.F, 4), beam={1: Beam.BEGIN, 2: Beam.BEGIN}))
    ScoreXML.addNote(IdPart1, Note(6, NoteType.SIXTEENTH, NotePitch(PitchStep.F, 4), beam={1: Beam.CONTINUE, 2: Beam.CONTINUE}))
    ScoreXML.addNote(IdPart1, Note(6, NoteType.SIXTEENTH, NotePitch(PitchStep.F, 4), beam={1: Beam.CONTINUE, 2: Beam.CONTINUE}))
    ScoreXML.addNote(IdPart1, Note(6, NoteType.SIXTEENTH, NotePitch(PitchStep.F, 4), beam={1: Beam.END, 2: Beam.END}, slur={1: Slur(SlurPlacement.ABOVE, SlurType.START)}))
    
    measure = ScoreXML.addMeasure(IdPart1)
    ScoreXML.addNote(IdPart1, Note(24, NoteType.QUARTER, NotePitch(PitchStep.C, 4), slur={2: Slur(SlurPlacement.BELOW, SlurType.START), 1: Slur(SlurPlacement.ABOVE, SlurType.CONTINUE)}))
    ScoreXML.addNote(IdPart1, Note(24, NoteType.QUARTER, NotePitch(PitchStep.C, 4), slur={2: Slur(SlurPlacement.BELOW, SlurType.STOP), 1: Slur(SlurPlacement.ABOVE, SlurType.STOP)}))
    ScoreXML.addNote(IdPart1, Note(24, NoteType.QUARTER))
    ScoreXML.addNote(IdPart1, Note(24, NoteType.QUARTER, NotePitch(PitchStep.C, 4)))
    
    ##### Part 2
    measure = ScoreXML.addMeasure(IdPart2)
    ScoreXML.addAttributes(IdPart2, E_attributes)
    ScoreXML.addNote(IdPart2, Note(12, NoteType.EIGHTH))
    ScoreXML.addNote(IdPart2, Note(12, NoteType.EIGHTH, NotePitch(PitchStep.A, 4)))
    ScoreXML.addNote(IdPart2, Note(24, NoteType.QUARTER, NotePitch(PitchStep.F, 4)))
    ScoreXML.addNote(IdPart2, Note(24, NoteType.QUARTER, NotePitch(PitchStep.D, 4, -1)))
    ScoreXML.addNote(IdPart2, Note(12, NoteType.EIGHTH, NotePitch(PitchStep.E, 4), beam={1: Beam.BEGIN}))
    ScoreXML.addNote(IdPart2, Note(6, NoteType.SIXTEENTH, NotePitch(PitchStep.F, 4), beam={1: Beam.CONTINUE, 2: Beam.BEGIN}))
    ScoreXML.addNote(IdPart2, Note(6, NoteType.SIXTEENTH, NotePitch(PitchStep.G, 4), beam={1: Beam.END, 2: Beam.END}))
    
    E_attributes_2 = CreateAttributesElement(
                        divisions=24,
                        key_fifths=-3,
                        key_mode="major",
                        clef_sign='F',
                        clef_line=4)
    
    measure = ScoreXML.addMeasure(IdPart2)
    ScoreXML.addAttributes(IdPart2, E_attributes_2)
    ScoreXML.addNote(IdPart2, Note(12, NoteType.EIGHTH, NotePitch(PitchStep.C, 4)))
    ScoreXML.addNote(IdPart2, Note(36, NoteType.QUARTER, NotePitch(PitchStep.C, 4), dot=True))
    ScoreXML.addNote(IdPart2, Note(24, NoteType.QUARTER, NotePitch(PitchStep.C, 2), slur={1: Slur(SlurPlacement.BELOW, SlurType.START)}))
    ScoreXML.addNote(IdPart2, Note(24, NoteType.QUARTER, NotePitch(PitchStep.B, 3), slur={1: Slur(SlurPlacement.BELOW, SlurType.STOP)}))
    
    ScoreXML.toFile("test.xml")
