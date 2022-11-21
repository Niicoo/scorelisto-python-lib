import struct
from typing import Union, Tuple, List
from enum import Enum


# Typing
MetaData = Union[
    int,                            # SEQUENCE_NUMBER, MIDI_CHANNEL_PREFIX, SET_TEMPO
    str,                            # TEXT_EVENT, COPYRIGHT_NOTICE, SEQUENCE_TRACK_NAME, INSTRUMENT_NAME, LYRIC, MARKER, CUE_POINT
    None,                           # END_OF_TRACK
    bytes,                          # SEQUENCER_SPECIFIC_META_EVENT
    Tuple[int, int, int, int, int], # SMTPE_OFFSET
    Tuple[int, int, int, int],      # TIME_SIGNATURE
    Tuple[int, int],                # KEY_SIGNATURE
]

MidiData = Union[
    Tuple[int, int],                # NOTE_OFF, NOTE_ON, POLYPHONIC_KEY_PRESSURE, CONTROLLER_CHANGE, PITCH_BEND
    int,                            # PROGRAM_CHANGE, CHANNEL_KEY_PRESSURE, LOCAL_CONTROL, MONO_MODE_ON
    None,                           # ALL_SOUND_OFF, RESET_ALL_CONTROLLERS, ALL_NOTES_OFF, OMNI_MODE_OFF, OMNI_MODE_ON, POLY_MODE_ON
]


class MetaEvent(Enum):
    SEQUENCE_NUMBER = 0
    TEXT_EVENT = 1
    COPYRIGHT_NOTICE = 2
    SEQUENCE_TRACK_NAME = 3
    INSTRUMENT_NAME = 4
    LYRIC = 5
    MARKER = 6
    CUE_POINT = 7
    MIDI_CHANNEL_PREFIX = 32
    END_OF_TRACK = 47
    SET_TEMPO = 81
    SMTPE_OFFSET = 84
    TIME_SIGNATURE = 88
    KEY_SIGNATURE = 89
    SEQUENCER_SPECIFIC_META_EVENT = 127


class MidiEvent(Enum):
    NOTE_OFF = 128
    NOTE_ON = 144
    POLYPHONIC_KEY_PRESSURE = 160
    CONTROLLER_CHANGE = 176
    PROGRAM_CHANGE = 192
    CHANNEL_KEY_PRESSURE = 208
    PITCH_BEND = 224
    ALL_SOUND_OFF = 176
    RESET_ALL_CONTROLLERS = 176
    LOCAL_CONTROL = 176
    ALL_NOTES_OFF = 176
    OMNI_MODE_OFF = 176
    OMNI_MODE_ON = 176
    MONO_MODE_ON = 176
    POLY_MODE_ON = 176


class MidiTrackMode(Enum):
    READY = 0
    WAITING_FOR_EVENT = 0


def int2VLQ(number: int) -> bytes:
    bin_str = bin(number)[2:]
    nb_bits = len(bin_str)
    nb_bytes = int(nb_bits / 7) + (nb_bits % 7 > 0)
    bytes_array = b''
    for k in range(0, nb_bytes):
        if(k == 0):
            nb_bits_last_byte = nb_bits % 7
            if(nb_bits_last_byte == 0):
                nb_bits_last_byte = 7
            nb_zeros = 7 - nb_bits_last_byte
            if(k == (nb_bytes - 1)):
                byte = '0' + nb_zeros * '0' + bin_str
            else:
                byte = '1' + nb_zeros * '0' + bin_str[0 : nb_bits_last_byte]
        elif(k == (nb_bytes - 1)):
            byte = '0' + bin_str[-7:]
        else:
            byte = '1' + bin_str[- (k + 1) * 7 : - k * 7]
        bytes_array += struct.pack('B', int(byte, 2))
    return(bytes_array)


class MidiTrack:
    def __init__(self):
        self.bytes_array: bytes = b''
        self.mode: MidiTrackMode = MidiTrackMode.READY
    
    def toBytes(self) -> bytes:
        if(self.mode == MidiTrackMode.READY):
            midi_bytes = "MTrk".encode('ascii')
            midi_bytes += struct.pack('>I', self.getTrackSize())
            midi_bytes += self.bytes_array
            return(midi_bytes)
        else:
            raise RuntimeError("Uncompleted track: a delta time has been defined with no event following")
    
    def getTrackSize(self) -> int:
        return(len(self.bytes_array))
    
    def addNote(self, delta_time: int, duration: int, key: int, volume: int, no_channel: int = 0):
        self.addDeltaTime(delta_time)
        self.addMidiEvent(MidiEvent.NOTE_ON,[key, volume], no_channel)
        self.addDeltaTime(duration)
        self.addMidiEvent(MidiEvent.NOTE_OFF,[key, 64], no_channel)
    
    def addDeltaTime(self, nb_ticks: int):
        if(self.mode == MidiTrackMode.READY):
            self.bytes_array += int2VLQ(nb_ticks)
            self.mode = MidiTrackMode.WAITING_FOR_EVENT
        else:
            raise RuntimeError("A delta time has already been defined, an event is now expected")
    
    def addMidiEvent(self, event: MidiEvent, data: MidiData = None, no_channel: int = 0):
        if(self.mode != MidiTrackMode.WAITING_FOR_EVENT):
            raise RuntimeError("You first need to defined a delta time before creating an event")
        event_bytes = struct.pack('B', event.value + no_channel)
        if(event in [MidiEvent.NOTE_OFF, MidiEvent.NOTE_ON, MidiEvent.POLYPHONIC_KEY_PRESSURE, MidiEvent.CONTROLLER_CHANGE, MidiEvent.PITCH_BEND]):
            event_bytes += struct.pack('B', data[0])
            event_bytes += struct.pack('B', data[1])
        elif(event in [MidiEvent.PROGRAM_CHANGE, MidiEvent.CHANNEL_KEY_PRESSURE]):
            event_bytes += struct.pack('B', data)
        elif(event == MidiEvent.ALL_SOUND_OFF):
            event_bytes += struct.pack('B', 120)
        elif(event == MidiEvent.RESET_ALL_CONTROLLERS):
            event_bytes += struct.pack('B', 121)
        elif(event == MidiEvent.LOCAL_CONTROL):
            event_bytes += struct.pack('B', 122)
            event_bytes += struct.pack('B', data)
        elif(event == MidiEvent.ALL_NOTES_OFF):
            event_bytes += struct.pack('B', 123)
        elif(event == MidiEvent.OMNI_MODE_OFF):
            event_bytes += struct.pack('B', 124)
        elif(event == MidiEvent.OMNI_MODE_ON):
            event_bytes += struct.pack('B', 125)
        elif(event == MidiEvent.MONO_MODE_ON):
            event_bytes += struct.pack('B', 126)
            event_bytes += struct.pack('B', data)
        elif(event == MidiEvent.POLY_MODE_ON):
            event_bytes += struct.pack('B', 127)
        else:
            raise ValueError(f"Wrong input event: {event}")
        self.bytes_array += event_bytes
        self.mode = MidiTrackMode.READY

    def addMetaEvent(self, event: Union[str, MetaEvent], data: MetaData = None):
        """
        You don't need to create a Delta Time, is it automatically created.
        """
        event = event if isinstance(event, MetaEvent) else MetaEvent[event]
        if(self.mode != MidiTrackMode.READY):
            raise RuntimeError("A previous delta time has been created without creating a Midi Event")
        event_bytes = b''
        event_bytes += struct.pack('B', 255)
        event_bytes += struct.pack('B', event.value)
        if(event == MetaEvent.SEQUENCE_NUMBER):
            event_bytes += struct.pack('B', 2)
            event_bytes += struct.pack('>H', data)
        elif(event in [ MetaEvent.TEXT_EVENT, MetaEvent.COPYRIGHT_NOTICE, MetaEvent.SEQUENCE_TRACK_NAME,
                        MetaEvent.INSTRUMENT_NAME, MetaEvent.LYRIC, MetaEvent.MARKER, MetaEvent.CUE_POINT]):
            event_bytes += int2VLQ(len(data))
            event_bytes += data.encode('ascii')
        elif(event == MetaEvent.MIDI_CHANNEL_PREFIX):
            event_bytes += struct.pack('B', 1)
            event_bytes += struct.pack('B', data)
        elif(event == MetaEvent.END_OF_TRACK):
            event_bytes += struct.pack('B', 0)
        elif(event == MetaEvent.SET_TEMPO):
            event_bytes += struct.pack('B', 3)
            event_bytes += struct.pack('B', data.to_bytes(3, 'big')[0])
            event_bytes += struct.pack('B', data.to_bytes(3, 'big')[1])
            event_bytes += struct.pack('B', data.to_bytes(3, 'big')[2])
        elif(event == MetaEvent.SMTPE_OFFSET):
            event_bytes += struct.pack('B', 5)
            event_bytes += struct.pack('B', data[0]) # hh
            event_bytes += struct.pack('B', data[1]) # mm
            event_bytes += struct.pack('B', data[2]) # ss
            event_bytes += struct.pack('B', data[3]) # fr
            event_bytes += struct.pack('B', data[4]) # ff
        elif(event == MetaEvent.TIME_SIGNATURE):
            event_bytes += struct.pack('B', 4)
            event_bytes += struct.pack('B', data[0]) # nn
            event_bytes += struct.pack('B', data[1]) # dd
            event_bytes += struct.pack('B', data[2]) # cc
            event_bytes += struct.pack('B', data[3]) # bb
        elif(event == MetaEvent.KEY_SIGNATURE):
            event_bytes += struct.pack('B', 2)
            event_bytes += struct.pack('b', data[0]) # sf
            event_bytes += struct.pack('B', data[1]) # mi
        elif(event == MetaEvent.SEQUENCER_SPECIFIC_META_EVENT):
            event_bytes += int2VLQ(len(data))
            event_bytes += data
        else:
            raise ValueError(f"Wrong value of event name: {event}")
        self.addDeltaTime(0)
        self.bytes_array += event_bytes
        

class MidiScore:
    def __init__(self, PPQ: int = 480):
        self.tracks: List[MidiTrack] = []
        self.PPQ: int = PPQ
        self.format: int = 1
    
    def lengthToTicks(self, length_s: float):
        return int(round(length_s * self.PPQ))

    def getNumberOfTracks(self) -> int:
        return(len(self.tracks))

    def addTrack(self) -> MidiTrack:
        self.tracks.append(MidiTrack())
        return self.tracks[-1]

    def toBytes(self) -> bytes:
        # Header
        header_bytes = "MThd".encode('ascii')
        header_bytes += struct.pack('>I', 6)
        header_bytes += struct.pack('>H', self.format)
        header_bytes += struct.pack('>H', self.getNumberOfTracks())
        header_bytes += struct.pack('>H', self.PPQ)
        # Tracks
        tracks_bytes = b''
        for track in self.tracks:
            tracks_bytes += track.toBytes()
        return(header_bytes + tracks_bytes)

if __name__ == "__main__":
    midi_score = MidiScore(PPQ = 480)
    track0 = midi_score.addTrack()
    track0.addMetaEvent(MetaEvent.SET_TEMPO, data=1000000) # microseconds per quarter note
    track0.addMetaEvent(MetaEvent.END_OF_TRACK)
    track1 = midi_score.addTrack()
    track1.addMetaEvent(MetaEvent.SEQUENCE_TRACK_NAME, data="Main track")
    track1.addDeltaTime(0)
    track1.addMidiEvent(MidiEvent.PROGRAM_CHANGE, 1)

    volume = 120
    delta_time = 0
    # Add a note
    duration = midi_score.lengthToTicks(1.0)
    track1.addNote(delta_time, duration, 36, volume)
    delta_time = 0
    # Add a note
    duration = midi_score.lengthToTicks(1.0)
    track1.addNote(delta_time, duration, 38, volume)
    delta_time = 0
    # Add a note
    duration = midi_score.lengthToTicks(1.0)
    track1.addNote(delta_time, duration, 40, volume)
    delta_time = 0
    # Add a rest
    delta_time += midi_score.lengthToTicks(3.0)
    # Add a note
    duration = midi_score.lengthToTicks(1.0)
    track1.addNote(delta_time, duration, 42, volume)
    delta_time = 0
    track1.addMetaEvent(MetaEvent.END_OF_TRACK)
    midi_bytes = midi_score.toBytes()
    with open("midi_test.mid", "wb") as f:
        f.write(midi_bytes)
