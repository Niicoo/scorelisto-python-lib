#!/usr/bin/env python
import pathlib
import sys
import os
import logging
from shutil import which
if "__file__" in globals():
    LIB_PATH = pathlib.Path(pathlib.Path(__file__).absolute().parent.parent, "lib").as_posix()
else:
    # assuming we ran a terminal at the root of the project
    LIB_PATH = pathlib.Path(pathlib.Path(".").absolute(), "lib").as_posix()
sys.path.append(LIB_PATH)
from scorelisto.audiofile_readers import LibrosaReader
from scorelisto.pitch_detectors import McLeodDetector
from scorelisto.note_detectors import HistogramNoteDetector
from scorelisto.rhythm_detectors import DigitalPartition
from scorelisto.tools import compute_energy, freq2tone
import numpy as np
import matplotlib.pyplot as plt
# Typing
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Get Logger
logger = logging.getLogger("Test")


def is_tool(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


# General parameters
input_path = "./samples/whistling_mono_8khz.wav"
mscore_executable = None

# Pitch Detector parameters
windows_size_s = 20e-3
step_size_s = 1e-3
sample_rate_hz = 44100

# Rhythm Detector parameters
delaymin = 60 / 200 # Conversion from BPM to delay
delaymax = 60 / 40 # Conversion from BPM to delay
maxdelayvar = 0.5
errormax = 10
combs_to_mask = []

# Logging level
logging_level = logging.INFO

# Get output file paths
in_path_p = pathlib.Path(input_path)
musicxml_path = pathlib.Path(in_path_p.parent, in_path_p.stem + ".musicxml").as_posix()
png_path = pathlib.Path(in_path_p.parent, in_path_p.stem).as_posix()


# Set Logging Level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
for logname in ["Test", "scorelisto"]:
    log = logging.getLogger(logname)
    log.setLevel(logging_level)
    log.addHandler(handler)


# Detecting Musescore if required
if mscore_executable is None:
    for executable in ["mscore", "musescore", "musescore3", "mscore-portable", "MuseScore3.exe"]:
        if is_tool(executable):
            mscore_executable = executable
            logger.info(f"Musescore executable detected: {mscore_executable}")
            break
        else:
            logger.debug(f"Musescore executable '{executable}' not available")
elif is_tool(mscore_executable):
    logger.info(f"Musescore executable detected: {mscore_executable}")
else:
    logger.warning(f"The path of musescore you provided ({mscore_executable}) is not valid")
    
# Compute all the audio files
audio_reader = LibrosaReader()
note_detector = HistogramNoteDetector()
logger.info(f"Processing input file: {input_path}")
logger.info(f"Output file path: {musicxml_path}")



# Reading and extracting pitch of the audio file
audio_reader.read(input_path, sample_rate_hz)
data_min, data_max = min(audio_reader.getData()), max(audio_reader.getData())
normalization_factor = data_max - data_min
pitch_detector = McLeodDetector(audio_reader.getSampleRate())
pitches = []
energies = []
for chunk in audio_reader.iterate(windows_size_s, step_size_s):
    pitch = pitch_detector.perform(chunk)
    energy = compute_energy(chunk / normalization_factor)
    pitches.append(pitch)
    energies.append(energy)
pitches = freq2tone(np.array(pitches, dtype='double'))
energies = 10.0 * np.log10(np.array(energies, dtype='double'))

# Splitting the notes
notes, params = note_detector.perform(pitches, step_size_s, energies)


# Plot the intermediate results
ax_length = 10 # seconds
nb_axs = int(np.ceil((len(pitches) *  step_size_s) / ax_length))
min_energy, max_energy = np.min(energies), np.max(energies)

fig = plt.figure(figsize=(15, nb_axs * 4))
for ind_ax in range(0, nb_axs):
    ax = fig.add_subplot(nb_axs, 1, ind_ax + 1)
    ax.grid()
    ind_start = int(np.round((ind_ax * ax_length) / step_size_s))
    ind_stop = int(np.round(((ind_ax + 1) * ax_length) / step_size_s))
    ax.set_xlim([ind_start * step_size_s - 0.05, ind_stop * step_size_s + 0.05])
    ind_stop = min(ind_stop, len(pitches))
    t = np.arange(ind_start, ind_stop) * step_size_s
    ax.plot(t, pitches[ind_start : ind_stop])
    ax.set_ylabel("Pitch [tone]")
    # Energy
    ax2 = ax.twinx()
    ax2.set_ylabel("Energy [dB]")
    energy_marging = 0.05 * (max_energy - min_energy)
    ax2.fill_between(t, np.ones(ind_stop - ind_start) * min_energy - energy_marging, energies[ind_start : ind_stop], color='g', alpha=.2, linewidth=0)
    ax2.set_ylim([min_energy - energy_marging, max_energy + energy_marging])
    ax2.axhline(params["tOn"], linestyle = "--")
    ax2.axhline(params["tOff"], linestyle = "--")
    # Notes
    tnote_start = params["offset"]
    for note in notes:
        tnote_stop = tnote_start + note.length_s
        if((t[0] <= tnote_start <= t[-1]) or (t[0] <= tnote_stop <= t[-1])) and not note.is_a_rest:
            ax.plot([tnote_start, tnote_stop], [note.height_tone, note.height_tone], linewidth=2.0, color='r')
        tnote_start = tnote_stop
fig.savefig(png_path + "_1.png")
plt.close(fig)


# Detecting Rhythm
digitalizer = DigitalPartition(delaymin, delaymax, maxdelayvar, errormax, combs_to_mask)
file_bytes = digitalizer.perform(notes, "musicxml")
with open(musicxml_path, "wb") as f:
    f.write(file_bytes)
if(mscore_executable is not None):
    out_file_p = pathlib.Path(musicxml_path)
    pdf_path = pathlib.Path(out_file_p.parent, out_file_p.stem + ".pdf").as_posix()
    logger.info(f"Generating PDF file in : {pdf_path}")
    os.system(f"{mscore_executable} {musicxml_path} -o {pdf_path}")


# def time_to_note(digitalizer: DigitalPartition, idnote: str):
#     length_s = 0
#     for notes in digitalizer.chrono_notes:
#         if(notes[0][0] == idnote):
#             return length_s
#         elif len(notes) == 2:
#             if(notes[1][0] == idnote):
#                 return length_s
#         elif len(notes[0]) == 2:
#             if notes[0][1] == idnote:
#                 length_s += digitalizer.dico_notes[notes[0][0]].analoglength
#                 return length_s
#         length_s += digitalizer.dico_notes[notes[-1][0]].analoglength
#     raise RuntimeError("Should not reach this point, id note does not exists")

# def plotConfigurations(ax: Axes, digitalizer: DigitalPartition, id_note: str):
#     dignote = digitalizer.dico_notes[id_note]
#     t = time_to_note(digitalizer, id_note)
#     for configuration in dignote.getConfigurations():
#         if configuration.isValid(errormax, delaymin, delaymax):
#             delay = configuration.delay
#             ax.scatter([t], [delay], s=20, color='k')

# totlength_s = sum([digitalizer.dico_notes[notes[-1][0]].analoglength for notes in digitalizer.chrono_notes])
# fig = plt.figure(figsize=(totlength_s * 1.5, 8), tight_layout=True)
# ax = fig.add_subplot(111)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Delay [s]")
# ax.set_ylim([delaymin, delaymax])
# for notes in digitalizer.chrono_notes:
#     t = time_to_note(digitalizer, notes[0][0])
#     ax.axvline(t, color="gray", linestyle = "--")
#     plotConfigurations(ax, digitalizer, notes[0][0])
#     if len(notes) == 2:
#         print(notes)
#         plotConfigurations(ax, digitalizer, notes[1][0])
#         if len(notes[0]) == 2:
#             plotConfigurations(ax, digitalizer, notes[0][1])

# fig.savefig(png_path + "_2.png")
