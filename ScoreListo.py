#!/usr/bin/env python
import argparse
import pathlib
import sys
import os
import logging
from shutil import which
LIB_PATH = pathlib.Path(pathlib.Path(__file__).absolute().parent, "lib")
sys.path.append(LIB_PATH.as_posix())
from scorelisto.audiofile_readers import LibrosaReader
from scorelisto.pitch_detectors import McLeodDetector
from scorelisto.note_detectors import HistogramNoteDetector
from scorelisto.rhythm_detectors import DigitalPartition
from scorelisto.tools import compute_energy, freq2tone
import numpy as np
# Typing
from typing import Tuple

# Get Logger
logger = logging.getLogger(__name__)


def is_tool(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ScoreListo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("path", type=str,
        help="Path of the audio(s) file(s) OR path of the folder containing multiple audio file to process"
    )
    parser.add_argument("--output", type=str, required=False,
        help="Path of the output score file, [default name: same name as input file, default path: path of the input file]"
    )
    parser.add_argument('--format', type=str,
        default="MUSICXML",
        help="Default Output format: MUSICXML or MIDI (you can also specify the format just by setting \
            the output file extension to '.xml', '.musicxml', '.mid' or '.midi'"
    )
    parser.add_argument("--rhythm", action=argparse.BooleanOptionalAction, default=True,
        help="Option to produce a midi output file without rhythm interpolation"
    )
    parser.add_argument('--mscore_executable', type=str,
        default=None,
        help="Path to the musescore executable (used to convert the score to PDF) \
            If not provided, it will search for the executable in PATH"
    )
    # Arguments for the Pitch Detector
    parser.add_argument("--windows_size", type=float, required=False,
        default=20.,
        help="Sliding Windows size in ms of the trunks [unit: Milliseconds]"
    )
    parser.add_argument("--step_size", type=float, required=False,
        default=1.,
        help="Step size in ms of sliding windows [unit: Milliseconds]"
    )
    parser.add_argument("--sample_rate", type=int, required=False,
        help=   "Sample rate in hertz to convert the audio before applying the pitch detection, \
                if not set, the original sample rate of the audio file is used. It is recommended \
                to set it to 44100 for better results [unit: Hertz]"
    )
    # Arguments for the rhythm detector
    parser.add_argument("--bpm_max", type=float, required=False,
        default=200.,
        help="Maximum BPM to consider when trying to recover the BPM"
    )
    parser.add_argument("--bpm_min", type=float, required=False,
        default=40.,
        help="Minimum BPM to consider when trying to recover the BPM"
    )
    parser.add_argument("--max_bpm_var", type=float, required=False,
        default=0.5,
        help="Maximum BPM variation: when trying to recover the BPM \
            (BPM note next) < max_bpm_var * (BPM note previous)"
    )
    parser.add_argument("--error_max", type=float, required=False,
        default=10.,
        help="Don't touch that parameter unless you know what you're doing"
    )
    parser.add_argument("--combs_to_mask", type=str, required=False, nargs='*',
        default=[],
        help="List of combinations not to consider for recovering rhythms \
            The names of the combinations are the one in the keys of the \
            COMBINATIONS variable in the file lib/scorelisto/combinations.py"
    )
    # Logging level
    parser.add_argument('--logging_level', type=str, nargs="?", const="INFO", default="INFO",
        help="Logging level: CRITICAL, ERROR, WARNING, INFO [Default] or DEBUG"
    )
    args = parser.parse_args()

    # General parameters
    input_path = pathlib.Path(args.path)
    output_path = None if (args.output is None) else pathlib.Path(args.output)
    output_format = args.format.upper()
    with_rhythm = args.rhythm
    mscore_executable = args.mscore_executable

    # Pitch Detector parameters
    windows_size_s = args.windows_size * 1e-3
    step_size_s = args.step_size * 1e-3
    sample_rate_hz = args.sample_rate

    # Rhythm Detector parameters
    delaymin = 60 / args.bpm_max # Conversion from BPM to delay
    delaymax = 60 / args.bpm_min # Conversion from BPM to delay
    maxdelayvar = args.max_bpm_var
    errormax = args.error_max
    combs_to_mask = args.combs_to_mask

    # Logging level
    logging_level = args.logging_level

    
    # Set Logging Level
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    for logname in ["__main__", "scorelisto"]:
        log = logging.getLogger(logname)
        log.setLevel(logging_level)
        log.addHandler(handler)


    # Detect the files to process depending on the input path is a specific file or a folder
    filepaths: Tuple[str, str] = []
    output_ext = {"MUSICXML": ".musicxml", "MIDI": ".mid"}[output_format]
    if input_path.is_file():
        if output_path is None:
            # Get the folder and name of the input file and generate the output path
            out_temp = pathlib.Path(input_path.parent, input_path.stem + output_ext)
            filepaths.append((str(input_path), str(out_temp)))
        elif output_path.suffix.lower() in [".mid", ".midi", ".xml", ".musicxml"]:
            filepaths.append((str(input_path), str(output_path)))
        else:
            # Assuming the path is a folder
            out_temp = pathlib.Path(output_path, input_path.stem + output_ext)
            filepaths.append((str(input_path), str(out_temp)))
    elif input_path.is_dir():
        # Only consider the files in the current folder (not sub directories)
        for filename in os.listdir(str(input_path)):
            p = pathlib.Path(input_path, filename)
            if p.is_file():
                ext = p.suffix.lower()
                if ext in [".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wave"]:
                    if output_path is None:
                        # Get the folder and name of the input file and generate the output path
                        out_temp = pathlib.Path(p.parent, p.stem + output_ext)
                        filepaths.append((str(p), str(out_temp)))
                    elif output_path.suffix.lower() in [".mid", ".midi", ".xml", ".musicxml"]:
                        raise ValueError("If you provide a folder as an input, you must set the path of folder for the output")
                    else:
                        # Assuming the path is a folder
                        out_temp = pathlib.Path(output_path, p.stem + output_ext)
                        filepaths.append((str(p), str(out_temp)))
     
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
    for in_file, out_file in filepaths:
        logger.info(f"Processing input file: {in_file}")
        logger.info(f"Output file path: {out_file}")
        output_ext = pathlib.Path(out_file).suffix.lower()
        if output_ext in [".xml", ".musicxml"]:
            output_fmt = "musicxml"
        elif output_ext in [".mid", ".midi"]:
            output_fmt = "midi"
        else:
            raise ValueError("Unrecognized output format")
        # Reading and extracting pitch of the audio file
        audio_reader.read(in_file, sample_rate_hz)
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
        notes, _ = note_detector.perform(pitches, step_size_s, energies)
        # Detecting Rhythm
        digitalizer = DigitalPartition(delaymin, delaymax, maxdelayvar, errormax, combs_to_mask)
        if((output_fmt == "midi") and (not with_rhythm)):
            file_bytes = digitalizer.performWithoutRhythm(notes)
            with open(out_file, "wb") as f:
                f.write(file_bytes)
            continue
        file_bytes = digitalizer.perform(notes, output_fmt)
        with open(out_file, "wb") as f:
            f.write(file_bytes)
        if((output_fmt == "musicxml") and (mscore_executable is not None)):
            out_file_p = pathlib.Path(out_file)
            pdf_path = pathlib.Path(out_file_p.parent, out_file_p.stem + ".pdf").as_posix()
            logger.info(f"Generating PDF file in : {pdf_path}")
            os.system(f"{mscore_executable} {out_file} -o {pdf_path}")
