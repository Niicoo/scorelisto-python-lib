A command line python script to try to recover the score from a single audio instrument/voice.

# Configuration
Create conda environment:  
`conda env create -f environment.yaml`

# Usage
Activate conda environement:  
`conda activate scorelisto`  
Use the script `LyricsChords.py` to generated a musicxml file:  
`python ScoreListo.py ./examples/whistling_mono_8khz.wav --logging_level INFO --sample_rate 44100`  
_(You can type: `python ScoreListo.py --help` to get more help on the commands)_  

# Sources
- Knowing if a command line tool is available: https://stackoverflow.com/a/34177358
- McLeod Pitch recovering method
- I'm sorry in advance if I forget to quote some sources as I forgot where I took some part of the codes
