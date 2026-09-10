# https://github.com/drfeinberg/Parselmouth-Guides/tree/master
# https://parselmouth.readthedocs.io/en/stable/examples/pitch_manipulation.html
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call
from tqdm.auto import tqdm

# ---- Corpus location ------------------------------------------------------
# Point THCHS30 at your local copy. Default is repo-relative; set the
# environment variable to your own layout (see README "Data requirements").
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CORPORA_ROOT = os.environ.get('CORPORA_ROOT', os.path.join(_REPO_ROOT, 'corpora'))
THCHS30_DIR = os.environ.get('THCHS30_DIR', os.path.join(CORPORA_ROOT, 'data_thchs30'))

# import librosa
# DID NOT USE BUT MAYBE USEFUL
# https://nbviewer.org/github/timmahrt/ProMo/blob/main/tutorials/tutorial1_2_pitch_manipulations.ipynb#basic_manipulations_problems

def flatten_pitch(sound):
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    pitch = sound.to_pitch()

    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    # print(mean_pitch)
    # median_pitch = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    # print(median_pitch)
    # min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    # print(min_pitch)
    # max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    # print(max_pitch)
    # standard_deviation_pitch = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    # print(standard_deviation_pitch)
    call(pitch_tier, "Remove points between...", sound.time_range[0], sound.time_range[1])
    call(pitch_tier, 'Add point...', np.average(sound.time_range), 116)
    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_flat = call(manipulation, "Get resynthesis (overlap-add)")
    return sound_flat

def batch_flattening(path):
    wav_files = glob.glob(os.path.join(os.path.expanduser(dataset_path), '*.wav'), recursive = True)
    unprocessed_wav_files = [x for x in wav_files if '_flat' not in x]
    for wav_file in tqdm(unprocessed_wav_files):
        tqdm.write(f"Processing {wav_file}...")
        # tqdm.write('DRY RUN')
        s = parselmouth.Sound(wav_file)
        s_flat = flatten_pitch(s)
        s_flat.save(os.path.splitext(wav_file)[0] + "_flat.wav", 'WAV')
    


def main(path):
    batch_flattening(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Use a pre-trained wav2vec2 model to generate embeddings")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Directory of .wav files to flatten. Defaults to the THCHS-30 data dir.",
    )
    args = parser.parse_args()


    return args


if __name__ == "__main__":
    args = parse_args()
    dataset_path = os.path.expanduser(args.path) if args.path else os.path.join(THCHS30_DIR, 'data')
    main(dataset_path)
# TODO get subset of audio data and generate an aligned dataset of this thing?
