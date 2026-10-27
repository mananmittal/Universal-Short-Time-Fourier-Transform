"""Run the full evaluation on recorded audio.

With no arguments it analyses the bundled synthetic hyperbolic chirp::

    python -m benchmarks.run_real_audio

Pass one or more WAV files to analyse your own recordings instead::

    python -m benchmarks.run_real_audio path/to/recording.wav

Results go to ``results/real/``.
"""

import os
import sys
import traceback

import numpy as np
import scipy.signal
from scipy.io import wavfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.evaluate import run_full_evaluation

WAV_DIR = os.path.join(os.path.dirname(__file__), 'wav')
DEFAULT_WAV = os.path.join(WAV_DIR, 'hyperbolic_chirp.wav')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'real')
MAX_SECONDS = 15


def read_wav(path):
    """Return ``(samples, sample_rate)`` as floats in [-1, 1], mixed to mono."""
    fs, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(np.iinfo(x.dtype).max)
    x = x.astype(float)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    return x, fs


def process_file(wav_file, out_dir):
    n_fft = 4096
    hop_size = 128
    name = os.path.splitext(os.path.basename(wav_file))[0]
    try:
        x, fs = read_wav(wav_file)

        # Recording-specific settings for the files used in the paper.
        if name in ('english_speech_1', 'english_speech_2'):
            x = x[int(1.2 * fs):int(6.2 * fs)]        # a 5 s stretch of speech
        elif name == 'humpback_whale_song':
            x = scipy.signal.resample_poly(x, up=1, down=6)
            fs = fs // 6
        x = x[:int(MAX_SECONDS * fs)]

        if name == 'hyperbolic_chirp':
            beta = 1.5
        elif name == 'humpback_whale_song':
            beta = 0.3
        else:
            beta = 0.4

        run_full_evaluation(x, fs, name, out_dir, n_fft=n_fft, hop_size=hop_size, beta=beta)
    except Exception as error:
        print(f"Error in {wav_file}: {error}")
        traceback.print_exc()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for wav_file in sys.argv[1:] or [DEFAULT_WAV]:
        process_file(wav_file, OUT_DIR)
