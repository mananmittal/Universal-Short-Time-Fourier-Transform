"""NumPy example: analyse two chirps, plot the diagnostics, then invert.

Run from the repository root:  python examples/example.py
"""

import os
import sys

import numpy as np
from scipy import signal

# Allow running straight from a clone (no install required).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from universal_stft.numpy import stft
from benchmarks.plot_suites import example_figures, example_figure_build


def hold_unavailable_scores(scores, lengths, hop_size):
    """For plotting, give windows longer than the samples seen so far the
    score of the longest available window instead of zero."""
    scores = scores.copy()
    for m in range(scores.shape[0]):
        longest_available = np.flatnonzero(lengths <= (m + 1) * hop_size)[-1]
        scores[m, longest_available + 1:] = scores[m, longest_available]
    return scores


def plot_result(x, fs, hop_size, title, result):
    duration = len(x) / fs
    lengths = result.blend_info[0][0]
    time_axis = np.linspace(0, duration, len(result.blended_entropies))

    # Prefix 2-D Rényi Entropy: exactly one calculation per raw prefix.
    candidate_renyi = [stft.running_2d_renyi(S) for S in result.all_candidate_spectrograms]
    ustft_renyi = stft.running_2d_renyi(result.analysis_spectrogram)

    example_figures(
        title, fs, hop_size, duration,
        result.analysis_spectrogram, result.pmf_history, lengths,
        result.all_candidate_spectrograms,
        hold_unavailable_scores(result.candidate_entropies, lengths, hop_size),
        result.blended_entropies,
        hold_unavailable_scores(result.candidate_h_time, lengths, hop_size),
        result.blended_h_time,
        hold_unavailable_scores(result.candidate_h_spec, lengths, hop_size),
        result.blended_h_spec,
        candidate_renyi, ustft_renyi, time_axis, "numpy")


@example_figure_build("numpy")
def main():
    fs = 4000
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    chirps = {
        "Hyperbolic Chirp": signal.chirp(t, f0=2000, t1=duration, f1=50, method="hyperbolic"),
        "Quadratic Chirp": signal.chirp(t, f0=50, t1=duration, f1=1000, method="quadratic"),
    }

    n_fft = 512
    hop_size = 64
    beta = 1

    for title, x in chirps.items():
        # Analysis mode: one causal pass, no companion stream.
        print(f"-> {title}: analysis mode")
        result = stft.universal_short_time_fourier_transform(
            x, n_fft=n_fft, hop_size=hop_size, beta=beta, mode="analysis")
        plot_result(x, fs, hop_size, title, result)

        # Filtering mode: same pass plus the invertible companion stream.
        print(f"-> {title}: filtering mode and reconstruction")
        result = stft.universal_short_time_fourier_transform(
            x, n_fft=n_fft, hop_size=hop_size, beta=beta, mode="filtering")
        x_rec = stft.inverse_universal_spectrogram(result.filtering_spectrogram, len(x), hop_size, n_fft)
        print(f"   reconstruction MSE: {np.mean((x - x_rec) ** 2):.2e}")


if __name__ == "__main__":
    main()
