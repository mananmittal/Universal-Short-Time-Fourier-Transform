"""PyTorch example: analyse a batch of two chirps on CPU or GPU, then invert.

Run from the repository root:  python examples/example_torch.py
"""

import os
import sys

import numpy as np
import torch
from scipy import signal

# Allow running straight from a clone (no install required).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from universal_stft.torch import stft
from benchmarks.plot_suites import example_figures, example_figure_build


def hold_unavailable_scores(scores, lengths, hop_size):
    """For plotting, give windows longer than the samples seen so far the
    score of the longest available window instead of zero."""
    scores = scores.copy()
    for m in range(scores.shape[0]):
        longest_available = np.flatnonzero(lengths <= (m + 1) * hop_size)[-1]
        scores[m, longest_available + 1:] = scores[m, longest_available]
    return scores


def plot_result(fs, hop_size, duration, title, result, row):
    """Plot one row of a batched result."""
    def numpy(tensor):
        return tensor[row].cpu().numpy()

    lengths = result.blend_info[0][0]
    time_axis = np.linspace(0, duration, result.blended_entropies.shape[1])
    candidate_specs = numpy(result.all_candidate_spectrograms)

    # Prefix 2-D Rényi Entropy: exactly one calculation per raw prefix.
    candidate_renyi = [stft.running_2d_renyi(S).cpu().numpy()
                       for S in result.all_candidate_spectrograms[row]]
    ustft_renyi = stft.running_2d_renyi(result.analysis_spectrogram[row]).cpu().numpy()

    example_figures(
        title, fs, hop_size, duration,
        numpy(result.analysis_spectrogram), numpy(result.pmf_history), lengths,
        list(candidate_specs),
        hold_unavailable_scores(numpy(result.candidate_entropies), lengths, hop_size),
        numpy(result.blended_entropies),
        hold_unavailable_scores(numpy(result.candidate_h_time), lengths, hop_size),
        numpy(result.blended_h_time),
        hold_unavailable_scores(numpy(result.candidate_h_spec), lengths, hop_size),
        numpy(result.blended_h_spec),
        candidate_renyi, ustft_renyi, time_axis, "torch")


@example_figure_build("torch")
def main():
    fs = 4000
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x_hyp = signal.chirp(t, f0=2000, t1=duration, f1=50, method="hyperbolic")
    x_quad = signal.chirp(t, f0=50, t1=duration, f1=1000, method="quadratic")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_batch = torch.tensor(np.stack([x_hyp, x_quad]), dtype=torch.float32, device=device)
    print(f"Input batch: {tuple(x_batch.shape)} on {device}")

    n_fft = 512
    hop_size = 64
    beta = 1

    # Analysis mode: one causal pass. This is the mode to use inside a network.
    result = stft.universal_short_time_fourier_transform(
        x_batch, n_fft=n_fft, hop_size=hop_size, beta=beta, mode="analysis")
    print(f"Analysis spectrogram: {tuple(result.analysis_spectrogram.shape)} (batch, frames, bins)")
    print(f"Posterior over windows: {tuple(result.pmf_history.shape)} (batch, frames, windows)")
    for row, title in enumerate(["Batch 0: Hyperbolic Chirp", "Batch 1: Quadratic Chirp"]):
        plot_result(fs, hop_size, duration, title, result, row)

    # Filtering mode: same pass plus the invertible companion stream.
    result = stft.universal_short_time_fourier_transform(
        x_batch, n_fft=n_fft, hop_size=hop_size, beta=beta, mode="filtering")
    x_rec = stft.inverse_universal_spectrogram(
        result.filtering_spectrogram, x_batch.shape[1], hop_size=hop_size, n_fft=n_fft)
    mse = torch.mean((x_batch - x_rec) ** 2, dim=1)
    print(f"Reconstruction MSE: batch 0 {mse[0]:.2e}, batch 1 {mse[1]:.2e}")


if __name__ == "__main__":
    main()
