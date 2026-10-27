"""
Pieces shared by the NumPy and PyTorch backends.

Everything here is plain Python / NumPy and does not depend on the signal, so
both backends use exactly the same window grid, transition probabilities,
frame geometry and output container.
"""

import math
from typing import Any, NamedTuple

import numpy as np


MODES = ("analysis", "filtering")

# Samples whose overlap envelope is below this value are treated as uncovered
# when the companion stream divides the signal by the envelope.
ENVELOPE_FLOOR = 1e-9


class USTFTResult(NamedTuple):
    """Everything returned by ``universal_short_time_fourier_transform``.

    It is a normal tuple, so it can be unpacked or indexed by position, but
    each field also has a name. Shapes are for a single 1-D signal; the
    PyTorch backend adds a leading batch dimension for batched input.
    ``frames = ceil(len(x) / hop_size)``, ``bins = n_fft // 2 + 1`` and
    ``states`` is the number of candidate window lengths.
    """

    analysis_spectrogram: Any
    """(frames, bins) complex. The adaptive spectrogram (posterior mixture)."""

    filtering_spectrogram: Any
    """(frames, bins) complex companion stream for exact inversion, or
    ``None`` when the transform ran in ``mode="analysis"``."""

    dominant_lengths: Any
    """(frames,) the most probable window length at every frame."""

    pmf_history: Any
    """(frames, states) the posterior over window lengths at every frame."""

    blend_info: Any
    """List with one ``(candidate_lengths, weights)`` pair per frame. Feed it to
    ``generate_analysis_spectrogram`` / ``generate_filtering_spectrogram`` to
    reuse this trajectory on another signal."""

    candidate_entropies: Any
    """(frames, states) the score of every candidate (zero where unavailable)."""

    blended_entropies: Any
    """(frames,) the score of the blended (effective-window) frame."""

    blended_h_time: Any
    """(frames,) the time part of ``blended_entropies``."""

    blended_h_spec: Any
    """(frames,) the frequency part of ``blended_entropies``."""

    candidate_h_time: Any
    """(frames, states) the time part of ``candidate_entropies``."""

    candidate_h_spec: Any
    """(frames, states) the frequency part of ``candidate_entropies``."""

    all_candidate_spectrograms: Any
    """(states, frames, bins) the spectrum of every candidate window."""


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def validate_transform_sizes(n_fft, hop_size):
    """Check the FFT size and hop size used by every transform."""
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("n_fft must be an integer greater than or equal to 2")
    if not isinstance(hop_size, (int, np.integer)) or hop_size < 2:
        raise ValueError("hop_size must be an integer greater than or equal to 2")
    if hop_size > n_fft:
        raise ValueError("hop_size must not exceed n_fft")


def candidate_lengths(n_fft, hop_size):
    """Return the candidate window lengths ``H, 2H, 4H, ...``.

    The longest window is the largest power-of-two multiple of ``H`` that does
    not exceed ``max(n_fft // 2, H)``. For example ``n_fft=512, H=32`` gives
    ``[32, 64, 128, 256]``.
    """
    validate_transform_sizes(n_fft, hop_size)
    longest_allowed = max(n_fft // 2, hop_size)
    lengths = []
    length = int(hop_size)
    while length <= longest_allowed:
        lengths.append(length)
        length *= 2
    return np.asarray(lengths, dtype=np.int64)


def frame_geometry(num_samples, n_fft, hop_size):
    """Return ``(pad_len, buffer_len, num_frames)`` for a signal.

    The signal is zero-padded by ``pad_len = n_fft`` samples on the left so the
    first frame has a full look-back, and on the right so every frame fits.
    Frame ``m`` (0-based) ends just before padded sample
    ``pad_len + (m + 1) * hop_size``.
    """
    validate_transform_sizes(n_fft, hop_size)
    num_frames = int(math.ceil(num_samples / hop_size))
    pad_len = n_fft
    buffer_len = max(num_samples + 2 * pad_len,
                     num_frames * hop_size + pad_len + n_fft)
    return pad_len, buffer_len, num_frames


def frame_end(frame_index, pad_len, hop_size):
    """Index (in the padded signal) one past the newest sample of a frame."""
    return pad_len + (frame_index + 1) * hop_size


# ---------------------------------------------------------------------------
# Stay / grow / reset transition probabilities
# ---------------------------------------------------------------------------
def _macro_growth_probability(run_length, steps):
    """Probability of growing ``steps`` hops in a row from ``run_length`` hops.

    One unit step grows with probability ``1 - 2 / (3u)`` at run length ``u``.
    The product over ``u = r, ..., r + steps - 1`` is computed with log-Gamma
    functions so wide steps neither underflow nor need a loop.
    """
    r = run_length
    log_probability = (
        math.lgamma(r + steps - 2.0 / 3.0)
        - math.lgamma(r - 2.0 / 3.0)
        + math.lgamma(r)
        - math.lgamma(r + steps)
    )
    return min(1.0, max(0.0, math.exp(log_probability)))


def transition_probabilities(lengths, hop_size):
    """Return a ``(states, 3)`` table with columns ``[reset, stay, grow]``.

    The prior on window changes is defined one hop at a time: at a run of
    ``u`` hops the window resets or stays with probability ``1/(3u)`` each and
    grows by one hop with probability ``1 - 2/(3u)``. The candidate grid only
    keeps the lengths ``H, 2H, 4H, ...``, so moving from ``L`` to the next
    length ``L_next`` needs ``(L_next - L) / H`` growth steps in a row. That
    product is the ``grow`` probability. The rest of the mass is split equally
    between ``reset`` and ``stay`` because their one-step probabilities are
    equal. The longest window cannot grow: it resets with probability
    ``1 / (2r)`` and otherwise stays.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) == 0:
        raise ValueError("candidate lengths must be a non-empty one-dimensional array")
    if np.any(np.diff(lengths) <= 0) or np.any(lengths % hop_size != 0):
        raise ValueError("candidate lengths must be increasing integer multiples of hop_size")

    table = np.zeros((len(lengths), 3))
    for state in range(len(lengths) - 1):
        run_length = int(lengths[state] // hop_size)
        steps = int((lengths[state + 1] - lengths[state]) // hop_size)
        p_grow = _macro_growth_probability(run_length, steps)
        table[state] = [(1.0 - p_grow) / 2, (1.0 - p_grow) / 2, p_grow]

    longest_run = int(lengths[-1] // hop_size)
    p_reset = 1.0 / (2.0 * longest_run)
    table[-1] = [p_reset, 1.0 - p_reset, 0.0]
    return table


# ---------------------------------------------------------------------------
# Lower bound on the score
# ---------------------------------------------------------------------------
def normalized_renyi3_lower_bound(n_fft):
    """Return a positive lower bound on the normalized Rényi-3 score.

    No nonzero frame can be concentrated in a single sample and a single
    frequency bin at once, so the score ``H3(time)/log2(L) + H3(freq)/log2(F)``
    computed on the one-sided rFFT can never reach zero. With ``P`` and ``Q``
    the largest time and frequency probabilities, a Fourier projector argument
    gives ``P * Q <= B(N)``, and ``H3 >= -log(max probability)`` turns that
    into the floor ``-log(B(N)) / log(N)``. See the README for ``B(N)``.
    """
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("n_fft must be an integer greater than or equal to 2")
    c = n_fft ** -0.5
    t = max(c, np.sqrt(2.0 + c) - 1.0)
    b_endpoint = 2.0 * t * min(1.0, 1.0 + c - t) / (1.0 + t)
    if n_fft == 2:
        b_max = b_endpoint
    else:
        b_interior = ((1.0 + np.sqrt(2.0 / n_fft)) / 2.0) ** 2
        b_max = max(b_endpoint, b_interior)
    return float(-np.log(b_max) / np.log(n_fft))
