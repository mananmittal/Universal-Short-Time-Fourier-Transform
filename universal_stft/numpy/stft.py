"""
Universal Short-Time Fourier Transform (NumPy backend).

The transform walks through the signal one hop at a time. At every frame it:

1. takes the ``n_fft`` most recent samples;
2. windows them with every candidate Gaussian window of length
   ``H, 2H, 4H, ...`` (each one ends on the newest sample) and takes the FFT;
3. scores each candidate with its normalized Rényi-3 entropy in time plus
   frequency (lower means more concentrated);
4. turns the scores into a likelihood and multiplies it by the prior from the
   previous frame to get a posterior over window lengths;
5. outputs the posterior-weighted sum of the candidate spectra. This equals a
   normal FFT taken with the effective window ``g = sum_s posterior[s] * w_s``;
6. moves the posterior through the stay / grow / reset model to get the
   prior for the next frame.

There are two modes:

``mode="analysis"`` (default)
    Steps 1-6 only. One causal pass, nothing is computed twice. Use this for
    visualization, features, or as a front end for a neural network.

``mode="filtering"``
    Also builds the companion ("filtering") stream, in the same pass. Its
    frames are ``FFT(u * g)`` where ``u = x / E`` and ``E`` is the sum of all
    effective windows placed at their frames. Overlap-adding the inverse FFTs
    of these frames gives back ``x`` exactly. ``E`` at a sample is known once
    every window that can cover it has been seen, so each companion frame is
    produced a fixed ``max_length / H - 1`` frames after its analysis frame.
    Use this mode when you want to filter and resynthesize.

The PyTorch backend in ``universal_stft.torch`` has the same functions and
returns the same fields in the same order.
"""

from collections import deque

import numpy as np
from scipy import signal

from .._shared import (
    ENVELOPE_FLOOR,
    USTFTResult,
    candidate_lengths,
    check_mode,
    frame_end,
    frame_geometry,
    normalized_renyi3_lower_bound,
    transition_probabilities,
    validate_transform_sizes,
)

__all__ = [
    "USTFTResult",
    "universal_short_time_fourier_transform",
    "generate_analysis_spectrogram",
    "generate_filtering_spectrogram",
    "blend_guided_stft",
    "inverse_universal_spectrogram",
    "multichannel_stft",
    "calculate_physics_loss_renyi",
    "running_2d_renyi",
    "normalized_renyi3_lower_bound",
]

# Older internal names, kept because the benchmark scripts use them.
_validate_transform_sizes = validate_transform_sizes
_candidate_lengths = candidate_lengths


# ---------------------------------------------------------------------------
# Windows, padding and the transition table
# ---------------------------------------------------------------------------
def _gaussian_analysis_window(length):
    """Gaussian window of the given length (std = length / 6), summing to 1."""
    window = signal.windows.gaussian(length, std=length / 6.0)
    return window / np.sum(window)


def _window_bank(n_fft, hop_size):
    """Return ``(lengths, windows)``; row ``s`` of ``windows`` is window ``s``
    placed at the right end of an ``n_fft``-long buffer."""
    lengths = candidate_lengths(n_fft, hop_size)
    windows = np.zeros((len(lengths), n_fft))
    for state, length in enumerate(lengths):
        windows[state, -length:] = _gaussian_analysis_window(length)
    return lengths, windows


def _padded_signal(x, n_fft, hop_size):
    """Zero-pad ``x``. Returns ``(x_padded, pad_len, num_frames)``."""
    pad_len, buffer_len, num_frames = frame_geometry(len(x), n_fft, hop_size)
    x_padded = np.pad(x, (pad_len, buffer_len - len(x) - pad_len))
    return x_padded, pad_len, num_frames


def _run_length_transition_matrix(lengths, hop_size):
    """``(states, 3)`` table of ``[reset, stay, grow]`` probabilities.

    Stored in float32, the precision used for the published results.
    """
    return transition_probabilities(lengths, hop_size).astype(np.float32)


def _next_prior(posterior, transitions):
    """Push the posterior one frame forward through stay / grow / reset."""
    reset = np.sum(posterior * transitions[:, 0])   # every state can reset to state 0
    stay = posterior * transitions[:, 1]
    grow = posterior * transitions[:, 2]            # state s grows into state s + 1
    prior = stay + np.concatenate(([reset], grow[:-1]))
    return prior / np.sum(prior)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
def calculate_physics_loss_renyi(fft_buffer, fft_vec, L, alpha=3.0):
    """Normalized Rényi entropy of windowed frames, in time and frequency.

    ``fft_buffer`` holds windowed frames (last axis = time), ``fft_vec`` their
    one-sided rFFTs and ``L`` the window lengths. The time entropy is divided
    by ``log2(L)`` and the frequency entropy by ``log2(n_fft // 2 + 1)`` so
    both lie in ``[0, 1]``. Returns ``(total, time_part, frequency_part)``.
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be positive and different from 1")
    n_fft = np.shape(fft_buffer)[-1]
    if n_fft < 2 or np.shape(fft_vec)[-1] != n_fft // 2 + 1:
        raise ValueError("fft_vec must be the rFFT of a buffer with at least two samples")
    if np.any(np.asarray(L) <= 1):
        raise ValueError("L must be greater than 1 for logarithmic normalization")

    def renyi(power, normalizer):
        pmf = power / (np.sum(power, axis=-1, keepdims=True) + 1e-12)
        entropy = np.log2(np.sum(pmf ** alpha, axis=-1) + 1e-12) / (1.0 - alpha)
        return entropy / np.log2(normalizer)

    h_time = renyi(np.abs(fft_buffer) ** 2, L)
    h_spec = renyi(np.abs(fft_vec) ** 2, np.shape(fft_vec)[-1])
    return h_time + h_spec, h_time, h_spec


def running_2d_renyi(S, alpha=3.0):
    """Rényi entropy of the spectrogram prefix ``S[:m]`` for every frame ``m``.

    ``S`` has shape ``(frames, bins)``; returns one value per frame.
    """
    power = np.abs(S) ** 2
    total_power = np.cumsum(np.sum(power, axis=1)) + 1e-12
    total_power_alpha = np.cumsum(np.sum(power ** alpha, axis=1)) + 1e-12
    return (np.log2(total_power_alpha) - alpha * np.log2(total_power)) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Companion (filtering) stream
# ---------------------------------------------------------------------------
class _CompanionStream:
    """Builds the filtering spectrogram while the effective windows arrive.

    Call ``push(window)`` once per frame with that frame's effective window,
    then ``finish()``. Frame ``m`` is ``rFFT(u * g_m)`` with ``u = x / E``.

    Every window has at most ``max_length = D * H`` nonzero samples ending on
    its frame's newest sample, so a hop-sized block of ``E`` is final once ``D``
    frames have covered it. ``pending`` holds the ``D`` blocks that may still
    change; ``finished`` holds the last ``D`` blocks that are final, which is
    exactly the stretch of ``E`` the oldest waiting frame needs.
    """

    def __init__(self, x_padded, pad_len, n_fft, hop_size, max_length):
        self.x_padded = x_padded
        self.pad_len = pad_len
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.max_length = int(max_length)
        self.blocks_per_window = self.max_length // hop_size   # D

        self.pending = np.zeros(self.max_length)
        self.finished = deque(maxlen=self.blocks_per_window)
        self.waiting_windows = deque()
        self.frames_pushed = 0
        self.blocks_finished = 0
        self.spectra = []

    def push(self, window):
        self.waiting_windows.append(window)
        self.pending = self.pending + window[-self.max_length:]
        self._finish_oldest_block()
        self.frames_pushed += 1
        if self.blocks_finished >= len(self.spectra) + self.blocks_per_window:
            self._emit_oldest_frame()

    def finish(self):
        """Emit the frames still waiting at the end of the signal."""
        # No more windows will arrive, so the pending blocks are final as well.
        while len(self.spectra) < self.frames_pushed:
            self._finish_oldest_block()
            if self.blocks_finished >= len(self.spectra) + self.blocks_per_window:
                self._emit_oldest_frame()
        if not self.spectra:
            return np.zeros((0, self.n_fft // 2 + 1), dtype=complex)
        return np.stack(self.spectra)

    def _finish_oldest_block(self):
        hop = self.hop_size
        self.blocks_finished += 1
        self.finished.append(self.pending[:hop])
        self.pending = np.concatenate((self.pending[hop:], np.zeros(hop)))

    def _emit_oldest_frame(self):
        frame_index = len(self.spectra)
        window = self.waiting_windows.popleft()
        envelope = np.concatenate(self.finished)

        end = frame_end(frame_index, self.pad_len, self.hop_size)
        samples = self.x_padded[end - self.max_length:end]
        covered = envelope > ENVELOPE_FLOOR
        flattened = np.where(covered, samples / np.where(covered, envelope, 1.0), 0.0)

        frame = np.zeros(self.n_fft)
        frame[-self.max_length:] = flattened * window[-self.max_length:]
        self.spectra.append(np.fft.rfft(frame))


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------
def universal_short_time_fourier_transform(x, n_fft=256, hop_size=32, beta=1.5,
                                           mode="analysis"):
    """Adaptive STFT of a 1-D signal ``x``.

    Parameters
    ----------
    x : array of shape (samples,)
    n_fft : FFT size ``N``. The longest candidate window is at most ``N // 2``.
    hop_size : hop ``H`` between frames, also the shortest window.
    beta : inverse temperature. Larger values follow the scores more closely.
    mode : ``"analysis"`` (one pass, no companion stream) or ``"filtering"``
        (also builds the invertible companion stream).

    Returns
    -------
    USTFTResult
        A named tuple. ``filtering_spectrogram`` is ``None`` in analysis mode.
    """
    check_mode(mode)
    x = np.asarray(x)
    x_padded, pad_len, num_frames = _padded_signal(x, n_fft, hop_size)
    lengths, windows = _window_bank(n_fft, hop_size)
    num_states = len(lengths)
    transitions = _run_length_transition_matrix(lengths, hop_size)

    companion = None
    if mode == "filtering":
        companion = _CompanionStream(x_padded, pad_len, n_fft, hop_size, lengths[-1])

    num_bins = n_fft // 2 + 1
    analysis = np.zeros((num_frames, num_bins), dtype=complex)
    pmf_history = np.zeros((num_frames, num_states))
    candidate_specs = np.zeros((num_states, num_frames, num_bins), dtype=complex)
    candidate_costs = np.zeros((num_frames, num_states))
    candidate_h_time = np.zeros((num_frames, num_states))
    candidate_h_spec = np.zeros((num_frames, num_states))
    blended_costs = np.zeros(num_frames)
    blended_h_time = np.zeros(num_frames)
    blended_h_spec = np.zeros(num_frames)

    prior = np.zeros(num_states)
    prior[0] = 1.0                   # start on the shortest window

    for m in range(num_frames):
        end = frame_end(m, pad_len, hop_size)
        frame = x_padded[end - n_fft:end]

        # 1-3. Window, FFT and score every candidate that fits in the samples
        # seen so far. Unavailable candidates keep a zero spectrum and score.
        available = lengths <= (m + 1) * hop_size
        windowed = frame * windows[available]
        spectra = np.fft.rfft(windowed, axis=-1)
        costs, h_time, h_spec = calculate_physics_loss_renyi(
            windowed, spectra, lengths[available])
        candidate_specs[available, m] = spectra
        candidate_costs[m, available] = costs
        candidate_h_time[m, available] = h_time
        candidate_h_spec[m, available] = h_spec

        # 4. Scores -> likelihood -> posterior. Dividing beta by the spread of
        # the scores makes the likelihood independent of the score scale.
        spread = np.std(costs) if len(costs) > 1 else 0.0
        likelihood = np.zeros(num_states)
        likelihood[available] = np.exp(-beta / (spread + 1e-6) * (costs - np.min(costs)))
        posterior = prior * likelihood
        evidence = np.sum(posterior)
        if evidence < 1e-30:
            posterior = np.zeros(num_states)
            posterior[0] = 1.0
        else:
            posterior = posterior / evidence
        pmf_history[m] = posterior

        # 5. Output: the posterior mixture of candidate spectra, which is the
        # FFT of the frame under the effective window.
        effective_window = posterior @ windows
        analysis[m] = posterior @ candidate_specs[:, m]
        blended_costs[m], blended_h_time[m], blended_h_spec[m] = calculate_physics_loss_renyi(
            frame * effective_window, analysis[m], posterior @ lengths)
        if companion is not None:
            companion.push(effective_window)

        # 6. Prior for the next frame.
        prior = _next_prior(posterior, transitions)

    return USTFTResult(
        analysis_spectrogram=analysis,
        filtering_spectrogram=companion.finish() if companion is not None else None,
        dominant_lengths=lengths[np.argmax(pmf_history, axis=1)].astype(float),
        pmf_history=pmf_history,
        blend_info=[(lengths, weights) for weights in pmf_history],
        candidate_entropies=candidate_costs,
        blended_entropies=blended_costs,
        blended_h_time=blended_h_time,
        blended_h_spec=blended_h_spec,
        candidate_h_time=candidate_h_time,
        candidate_h_spec=candidate_h_spec,
        all_candidate_spectrograms=candidate_specs,
    )


# ---------------------------------------------------------------------------
# Reusing a known trajectory
# ---------------------------------------------------------------------------
def _effective_windows(blend_info, num_frames, n_fft, hop_size):
    """Yield the effective window of every frame of a known trajectory."""
    if len(blend_info) < num_frames:
        raise ValueError("blend_info must contain one entry per signal frame")
    grid, windows = _window_bank(n_fft, hop_size)
    state_of = {int(length): state for state, length in enumerate(grid)}
    for m in range(num_frames):
        frame_lengths, weights = blend_info[m]
        try:
            states = [state_of[int(length)] for length in frame_lengths]
        except KeyError:
            raise ValueError("blend_info contains a length outside the candidate grid") from None
        yield np.asarray(weights, dtype=float) @ windows[states]


def generate_analysis_spectrogram(x, blend_info, n_fft=256, hop_size=32):
    """Analysis spectrogram of ``x`` along a known trajectory ``blend_info``."""
    x_padded, pad_len, num_frames = _padded_signal(np.asarray(x), n_fft, hop_size)
    analysis = np.zeros((num_frames, n_fft // 2 + 1), dtype=complex)
    for m, window in enumerate(_effective_windows(blend_info, num_frames, n_fft, hop_size)):
        end = frame_end(m, pad_len, hop_size)
        analysis[m] = np.fft.rfft(x_padded[end - n_fft:end] * window)
    return analysis


def generate_filtering_spectrogram(x, blend_info, n_fft=256, hop_size=32):
    """Companion (filtering) spectrogram of ``x`` along a known trajectory.

    Inverting it with :func:`inverse_universal_spectrogram` gives back ``x``.
    """
    x_padded, pad_len, num_frames = _padded_signal(np.asarray(x), n_fft, hop_size)
    longest = candidate_lengths(n_fft, hop_size)[-1]
    companion = _CompanionStream(x_padded, pad_len, n_fft, hop_size, longest)
    for window in _effective_windows(blend_info, num_frames, n_fft, hop_size):
        companion.push(window)
    return companion.finish()


def blend_guided_stft(x, blend_info, n_fft=256, hop_size=32):
    """Return ``(analysis_spectrogram, filtering_spectrogram)`` for a known trajectory."""
    return (generate_analysis_spectrogram(x, blend_info, n_fft, hop_size),
            generate_filtering_spectrogram(x, blend_info, n_fft, hop_size))


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------
def inverse_universal_spectrogram(spectrogram, num_original_samples, hop_size=32,
                                  n_fft=256, max_L=None):
    """Overlap-add inverse of the **filtering** spectrogram.

    Use it on ``filtering_spectrogram`` (not the analysis stream). Unmodified
    coefficients give back the input exactly. Multiplying every frame by the
    same frequency response before inverting applies that filter.

    ``max_L`` is the longest candidate window; by default it is derived from
    ``n_fft`` and ``hop_size``.
    """
    if spectrogram is None:
        raise ValueError(
            "no filtering spectrogram: run the transform with mode='filtering'")
    if max_L is None:
        max_L = int(candidate_lengths(n_fft, hop_size)[-1])
    num_frames = spectrogram.shape[0]
    pad_len, buffer_len, _ = frame_geometry(num_original_samples, n_fft, hop_size)
    buffer_len = max(buffer_len, num_frames * hop_size + pad_len + n_fft)

    output = np.zeros(buffer_len)
    for m in range(num_frames):
        # Rotate so the window's last sample lands at index max_L, then place
        # the frame so that index lines up with the frame's end.
        frame = np.roll(np.fft.irfft(spectrogram[m], n=n_fft), max_L)
        start = frame_end(m, pad_len, hop_size) - max_L
        output[start:start + n_fft] += frame
    return output[pad_len:pad_len + num_original_samples]


# ---------------------------------------------------------------------------
# Several channels
# ---------------------------------------------------------------------------
def multichannel_stft(x_multi, reference_sensor_id, n_fft=256, beta=2.0, hop_size=32,
                      mode="analysis"):
    """Adaptive STFT of several channels along one shared trajectory.

    The trajectory is tracked on the reference channel and then applied to
    every channel, so all channels share the same effective windows (useful
    for beamforming and direction finding). ``x_multi`` has shape
    ``(channels, samples)``. Returns a ``USTFTResult`` whose two spectrograms
    have a leading channel axis; the other fields describe the reference.
    """
    check_mode(mode)
    x_multi = np.asarray(x_multi)
    reference = universal_short_time_fourier_transform(
        x_multi[reference_sensor_id], n_fft=n_fft, hop_size=hop_size, beta=beta, mode=mode)

    analysis, filtering = [], []
    for channel, x in enumerate(x_multi):
        if channel == reference_sensor_id:
            analysis.append(reference.analysis_spectrogram)
            filtering.append(reference.filtering_spectrogram)
        else:
            analysis.append(generate_analysis_spectrogram(x, reference.blend_info, n_fft, hop_size))
            if mode == "filtering":
                filtering.append(generate_filtering_spectrogram(
                    x, reference.blend_info, n_fft, hop_size))

    return reference._replace(
        analysis_spectrogram=np.stack(analysis),
        filtering_spectrogram=np.stack(filtering) if mode == "filtering" else None,
    )
