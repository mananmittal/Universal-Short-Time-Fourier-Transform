"""
Universal Short-Time Fourier Transform (PyTorch backend).

Same algorithm, functions, modes and output fields as the NumPy backend in
``universal_stft.numpy.stft`` (read that module's docstring for the
step-by-step description). In addition this backend:

* accepts a single signal ``(samples,)`` or a batch ``(batch, samples)``;
* runs on any device (CPU or GPU) and keeps the input precision
  (``float32`` -> ``complex64``, ``float64`` -> ``complex128``);
* is differentiable, so it can sit inside a neural network.

``mode="analysis"`` (default) is a single causal pass and is the one to use
for training. ``mode="filtering"`` also builds the invertible companion stream
in the same pass.
"""

from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

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

_validate_transform_sizes = validate_transform_sizes
_candidate_lengths = candidate_lengths


# ---------------------------------------------------------------------------
# Windows, padding and the transition table
# ---------------------------------------------------------------------------
def _complex_dtype(real_dtype):
    """complex64 for float32 input, complex128 for float64 input."""
    return torch.promote_types(real_dtype, torch.complex64)


def _gaussian_analysis_window(length, dtype, device):
    """Gaussian window of the given length (std = length / 6), summing to 1."""
    n = torch.arange(length, dtype=dtype, device=device)
    window = torch.exp(-0.5 * ((n - (length - 1) / 2) / (length / 6.0)) ** 2)
    return window / torch.sum(window)


def _window_bank(n_fft, hop_size, dtype, device):
    """Return ``(lengths, windows)``; row ``s`` of ``windows`` is window ``s``
    placed at the right end of an ``n_fft``-long buffer."""
    lengths = candidate_lengths(n_fft, hop_size)
    windows = torch.zeros((len(lengths), n_fft), dtype=dtype, device=device)
    for state, length in enumerate(lengths):
        windows[state, -length:] = _gaussian_analysis_window(int(length), dtype, device)
    return lengths, windows


def _padded_signal(x, n_fft, hop_size):
    """Zero-pad a batch ``x`` of shape (batch, samples).

    Returns ``(x_padded, pad_len, num_frames)``.
    """
    num_samples = x.shape[-1]
    pad_len, buffer_len, num_frames = frame_geometry(num_samples, n_fft, hop_size)
    x_padded = F.pad(x, (pad_len, buffer_len - num_samples - pad_len))
    return x_padded, pad_len, num_frames


def _run_length_transition_matrix(lengths, hop_size, dtype, device):
    """``(states, 3)`` table of ``[reset, stay, grow]`` probabilities."""
    table = transition_probabilities(lengths, hop_size)
    return torch.as_tensor(table, dtype=dtype, device=device)


def _next_prior(posterior, transitions):
    """Push a batch of posteriors one frame forward through stay / grow / reset."""
    reset = posterior @ transitions[:, 0]           # every state can reset to state 0
    stay = posterior * transitions[:, 1]
    grow = posterior * transitions[:, 2]            # state s grows into state s + 1
    prior = stay + torch.cat((reset.unsqueeze(1), grow[:, :-1]), dim=1)
    return prior / torch.sum(prior, dim=1, keepdim=True)


def _causal_frames(x_padded, num_frames, n_fft, hop_size):
    """All frames as a (batch, frames, n_fft) view.

    Frame ``m`` ends at ``pad_len + (m + 1) * hop_size`` with ``pad_len = n_fft``,
    so it starts at ``(m + 1) * hop_size``: that is unfold window ``m + 1``.
    """
    return x_padded.unfold(1, n_fft, hop_size)[:, 1:num_frames + 1, :]


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
def calculate_physics_loss_renyi(fft_buffer, fft_vec, L, alpha=3.0):
    """Normalized Rényi entropy of windowed frames, in time and frequency.

    ``fft_buffer`` holds windowed frames (last axis = time), ``fft_vec`` their
    one-sided rFFTs and ``L`` the window lengths (a number or a tensor that
    broadcasts against the leading axes). The time entropy is divided by
    ``log2(L)`` and the frequency entropy by ``log2(n_fft // 2 + 1)``.
    Returns ``(total, time_part, frequency_part)``.
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be positive and different from 1")
    n_fft = fft_buffer.shape[-1]
    if n_fft < 2 or fft_vec.shape[-1] != n_fft // 2 + 1:
        raise ValueError("fft_vec must be the rFFT of a buffer with at least two samples")
    if not torch.is_tensor(L) and np.any(np.asarray(L) <= 1):
        raise ValueError("L must be greater than 1 for logarithmic normalization")
    real_dtype = fft_buffer.real.dtype if fft_buffer.is_complex() else fft_buffer.dtype
    device = fft_buffer.device

    def renyi(power, normalizer):
        pmf = power / (torch.sum(power, dim=-1, keepdim=True) + 1e-12)
        entropy = torch.log2(torch.sum(pmf ** alpha, dim=-1) + 1e-12) / (1.0 - alpha)
        return entropy / torch.log2(torch.as_tensor(normalizer, dtype=real_dtype, device=device))

    h_time = renyi(torch.abs(fft_buffer) ** 2, L)
    h_spec = renyi(torch.abs(fft_vec) ** 2, fft_vec.shape[-1])
    return h_time + h_spec, h_time, h_spec


def running_2d_renyi(S, alpha=3.0):
    """Rényi entropy of the spectrogram prefix ``S[..., :m, :]`` for every frame.

    ``S`` has shape ``(frames, bins)`` or ``(batch, frames, bins)``.
    """
    power = torch.abs(S) ** 2
    total_power = torch.cumsum(torch.sum(power, dim=-1), dim=-1) + 1e-12
    total_power_alpha = torch.cumsum(torch.sum(power ** alpha, dim=-1), dim=-1) + 1e-12
    return (torch.log2(total_power_alpha) - alpha * torch.log2(total_power)) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Companion (filtering) stream
# ---------------------------------------------------------------------------
class _CompanionStream:
    """Builds the filtering spectrogram while the effective windows arrive.

    Same bookkeeping as the NumPy version (see its docstring). All updates
    create new tensors instead of writing in place, so gradients flow through
    the envelope.
    """

    def __init__(self, x_padded, pad_len, n_fft, hop_size, max_length):
        self.x_padded = x_padded                    # (batch, padded samples)
        self.pad_len = pad_len
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.max_length = int(max_length)
        self.blocks_per_window = self.max_length // hop_size

        self.pending = x_padded.new_zeros((1, self.max_length))
        self.finished = deque(maxlen=self.blocks_per_window)
        self.waiting_windows = deque()
        self.frames_pushed = 0
        self.blocks_finished = 0
        self.spectra = []

    def push(self, window):
        """``window`` has shape (1, n_fft) or (batch, n_fft)."""
        self.waiting_windows.append(window)
        self.pending = self.pending + window[:, -self.max_length:]
        self._finish_oldest_block()
        self.frames_pushed += 1
        if self.blocks_finished >= len(self.spectra) + self.blocks_per_window:
            self._emit_oldest_frame()

    def finish(self):
        """Emit the frames still waiting; returns (batch, frames, bins)."""
        # No more windows will arrive, so the pending blocks are final as well.
        while len(self.spectra) < self.frames_pushed:
            self._finish_oldest_block()
            if self.blocks_finished >= len(self.spectra) + self.blocks_per_window:
                self._emit_oldest_frame()
        if not self.spectra:
            batch = self.x_padded.shape[0]
            return torch.zeros((batch, 0, self.n_fft // 2 + 1),
                               dtype=_complex_dtype(self.x_padded.dtype),
                               device=self.x_padded.device)
        return torch.stack(self.spectra, dim=1)

    def _finish_oldest_block(self):
        hop = self.hop_size
        self.blocks_finished += 1
        self.finished.append(self.pending[:, :hop])
        self.pending = F.pad(self.pending[:, hop:], (0, hop))

    def _emit_oldest_frame(self):
        frame_index = len(self.spectra)
        window = self.waiting_windows.popleft()
        envelope = torch.cat(tuple(self.finished), dim=1)

        end = frame_end(frame_index, self.pad_len, self.hop_size)
        samples = self.x_padded[:, end - self.max_length:end]
        covered = envelope > ENVELOPE_FLOOR
        safe_envelope = torch.where(covered, envelope, torch.ones_like(envelope))
        flattened = torch.where(covered, samples / safe_envelope, torch.zeros_like(samples))

        frame = F.pad(flattened * window[:, -self.max_length:],
                      (self.n_fft - self.max_length, 0))
        self.spectra.append(torch.fft.rfft(frame, dim=-1))


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------
def universal_short_time_fourier_transform(x, n_fft=256, hop_size=32, beta=1.5,
                                           mode="analysis"):
    """Adaptive STFT of ``x`` with shape ``(samples,)`` or ``(batch, samples)``.

    Parameters are the same as the NumPy backend. Returns a ``USTFTResult``;
    every tensor gets a leading batch axis when ``x`` is 2-D, and
    ``filtering_spectrogram`` is ``None`` in analysis mode.
    """
    check_mode(mode)
    single_signal = x.dim() == 1
    if single_signal:
        x = x.unsqueeze(0)

    batch, device, dtype = x.shape[0], x.device, x.dtype
    x_padded, pad_len, num_frames = _padded_signal(x, n_fft, hop_size)
    lengths, windows = _window_bank(n_fft, hop_size, dtype, device)
    lengths_t = torch.as_tensor(lengths, dtype=dtype, device=device)
    num_states = len(lengths)
    num_bins = n_fft // 2 + 1
    transitions = _run_length_transition_matrix(lengths, hop_size, dtype, device)

    companion = None
    if mode == "filtering":
        companion = _CompanionStream(x_padded, pad_len, n_fft, hop_size, lengths[-1])

    def zeros(*shape, complex_valued=False):
        kind = _complex_dtype(dtype) if complex_valued else dtype
        return torch.zeros(shape, dtype=kind, device=device)

    analysis = zeros(batch, num_frames, num_bins, complex_valued=True)
    pmf_history = zeros(batch, num_frames, num_states)
    candidate_specs = zeros(batch, num_states, num_frames, num_bins, complex_valued=True)
    candidate_costs = zeros(batch, num_frames, num_states)
    candidate_h_time = zeros(batch, num_frames, num_states)
    candidate_h_spec = zeros(batch, num_frames, num_states)
    blended_costs = zeros(batch, num_frames)
    blended_h_time = zeros(batch, num_frames)
    blended_h_spec = zeros(batch, num_frames)

    prior = zeros(batch, num_states)
    prior[:, 0] = 1.0                # start on the shortest window
    shortest_only = torch.zeros_like(prior)
    shortest_only[:, 0] = 1.0

    for m in range(num_frames):
        end = frame_end(m, pad_len, hop_size)
        frame = x_padded[:, end - n_fft:end]

        # 1-3. Window, FFT and score every candidate that fits in the samples
        # seen so far. The lengths are sorted, so these are the first ones.
        n_available = int(np.searchsorted(lengths, (m + 1) * hop_size, side="right"))
        windowed = frame.unsqueeze(1) * windows[:n_available]          # (batch, avail, n_fft)
        spectra = torch.fft.rfft(windowed, dim=-1)
        costs, h_time, h_spec = calculate_physics_loss_renyi(
            windowed, spectra, lengths_t[:n_available])
        candidate_specs[:, :n_available, m] = spectra
        candidate_costs[:, m, :n_available] = costs
        candidate_h_time[:, m, :n_available] = h_time
        candidate_h_spec[:, m, :n_available] = h_spec

        # 4. Scores -> likelihood -> posterior.
        if n_available > 1:
            spread = torch.std(costs, dim=1, keepdim=True, unbiased=False)
        else:
            spread = torch.zeros((batch, 1), dtype=dtype, device=device)
        min_cost = torch.min(costs, dim=1, keepdim=True).values
        likelihood = F.pad(torch.exp(-beta / (spread + 1e-6) * (costs - min_cost)),
                           (0, num_states - n_available))
        posterior = prior * likelihood
        evidence = torch.sum(posterior, dim=1, keepdim=True)
        degenerate = evidence < 1e-30
        posterior = torch.where(degenerate, shortest_only,
                                posterior / torch.where(degenerate, torch.ones_like(evidence), evidence))
        pmf_history[:, m] = posterior

        # 5. Output: the posterior mixture of candidate spectra.
        effective_window = posterior @ windows                         # (batch, n_fft)
        analysis_frame = torch.sum(posterior[:, :n_available, None] * spectra, dim=1)
        analysis[:, m] = analysis_frame
        blended_costs[:, m], blended_h_time[:, m], blended_h_spec[:, m] = \
            calculate_physics_loss_renyi(frame * effective_window, analysis_frame,
                                         posterior @ lengths_t)
        if companion is not None:
            companion.push(effective_window)

        # 6. Prior for the next frame.
        prior = _next_prior(posterior, transitions)

    result = USTFTResult(
        analysis_spectrogram=analysis,
        filtering_spectrogram=companion.finish() if companion is not None else None,
        dominant_lengths=lengths_t[torch.argmax(pmf_history, dim=-1)],
        pmf_history=pmf_history,
        blend_info=None,
        candidate_entropies=candidate_costs,
        blended_entropies=blended_costs,
        blended_h_time=blended_h_time,
        blended_h_spec=blended_h_spec,
        candidate_h_time=candidate_h_time,
        candidate_h_spec=candidate_h_spec,
        all_candidate_spectrograms=candidate_specs,
    )
    if single_signal:
        result = USTFTResult(*(
            value.squeeze(0) if torch.is_tensor(value) else value for value in result))

    # blend_info uses NumPy weights, like the NumPy backend. One device-to-host
    # copy for the whole signal avoids a GPU sync per frame.
    weights = result.pmf_history.detach().cpu().numpy()
    frame_axis = 0 if single_signal else 1
    blend_info = [(lengths, np.take(weights, m, axis=frame_axis)) for m in range(num_frames)]
    return result._replace(blend_info=blend_info)


# ---------------------------------------------------------------------------
# Reusing a known trajectory
# ---------------------------------------------------------------------------
def _effective_windows(blend_info, batch, num_frames, n_fft, hop_size, dtype, device):
    """Return the effective window of every frame, each shaped (rows, n_fft).

    ``rows`` is 1 for a trajectory shared by the whole batch, or ``batch``.
    """
    if len(blend_info) < num_frames:
        raise ValueError("blend_info must contain one entry per signal frame")
    grid, windows = _window_bank(n_fft, hop_size, dtype, device)
    state_of = {int(length): state for state, length in enumerate(grid)}

    effective = []
    for m in range(num_frames):
        frame_lengths, weights = blend_info[m]
        try:
            states = [state_of[int(length)] for length in frame_lengths]
        except KeyError:
            raise ValueError("blend_info contains a length outside the candidate grid") from None
        weights = torch.as_tensor(weights, dtype=dtype, device=device)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        if weights.shape[0] not in (1, batch) or weights.shape[1] != len(states):
            raise ValueError("blend weights must have shape (states,), (1, states) or (batch, states)")
        effective.append(weights @ windows[states])
    return effective


def _prepare_batch(x):
    single_signal = x.dim() == 1
    return (x.unsqueeze(0) if single_signal else x), single_signal


def generate_analysis_spectrogram(x, blend_info, n_fft=256, hop_size=32):
    """Analysis spectrogram of ``x`` along a known trajectory ``blend_info``.

    ``x`` may be ``(samples,)`` or ``(batch, samples)``; a trajectory with one
    row of weights per frame is shared by the whole batch.
    """
    x, single_signal = _prepare_batch(x)
    x_padded, _, num_frames = _padded_signal(x, n_fft, hop_size)
    windows = _effective_windows(blend_info, x.shape[0], num_frames, n_fft, hop_size,
                                 x.dtype, x.device)
    frames = _causal_frames(x_padded, num_frames, n_fft, hop_size)
    if num_frames:
        analysis = torch.fft.rfft(frames * torch.stack(windows, dim=1), dim=-1)
    else:
        analysis = torch.fft.rfft(frames, dim=-1)
    return analysis.squeeze(0) if single_signal else analysis


def generate_filtering_spectrogram(x, blend_info, n_fft=256, hop_size=32):
    """Companion (filtering) spectrogram of ``x`` along a known trajectory.

    Inverting it with :func:`inverse_universal_spectrogram` gives back ``x``.
    """
    x, single_signal = _prepare_batch(x)
    x_padded, pad_len, num_frames = _padded_signal(x, n_fft, hop_size)
    longest = candidate_lengths(n_fft, hop_size)[-1]
    companion = _CompanionStream(x_padded, pad_len, n_fft, hop_size, longest)
    for window in _effective_windows(blend_info, x.shape[0], num_frames, n_fft, hop_size,
                                     x.dtype, x.device):
        companion.push(window)
    filtering = companion.finish()
    return filtering.squeeze(0) if single_signal else filtering


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

    Accepts ``(frames, bins)`` or ``(batch, frames, bins)``. See the NumPy
    version for details; ``max_L`` defaults to the longest candidate window.
    """
    if spectrogram is None:
        raise ValueError(
            "no filtering spectrogram: run the transform with mode='filtering'")
    if max_L is None:
        max_L = int(candidate_lengths(n_fft, hop_size)[-1])
    single_signal = spectrogram.dim() == 2
    if single_signal:
        spectrogram = spectrogram.unsqueeze(0)

    batch, num_frames, _ = spectrogram.shape
    pad_len, buffer_len, _ = frame_geometry(num_original_samples, n_fft, hop_size)
    buffer_len = max(buffer_len, num_frames * hop_size + pad_len + n_fft)

    # Rotate so each window's last sample lands at index max_L, then
    # overlap-add the frames, frame m starting at its end minus max_L.
    frames = torch.roll(torch.fft.irfft(spectrogram, n=n_fft, dim=-1), shifts=max_L, dims=-1)
    first_start = frame_end(0, pad_len, hop_size) - max_L
    if num_frames:
        added_length = (num_frames - 1) * hop_size + n_fft
        added = F.fold(frames.transpose(1, 2), output_size=(1, added_length),
                       kernel_size=(1, n_fft), stride=(1, hop_size)).reshape(batch, added_length)
        output = F.pad(added, (first_start, buffer_len - first_start - added_length))
    else:
        output = frames.new_zeros((batch, buffer_len))

    output = output[:, pad_len:pad_len + num_original_samples]
    return output.squeeze(0) if single_signal else output


# ---------------------------------------------------------------------------
# Several channels
# ---------------------------------------------------------------------------
def multichannel_stft(x_multi, reference_sensor_id, n_fft=256, beta=2.0, hop_size=32,
                      mode="analysis"):
    """Adaptive STFT of several channels along one shared trajectory.

    ``x_multi`` has shape ``(channels, samples)``. The trajectory is tracked on
    the reference channel and applied to all channels in one batched call.
    Returns a ``USTFTResult`` whose two spectrograms have a leading channel
    axis; the other fields describe the reference channel.
    """
    check_mode(mode)
    reference = universal_short_time_fourier_transform(
        x_multi[reference_sensor_id], n_fft=n_fft, hop_size=hop_size, beta=beta, mode=mode)

    others = [c for c in range(x_multi.shape[0]) if c != reference_sensor_id]
    analysis = torch.zeros((x_multi.shape[0],) + reference.analysis_spectrogram.shape,
                           dtype=reference.analysis_spectrogram.dtype, device=x_multi.device)
    analysis[reference_sensor_id] = reference.analysis_spectrogram
    filtering = None
    if mode == "filtering":
        filtering = torch.zeros_like(analysis)
        filtering[reference_sensor_id] = reference.filtering_spectrogram

    if others:
        analysis[others] = generate_analysis_spectrogram(
            x_multi[others], reference.blend_info, n_fft, hop_size)
        if mode == "filtering":
            filtering[others] = generate_filtering_spectrogram(
                x_multi[others], reference.blend_info, n_fft, hop_size)

    return reference._replace(analysis_spectrogram=analysis, filtering_spectrogram=filtering)
