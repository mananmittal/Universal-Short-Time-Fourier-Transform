"""Reproduce the Universal STFT paper experiments from the released package.

The dyadic transform is imported from :mod:`universal_stft`; this file contains
only experimental baselines, metrics, audits, tables, and figures.  The
hop-linear tracker is retained solely as a candidate-grid ablation.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import signal


from universal_stft.numpy import stft as _repository_stft
from benchmarks.figure_export import save_figure, resolve_figure, paper_figure_build


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WAV_ROOT = _REPOSITORY_ROOT / "benchmarks" / "wav"
_PAPER_AUDIO = (
    "hyperbolic_chirp.wav",
    "humpback_whale_song.wav",
    "snapping_shrimp.wav",
)
_PAPER_AUDIO_SHA256 = {
    "hyperbolic_chirp.wav": (
        "e95a075a4554a855536739d7eed11c9e36f777d6b6e40a4d8b043c134f94407c"
    ),
    "humpback_whale_song.wav": (
        "261157108e59afcd64984d01ef8bf632dbdad8c88ba261622360c35c8ee7295f"
    ),
    "snapping_shrimp.wav": (
        "98fc0a1db44be63076d1336b04fa32dc25d6d164652b744a805ae83f69eabdc5"
    ),
}
_CANONICAL_PAPER_FIGURES = (
    "window_length_tradeoff.png",
    "window_length_trellis.png",
    "controlled_evidence.png",
    "spectrogram_dyadic_bank.png",
    "tone-impulse-conflict_trajectory_comparison.png",
    "tone-impulse-conflict_loss_regret_comparison.png",
    "tone-impulse-conflict_joint_renyi3_comparison.png",
    "controlled_snr_joint_renyi3.png",
    "ground_truth_resolution.png",
    "hyperbolic-chirp_spectrogram_comparison.png",
    "hyperbolic-chirp_loss_regret_comparison.png",
    "hyperbolic-chirp_joint_renyi3_comparison.png",
    "hyperbolic-chirp_diagnostics.png",
    "humpback-whale_trajectory_comparison.png",
    "humpback-whale_loss_regret_comparison.png",
    "humpback-whale_joint_renyi3_comparison.png",
    "snapping-shrimp_trajectory_comparison.png",
    "snapping-shrimp_loss_regret_comparison.png",
    "snapping-shrimp_joint_renyi3_comparison.png",
    "field_spectrogram_comparison.png",
)

# Re-export the public NumPy API used by the paper.  These are aliases, not copies.
universal_switching_spectrogram = (
    _repository_stft.universal_short_time_fourier_transform
)
inverse_universal_spectrogram = _repository_stft.inverse_universal_spectrogram
blend_guided_stft = _repository_stft.blend_guided_stft
multichannel_stft = _repository_stft.multichannel_stft
calculate_physics_loss_renyi = _repository_stft.calculate_physics_loss_renyi
running_2d_renyi = _repository_stft.running_2d_renyi
normalized_renyi3_lower_bound = _repository_stft.normalized_renyi3_lower_bound


def candidate_window_lengths(n_fft, hop_size, spacing="geometric"):
    """Return the repository dyadic grid or the isolated hop-linear ablation."""
    if spacing in {"geometric", "dyadic", "log"}:
        return _repository_stft._candidate_lengths(n_fft, hop_size).copy()
    if spacing != "linear":
        raise ValueError("spacing must be 'geometric' (dyadic) or 'linear'")

    _repository_stft._validate_transform_sizes(n_fft, hop_size)
    limit = max(int(n_fft) // 2, int(hop_size))
    return np.arange(int(hop_size), limit + 1, int(hop_size), dtype=np.int64)


def normalized_renyi_entropy(values, alpha=3.0):
    """Scale-invariant Renyi entropy in [0, 1] for any nonzero array."""
    power = np.abs(np.asarray(values)) ** 2
    if power.size <= 1 or np.sum(power) <= np.finfo(float).tiny:
        return 0.0
    pmf = power.ravel() / np.sum(power)
    entropy = np.log2(np.sum(pmf ** alpha)) / (1.0 - alpha)
    return float(entropy / np.log2(pmf.size))


def _require_paper_audio(wav_root):
    """Resolve the three external WAV files used by the complete paper suite."""
    wav_root = Path(wav_root).expanduser().resolve()
    paths = [wav_root / name for name in _PAPER_AUDIO]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            "The complete paper suite requires three checksum-locked WAV files. "
            "Restore them in benchmarks/wav or pass --wav-dir.\n"
            f"Missing from {wav_root}:\n{missing_list}\n"
            "See benchmarks/wav/README.md for filenames and checksums."
        )
    mismatched = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != _PAPER_AUDIO_SHA256[path.name]:
            mismatched.append(path.name)
    if mismatched:
        mismatch_list = "\n".join(f"  - {name}" for name in mismatched)
        raise ValueError(
            "Paper WAV checksum mismatch:\n"
            f"{mismatch_list}\n"
            "See benchmarks/wav/README.md for the expected SHA-256 values."
        )
    return paths


def _require_canonical_paper_figures(figure_dir):
    """Verify that a full run produced every manuscript figure."""
    figure_dir = Path(figure_dir)
    paths = [resolve_figure(figure_dir / name) for name in _CANONICAL_PAPER_FIGURES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Full paper run did not produce canonical figures: "
            + ", ".join(missing)
        )
    return paths


def _milliseconds_per_frame(elapsed_seconds, num_frames):
    """Convert one complete-transform timing to milliseconds per frame."""
    return 1000.0 * float(elapsed_seconds) / int(num_frames)


# ---------------------------------------------------------------------------
# Isolated hop-linear grid ablation
# ---------------------------------------------------------------------------
def _linear_blend_guided_stft(x, blend_info, n_fft, hop_size):
    """Generate both streams for a non-dyadic experimental trajectory.

    The released repository intentionally accepts only its dyadic bank.  This
    local helper exists solely so the paper can compare that bank with the old
    H, 2H, 3H, ... candidate grid under otherwise matched windowing and exact
    two-pass envelope normalization.
    """
    x = np.asarray(x)
    x_padded, pad_len, num_frames = _repository_stft._padded_signal(
        x, n_fft, hop_size
    )
    if len(blend_info) != num_frames:
        raise ValueError("blend_info must contain one entry per output frame")

    lengths = np.unique(np.concatenate([
        np.asarray(frame_lengths, dtype=np.int64)
        for frame_lengths, _ in blend_info
    ]))
    windows = {
        int(L): _repository_stft._gaussian_analysis_window(int(L))
        for L in lengths
    }
    num_bins = n_fft // 2 + 1
    analysis = np.zeros((num_frames, num_bins), dtype=complex)
    filtering = np.zeros_like(analysis)

    for frame in range(num_frames):
        current_end = pad_len + (frame + 1) * hop_size
        frame_lengths, weights = blend_info[frame]
        blended = np.zeros(n_fft)
        for L, weight in zip(frame_lengths, weights):
            if weight > 0.0:
                L = int(L)
                blended[-L:] += (
                    weight
                    * x_padded[current_end - L:current_end]
                    * windows[L]
                )
        analysis[frame] = np.fft.rfft(blended)

    envelope = np.zeros_like(x_padded)
    for frame in range(num_frames):
        frame_lengths, weights = blend_info[frame]
        max_length = int(np.max(frame_lengths))
        blended_window = np.zeros(n_fft)
        for L, weight in zip(frame_lengths, weights):
            if weight > 0.0:
                L = int(L)
                blended_window[-L:] += weight * windows[L]
        blended_window = np.roll(blended_window, max_length)
        start = (frame + 1) * hop_size - max_length + pad_len
        envelope[start:start + n_fft] += blended_window

    flattened = np.zeros_like(x_padded)
    covered = envelope > 1e-9
    flattened[covered] = x_padded[covered] / envelope[covered]

    for frame in range(num_frames):
        current_end = pad_len + (frame + 1) * hop_size
        frame_lengths, weights = blend_info[frame]
        blended = np.zeros(n_fft)
        for L, weight in zip(frame_lengths, weights):
            if weight > 0.0:
                L = int(L)
                buffer = np.zeros(n_fft)
                buffer[-L:] = (
                    flattened[current_end - L:current_end] * windows[L]
                )
                blended += weight * buffer
        filtering[frame] = np.fft.rfft(blended)

    return analysis, filtering


def _linear_grid_ablation_transform(x, n_fft=256, hop_size=32, beta=1.5):
    """Run the repository tracker semantics on the old hop-linear state grid.

    This is deliberately not part of the released algorithm.  It reuses the
    repository's padding, Gaussian window, Renyi score, and collapsed transition
    helpers; only the explicit non-dyadic candidate loop is local.
    """
    x = np.asarray(x)
    x_padded, pad_len, num_frames = _repository_stft._padded_signal(
        x, n_fft, hop_size
    )
    lengths = candidate_window_lengths(n_fft, hop_size, "linear")
    windows = np.zeros((len(lengths), n_fft))
    for state, L in enumerate(lengths):
        windows[state, -L:] = _repository_stft._gaussian_analysis_window(int(L))

    transitions = _repository_stft._run_length_transition_matrix(
        lengths, hop_size
    )
    num_states = len(lengths)
    num_bins = n_fft // 2 + 1
    prior = np.zeros(num_states)
    prior[0] = 1.0

    blend_info = []
    dominant_lengths = np.zeros(num_frames)
    pmf_history = np.zeros((num_frames, num_states))
    candidate_entropies = np.zeros((num_frames, num_states))
    blended_entropies = np.zeros(num_frames)
    blended_h_time = np.zeros(num_frames)
    blended_h_spec = np.zeros(num_frames)
    candidate_h_time = np.zeros((num_frames, num_states))
    candidate_h_spec = np.zeros((num_frames, num_states))
    candidate_spectrograms = np.zeros(
        (num_states, num_frames, num_bins), dtype=complex
    )

    for frame_index in range(num_frames):
        current_end = pad_len + (frame_index + 1) * hop_size
        frame = x_padded[current_end - n_fft:current_end]
        valid = lengths <= (frame_index + 1) * hop_size
        candidate_specs = np.zeros((num_states, num_bins), dtype=complex)
        costs = np.zeros(num_states)

        windowed = frame[None, :] * windows[valid]
        valid_specs = np.fft.rfft(windowed, axis=-1)
        candidate_specs[valid] = valid_specs
        (
            costs[valid],
            candidate_h_time[frame_index, valid],
            candidate_h_spec[frame_index, valid],
        ) = calculate_physics_loss_renyi(
            windowed, valid_specs, lengths[valid]
        )

        valid_costs = costs[valid]
        sigma = np.std(valid_costs) if len(valid_costs) > 1 else 0.0
        dynamic_beta = beta / (sigma + 1e-6)
        likelihood = np.zeros(num_states)
        likelihood[valid] = np.exp(
            -dynamic_beta * (valid_costs - np.min(valid_costs))
        )
        posterior = prior * likelihood
        evidence = np.sum(posterior)
        if evidence < 1e-30:
            posterior = np.zeros(num_states)
            posterior[0] = 1.0
        else:
            posterior /= evidence

        blended_frame = np.sum(
            posterior[:, None] * (frame * windows), axis=0
        )
        blended_spec = np.sum(
            posterior[:, None] * candidate_specs, axis=0
        )
        expected_length = np.sum(posterior * lengths)
        (
            blended_entropies[frame_index],
            blended_h_time[frame_index],
            blended_h_spec[frame_index],
        ) = calculate_physics_loss_renyi(
            blended_frame, blended_spec, expected_length
        )

        blend_info.append((lengths, posterior.copy()))
        dominant_lengths[frame_index] = lengths[np.argmax(posterior)]
        pmf_history[frame_index] = posterior
        candidate_entropies[frame_index] = costs
        candidate_spectrograms[:, frame_index] = candidate_specs

        mass = posterior[:, None] * transitions
        prior = np.zeros(num_states)
        prior[0] = np.sum(mass[:, 0])
        prior += mass[:, 1]
        prior[1:] += mass[:-1, 2]
        prior /= np.sum(prior)

    # Build both streams from the finished trajectory.
    analysis, filtering = _linear_blend_guided_stft(
        x, blend_info, n_fft, hop_size
    )
    return (
        analysis,
        filtering,
        dominant_lengths,
        pmf_history,
        blend_info,
        candidate_entropies,
        blended_entropies,
        blended_h_time,
        blended_h_spec,
        candidate_h_time,
        candidate_h_spec,
        candidate_spectrograms,
    )


def _run_grid_transform(x, n_fft, hop_size, beta, spacing):
    """Dispatch to released dyadic code or the isolated linear ablation."""
    if spacing in {"geometric", "dyadic", "log"}:
        return universal_switching_spectrogram(
            x, n_fft=n_fft, hop_size=hop_size, beta=beta, mode="filtering"
        )
    if spacing == "linear":
        return _linear_grid_ablation_transform(
            x, n_fft=n_fft, hop_size=hop_size, beta=beta
        )
    raise ValueError("spacing must be 'geometric' (dyadic) or 'linear'")


# =============================================================================
# Reproducible paper experiments
# =============================================================================

def _one_hot_blend_info(lengths, state_indices):
    """Build a deterministic trajectory compatible with ``blend_guided_stft``."""
    lengths = np.asarray(lengths, dtype=int)
    blend_info = []
    for index in np.asarray(state_indices, dtype=int):
        weights = np.zeros(len(lengths))
        weights[index] = 1.0
        blend_info.append((lengths, weights))
    return blend_info


def _benchmark_signal(seed=0, snr_db=np.inf, fs=4096, duration=4.0):
    """Controlled conflict signal with known tonal and transient targets."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration), dtype=float) / fs
    x = np.zeros_like(t)

    # A fast sweep rewards short context in the first second.
    first = t < 1.0
    x[first] = 0.8 * signal.chirp(t[first], f0=180.0, t1=1.0, f1=1750.0,
                                  method="quadratic")

    # Two close stationary tones reward the longest candidate thereafter.
    steady = ~first
    x[steady] = (0.55 * np.sin(2 * np.pi * 500.0 * t[steady])
                 + 0.55 * np.sin(2 * np.pi * 532.0 * t[steady] + 0.3))

    # Isolated one-sample impulses test whether the tracker can reset rapidly.
    click_times = np.asarray([2.0, 3.0])
    for click_time in click_times:
        index = int(round(click_time * fs))
        if index < len(x):
            x[index] += 8.0

    if np.isfinite(snr_db):
        signal_power = np.mean(x ** 2)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        x += rng.normal(scale=np.sqrt(noise_power), size=len(x))

    return x, {
        "fs": fs,
        "tone_frequencies": (500.0, 532.0),
        "long_context_start": 1.0,
        "click_times": click_times,
    }


def _tone_separation_db(S, fs, n_fft, hop_size, tone_frequencies, click_times):
    """Peak-to-valley separation for the known close-tone pair (larger is better)."""
    frame_times = (np.arange(S.shape[0]) + 1) * hop_size / fs
    stable = (frame_times >= 1.25) & (frame_times <= frame_times[-1] - 0.1)
    for click_time in click_times:
        stable &= np.abs(frame_times - click_time) > 0.22
    if not np.any(stable):
        return np.nan

    spectrum = np.mean(np.abs(S[stable]) ** 2, axis=0)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    f1, f2 = tone_frequencies

    def local_peak(freq):
        mask = np.abs(frequencies - freq) <= 8.0
        return np.max(spectrum[mask]) if np.any(mask) else np.nan

    between = (frequencies > f1 + 8.0) & (frequencies < f2 - 8.0)
    if not np.any(between):
        return np.nan
    peak = min(local_peak(f1), local_peak(f2))
    valley = np.min(spectrum[between])
    return float(10.0 * np.log10((peak + 1e-30) / (valley + 1e-30)))


def _transient_width_ms(S, fs, n_fft, hop_size, click_times):
    """Equivalent high-frequency response width around each known impulse."""
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    high_band = frequencies >= 1000.0
    frame_times = (np.arange(S.shape[0]) + 1) * hop_size / fs
    envelope = np.sum(np.abs(S[:, high_band]) ** 2, axis=1)
    widths = []
    for click_time in click_times:
        local = np.abs(frame_times - click_time) <= 0.30
        response = envelope[local]
        if len(response) == 0:
            continue
        floor = np.percentile(response, 20)
        response = np.maximum(response - floor, 0.0)
        peak = np.max(response)
        if peak > 0:
            widths.append(np.sum(response / peak) * hop_size / fs * 1000.0)
    return float(np.mean(widths)) if widths else np.nan


def _transient_peak_delay_ms(S, fs, n_fft, hop_size, click_times):
    """Delay from each known impulse to its high-band energy maximum."""
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    high_band = frequencies >= 1000.0
    frame_times = (np.arange(S.shape[0]) + 1) * hop_size / fs
    envelope = np.sum(np.abs(S[:, high_band]) ** 2, axis=1)
    delays = []
    for click_time in click_times:
        local_indices = np.flatnonzero(
            (frame_times >= click_time) & (frame_times <= click_time + 0.30)
        )
        if len(local_indices):
            peak_index = local_indices[np.argmax(envelope[local_indices])]
            delays.append(1000.0 * (frame_times[peak_index] - click_time))
    return float(np.mean(delays)) if delays else np.nan


def _trajectory_metrics(length_path, framewise_path, fs, hop_size, metadata,
                        min_candidate_length, max_candidate_length, frame_offset=0):
    """Chattering, distance to myopic choice, and response latency."""
    length_path = np.asarray(length_path, dtype=float)
    framewise_path = np.asarray(framewise_path, dtype=float)
    duration = len(length_path) * hop_size / fs
    log_path = np.log2(length_path)
    variation = np.sum(np.abs(np.diff(log_path))) / max(duration, 1e-12)
    switch_rate = np.count_nonzero(np.diff(length_path)) / max(duration, 1e-12)
    framewise_error = np.mean(np.abs(np.log2(length_path / framewise_path)))
    frame_times = (np.arange(len(length_path)) + frame_offset + 1) * hop_size / fs

    threshold_long = 0.5 * max_candidate_length
    long_enough = length_path >= threshold_long
    hold_frames = max(2, int(np.ceil(0.10 * fs / hop_size)))
    sustained = np.zeros_like(long_enough, dtype=bool)
    if len(long_enough) >= hold_frames:
        forward_counts = np.convolve(
            long_enough.astype(int), np.ones(hold_frames, dtype=int), mode="valid"
        )
        sustained[:len(forward_counts)] = forward_counts == hold_frames
    after_onset = np.flatnonzero(
        (frame_times >= metadata["long_context_start"])
        & (frame_times <= metadata["long_context_start"] + 0.9)
        & sustained
    )
    acquisition_ms = (1000.0 * (frame_times[after_onset[0]]
                      - metadata["long_context_start"])) if len(after_onset) else np.nan

    reset_latencies = []
    threshold_short = 2.0 * min_candidate_length
    for click_time in metadata["click_times"]:
        reset = np.flatnonzero(
            (frame_times >= click_time)
            & (frame_times <= click_time + 0.30)
            & (length_path <= threshold_short)
        )
        if len(reset):
            reset_latencies.append(1000.0 * (frame_times[reset[0]] - click_time))
    reset_ms = float(np.mean(reset_latencies)) if reset_latencies else np.nan
    return variation, switch_rate, framewise_error, acquisition_ms, reset_ms


def _constrained_hindsight_path(costs):
    """Offline minimum-loss path under stay/grow/reset trellis transitions."""
    costs = np.asarray(costs, dtype=float)
    num_frames, num_states = costs.shape
    dp = np.full((num_frames, num_states), np.inf)
    back = np.full((num_frames, num_states), -1, dtype=int)
    if np.isfinite(costs[0, 0]):
        dp[0, 0] = costs[0, 0]

    for frame in range(1, num_frames):
        for state in range(num_states):
            if not np.isfinite(costs[frame, state]):
                continue
            predecessors = [state]
            if state == 0:
                predecessors.extend(range(1, num_states))
            elif state > 0:
                predecessors.append(state - 1)
            predecessors = np.unique(predecessors)
            best = predecessors[np.argmin(dp[frame - 1, predecessors])]
            dp[frame, state] = dp[frame - 1, best] + costs[frame, state]
            back[frame, state] = best

    state = int(np.argmin(dp[-1]))
    path = np.zeros(num_frames, dtype=int)
    path[-1] = state
    for frame in range(num_frames - 1, 0, -1):
        state = back[frame, state]
        if state < 0:
            state = 0
        path[frame - 1] = state
    return path, float(np.min(dp[-1]))


def _dense_transition_matrix(candidate_lengths, hop_size):
    """Expand the repository transition rows into a dense HMM matrix."""
    rows = _repository_stft._run_length_transition_matrix(
        candidate_lengths, hop_size
    )
    num_states = len(candidate_lengths)
    matrix = np.zeros((num_states, num_states))
    for state in range(num_states):
        matrix[state, 0] += rows[state, 0]
        matrix[state, state] += rows[state, 1]
        if state + 1 < num_states:
            matrix[state, state + 1] += rows[state, 2]
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _hmm_path_code_audit(costs, beta, transition_matrix):
    """Verify the exact HMM mix-loss bound against one allowed offline path.

    Emission losses are the actual standardized negative log likelihoods used
    by the tracker.  For any allowed path, HMM mix loss is at most its emission
    loss plus its negative-log prior probability (path code).  This exact bound
    remains valid with frame-varying beta, but it is not an O(k log M) claim.
    """
    costs = np.asarray(costs, dtype=float)
    num_frames, num_states = costs.shape
    emission_loss = np.full_like(costs, np.inf)
    for frame in range(num_frames):
        valid = np.isfinite(costs[frame])
        valid_costs = costs[frame, valid]
        dynamic_beta = beta / (
            (np.std(valid_costs) if len(valid_costs) > 1 else 0.0) + 1e-6
        )
        emission_loss[frame, valid] = dynamic_beta * (
            valid_costs - np.min(valid_costs)
        )

    prior = np.zeros(num_states)
    prior[0] = 1.0
    mix_loss = 0.0
    mix_curve = np.zeros(num_frames)
    posterior_curve = np.zeros((num_frames, num_states))
    for frame in range(num_frames):
        likelihood = np.zeros(num_states)
        finite = np.isfinite(emission_loss[frame])
        likelihood[finite] = np.exp(-emission_loss[frame, finite])
        evidence = float(np.sum(prior * likelihood))
        if evidence <= 0.0:
            return {
                "mix_log_loss": np.inf,
                "comparator_emission_log_loss": np.inf,
                "comparator_path_code": np.inf,
                "bound_margin": np.nan,
                "average_mix_gap": np.nan,
                "mix_curve": np.full(num_frames, np.nan),
                "comparator_emission_curve": np.full(num_frames, np.nan),
                "certificate_curve": np.full(num_frames, np.nan),
                "regret_curve": np.full(num_frames, np.nan),
                "average_gap_curve": np.full(num_frames, np.nan),
                "path_code_bound_curve": np.full(num_frames, np.nan),
                "comparator_path": np.full(num_frames, -1, dtype=int),
                "posterior_curve": np.full((num_frames, num_states), np.nan),
            }
        mix_loss -= np.log(evidence)
        mix_curve[frame] = mix_loss
        posterior = prior * likelihood / evidence
        posterior_curve[frame] = posterior
        prior = posterior @ transition_matrix

    # Best emission-only path under the same directed support.
    dp = np.full((num_frames, num_states), np.inf)
    back = np.full((num_frames, num_states), -1, dtype=int)
    dp[0, 0] = emission_loss[0, 0]
    for frame in range(1, num_frames):
        for state in range(num_states):
            if not np.isfinite(emission_loss[frame, state]):
                continue
            predecessors = np.flatnonzero(transition_matrix[:, state] > 0.0)
            if len(predecessors) == 0:
                continue
            previous = predecessors[np.argmin(dp[frame - 1, predecessors])]
            dp[frame, state] = dp[frame - 1, previous] + emission_loss[frame, state]
            back[frame, state] = previous

    path = np.zeros(num_frames, dtype=int)
    path[-1] = int(np.argmin(dp[-1]))
    for frame in range(num_frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    comparator_emission = float(np.sum(emission_loss[np.arange(num_frames), path]))
    transition_probabilities = transition_matrix[path[:-1], path[1:]]
    path_code = float(-np.sum(np.log(transition_probabilities)))
    margin = comparator_emission + path_code - mix_loss
    # For every prefix t, D_t^* is the emission loss of the best schedule in
    # hindsight among paths allowed by the same stay/grow/reset support.  Its
    # path code C_t^* is evaluated under the transform's actual transition
    # probabilities.  Therefore Lambda_t - D_t^* <= C_t^* exactly.
    optimal_emission_curve = np.min(dp, axis=1)
    path_code_curve = np.zeros(num_frames)
    for end_frame in range(1, num_frames):
        prefix_path = np.zeros(end_frame + 1, dtype=int)
        prefix_path[-1] = int(np.argmin(dp[end_frame]))
        for frame in range(end_frame, 0, -1):
            prefix_path[frame - 1] = back[frame, prefix_path[frame]]
        probabilities = transition_matrix[prefix_path[:-1], prefix_path[1:]]
        path_code_curve[end_frame] = -np.sum(np.log(probabilities))
    frames = np.arange(1, num_frames + 1)
    regret_curve = mix_curve - optimal_emission_curve
    certificate_curve = optimal_emission_curve + path_code_curve
    tolerance = 1e-9 * max(1.0, float(np.nanmax(certificate_curve)))
    if np.any(mix_curve > certificate_curve + tolerance):
        raise AssertionError("HMM path-code certificate failed numerically")
    return {
        "mix_log_loss": float(mix_loss),
        "comparator_emission_log_loss": comparator_emission,
        "comparator_path_code": path_code,
        "bound_margin": float(margin),
        "average_mix_gap": float((mix_loss - comparator_emission) / num_frames),
        "mix_curve": mix_curve,
        "comparator_emission_curve": optimal_emission_curve,
        "certificate_curve": certificate_curve,
        "regret_curve": regret_curve,
        "average_gap_curve": regret_curve / frames,
        "path_code_bound_curve": path_code_curve / frames,
        "comparator_path": path,
        "posterior_curve": posterior_curve,
    }


def _case_outputs(x, n_fft, hop_size, beta, spacing):
    """Run the trellis once and derive greedy and best-fixed comparators."""
    start = time.perf_counter()
    output = _run_grid_transform(x, n_fft, hop_size, beta, spacing)
    elapsed = time.perf_counter() - start
    (S_trellis, F_trellis, dominant, pmf, blend_info, raw_candidate_costs,
     blended_costs, blended_h_time, blended_h_spec, candidate_h_time,
     candidate_h_spec, candidate_specs) = output
    lengths = np.asarray(blend_info[0][0], dtype=int)
    num_frames = len(blend_info)
    warmup = int(np.ceil(np.max(lengths) / hop_size))

    # The repository preserves zero-filled diagnostics for causally unavailable
    # candidates.  Experiments mask those entries explicitly before selecting a
    # myopic or hindsight path; this does not modify the transform output.
    candidate_costs = np.asarray(raw_candidate_costs, dtype=float).copy()
    samples_seen = (np.arange(num_frames) + 1)[:, None] * hop_size
    candidate_costs[lengths[None, :] > samples_seen] = np.nan

    greedy_indices = np.zeros(num_frames, dtype=int)
    for frame in range(num_frames):
        valid = np.flatnonzero(np.isfinite(candidate_costs[frame]))
        greedy_indices[frame] = valid[np.argmin(candidate_costs[frame, valid])]
    direct_framewise_indices = np.nanargmin(candidate_costs, axis=1)
    if not np.array_equal(greedy_indices, direct_framewise_indices):
        raise AssertionError(
            "framewise greedy path must equal the per-frame metric argmin"
        )
    greedy_blend = _one_hot_blend_info(lengths, greedy_indices)
    guided_transform = (
        blend_guided_stft
        if spacing in {"geometric", "dyadic", "log"}
        else _linear_blend_guided_stft
    )
    S_greedy, F_greedy = guided_transform(
        x, greedy_blend, n_fft, hop_size
    )

    # Select one fixed-support baseline with the scaled Gibbs emission
    # beta*d_mj, where d_mj=(ell_mj-min_r ell_mr)/(std_m+eps).  The post-warm-up
    # suffix is common to every support, so the comparison never rewards a
    # causally unavailable candidate.  This one index is reused in every table
    # and figure; the separate full-plane H3 optimum is an audit reference.
    common_costs = candidate_costs[warmup:]
    common_eta = beta / (np.std(common_costs, axis=1) + 1e-6)
    common_emission = common_eta[:, None] * (
        common_costs - np.min(common_costs, axis=1, keepdims=True)
    )
    fixed_tracking_totals = np.sum(common_emission, axis=0)
    fixed_index = int(np.argmin(fixed_tracking_totals))
    if fixed_tracking_totals[fixed_index] > (
            np.min(fixed_tracking_totals) + 1e-12):
        raise AssertionError(
            "best fixed support is not the cumulative tracking-loss minimum"
        )
    fixed_indices = np.full(num_frames, fixed_index, dtype=int)
    fixed_blend = _one_hot_blend_info(lengths, fixed_indices)
    S_fixed, F_fixed = guided_transform(x, fixed_blend, n_fft, hop_size)

    framewise_indices = np.nanargmin(candidate_costs[warmup:], axis=1)
    if not np.array_equal(framewise_indices, greedy_indices[warmup:]):
        raise AssertionError(
            "post-warm-up framewise path differs from the greedy transform"
        )
    # The displayed hindsight schedule is exactly the theorem comparator:
    # it minimizes standardized emission over the repository's nonzero
    # stay/grow/reset edges.  Transition magnitudes enter only the path code.
    # A raw-cost schedule can differ because the inverse temperature is
    # normalized separately at every frame.
    transition_matrix = _dense_transition_matrix(lengths, hop_size)
    hmm_audit = _hmm_path_code_audit(
        candidate_costs, beta, transition_matrix
    )
    constrained_indices = hmm_audit["comparator_path"]
    constrained_loss = hmm_audit["comparator_emission_log_loss"]
    constrained_blend = _one_hot_blend_info(lengths, constrained_indices)
    S_constrained, F_constrained = guided_transform(
        x, constrained_blend, n_fft, hop_size
    )
    trellis_reconstruction = inverse_universal_spectrogram(
        F_trellis, len(x), hop_size=hop_size, n_fft=n_fft,
        max_L=int(np.max(lengths)),
    )
    trellis_reconstruction_mse = float(np.mean(
        (np.asarray(x) - trellis_reconstruction) ** 2
    ))
    return {
        "trellis": (S_trellis, F_trellis, dominant),
        "constrained": (
            S_constrained, F_constrained, lengths[constrained_indices]
        ),
        "greedy": (S_greedy, F_greedy, lengths[greedy_indices]),
        "best_fixed": (S_fixed, F_fixed, lengths[fixed_indices]),
        "lengths": lengths,
        "pmf": pmf,
        "candidate_costs": candidate_costs,
        "transition_matrix": transition_matrix,
        "blended_costs": blended_costs,
        "blended_h_time": blended_h_time,
        "blended_h_spec": blended_h_spec,
        "candidate_h_time": candidate_h_time,
        "candidate_h_spec": candidate_h_spec,
        "candidate_specs": candidate_specs,
        "framewise_indices": framewise_indices,
        "framewise_indices_full": greedy_indices,
        "constrained_indices": constrained_indices,
        "constrained_loss": constrained_loss,
        "hmm_audit": hmm_audit,
        "fixed_index": fixed_index,
        "fixed_tracking_totals": fixed_tracking_totals,
        "trellis_reconstruction_mse": trellis_reconstruction_mse,
        "warmup": warmup,
        "elapsed": elapsed,
        "output": output,
    }


def _evaluate_case(x, metadata, seed, snr_db, spacing, n_fft=1024,
                   hop_size=32, beta=1.5):
    """Return one row per method plus traces needed for diagnostic figures."""
    case = _case_outputs(x, n_fft, hop_size, beta, spacing)
    fs = metadata["fs"]
    warmup = case["warmup"]
    lengths = case["lengths"]
    costs = case["candidate_costs"][warmup:]
    pmf = case["pmf"][warmup:]
    framewise_indices = case["framewise_indices"]
    framewise_lengths = lengths[framewise_indices]
    framewise_loss = np.min(costs, axis=1)
    expected_loss = np.sum(pmf * costs, axis=1)
    fixed_index = case["fixed_index"]
    hmm_audit = case["hmm_audit"]

    method_losses = {
        "trellis": expected_loss,
        "greedy": framewise_loss,
        "best_fixed": costs[:, fixed_index],
    }
    rows = []
    for method, (analysis, filtering, length_path) in (
            (name, case[name]) for name in ("trellis", "greedy", "best_fixed")):
        reconstruction = inverse_universal_spectrogram(
            filtering, len(x), hop_size=hop_size, n_fft=n_fft,
            max_L=int(np.max(lengths))
        )
        mse = float(np.mean((x - reconstruction) ** 2))
        path_valid = np.asarray(length_path)[warmup:]
        trajectory = _trajectory_metrics(
            path_valid, framewise_lengths, fs, hop_size, metadata,
            min_candidate_length=np.min(lengths),
            max_candidate_length=np.max(lengths),
            frame_offset=warmup,
        )
        rows.append({
            "seed": seed,
            "snr_db": "inf" if not np.isfinite(snr_db) else float(snr_db),
            "spacing": spacing,
            "method": method,
            "states": len(lengths),
            "runtime_ms_per_frame": (
                _milliseconds_per_frame(case["elapsed"], len(length_path))
                if method == "trellis" else np.nan
            ),
            "normalized_renyi3": normalized_renyi_entropy(analysis[warmup:]),
            "mean_framewise_excess": float(
                np.mean(method_losses[method] - framewise_loss)
            ),
            "final_average_framewise_excess": float(
                np.cumsum(method_losses[method] - framewise_loss)[-1]
                / len(framewise_loss)
            ),
            "tone_separation_db": _tone_separation_db(
                analysis, fs, n_fft, hop_size,
                metadata["tone_frequencies"], metadata["click_times"]
            ),
            "transient_width_ms": _transient_width_ms(
                analysis, fs, n_fft, hop_size, metadata["click_times"]
            ),
            "transient_peak_delay_ms": _transient_peak_delay_ms(
                analysis, fs, n_fft, hop_size, metadata["click_times"]
            ),
            "trajectory_variation_octaves_per_s": trajectory[0],
            "switches_per_s": trajectory[1],
            "framewise_length_error_octaves": trajectory[2],
            "long_acquisition_latency_ms": trajectory[3],
            "impulse_reset_latency_ms": trajectory[4],
            "reconstruction_mse": mse,
            "hmm_mix_log_loss_per_frame": (
                hmm_audit["mix_log_loss"] / len(case["candidate_costs"])
                if method == "trellis" else np.nan
            ),
            "hmm_comparator_emission_per_frame": (
                hmm_audit["comparator_emission_log_loss"] / len(case["candidate_costs"])
                if method == "trellis" else np.nan
            ),
            "hmm_path_code_per_frame": (
                hmm_audit["comparator_path_code"] / len(case["candidate_costs"])
                if method == "trellis" else np.nan
            ),
            "hmm_bound_margin": (
                hmm_audit["bound_margin"] if method == "trellis" else np.nan
            ),
            "hmm_average_mix_gap": (
                hmm_audit["average_mix_gap"] if method == "trellis" else np.nan
            ),
        })

    # Audit the legacy statistic and the unproved posterior Jensen upgrade.
    blended = case["blended_costs"][warmup:]
    beta_dynamic = beta / (np.std(costs, axis=1) + 1e-6)
    shifted_candidates = costs - framewise_loss[:, None]
    shifted_blend = blended - framewise_loss
    lhs = np.exp(-beta_dynamic * shifted_blend)
    rhs = np.sum(pmf * np.exp(-beta_dynamic[:, None] * shifted_candidates), axis=1)
    case["audit"] = {
        "legacy_negative_fraction": float(np.mean(shifted_blend < 0.0)),
        "legacy_final_average": float(np.mean(shifted_blend)),
        "posterior_jensen_violation_fraction": float(np.mean(lhs + 1e-12 < rhs)),
        "mean_framewise_expected_excess": float(
            np.mean(expected_loss - framewise_loss)
        ),
        "hmm_mix_log_loss": hmm_audit["mix_log_loss"],
        "hmm_comparator_emission_log_loss": hmm_audit["comparator_emission_log_loss"],
        "hmm_comparator_path_code": hmm_audit["comparator_path_code"],
        "hmm_bound_margin": hmm_audit["bound_margin"],
        "hmm_average_mix_gap": hmm_audit["average_mix_gap"],
    }
    case["hmm_curves"] = {
        "mix": hmm_audit["mix_curve"],
        "comparator_emission": hmm_audit["comparator_emission_curve"],
        "certificate": hmm_audit["certificate_curve"],
        "regret": hmm_audit["regret_curve"],
        "average_gap": hmm_audit["average_gap_curve"],
        "path_code_bound": hmm_audit["path_code_bound_curve"],
    }
    case["hmm_comparator_path"] = hmm_audit["comparator_path"]
    case["legacy_curve"] = {
        "blend_minus_framewise": (
            np.cumsum(shifted_blend) / np.arange(1, len(costs) + 1)
        ),
    }
    return rows, case


def _read_wav_mono(path, max_seconds=5.0):
    """Read a WAV with SciPy only and return floating mono samples."""
    from scipy.io import wavfile

    fs, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim > 1:
        x = np.mean(x.astype(float), axis=1)
    if np.issubdtype(x.dtype, np.integer):
        scale = max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
        x = x.astype(float) / scale
    else:
        x = x.astype(float)
    x = x[:int(fs * max_seconds)]
    x -= np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return fs, x


def _evaluate_real_audio(wav_paths, beta=1.5, return_cases=False):
    """Self-supervised real-audio checks; no ground-truth resolution is claimed."""
    rows = []
    cases = {}
    for wav_path in wav_paths:
        fs, x = _read_wav_mono(wav_path)
        n_fft = 4096
        hop_size = 128
        case = _case_outputs(x, n_fft, hop_size, beta, "geometric")
        cases[Path(wav_path).stem] = {
            "fs": fs,
            "n_fft": n_fft,
            "hop_size": hop_size,
            "case": case,
        }
        warmup = case["warmup"]
        lengths = case["lengths"]
        costs = case["candidate_costs"][warmup:]
        framewise_indices = case["framewise_indices"]
        framewise_lengths = lengths[framewise_indices]
        framewise_loss = np.min(costs, axis=1)
        expected_loss = np.sum(case["pmf"][warmup:] * costs, axis=1)
        fixed_index = case["fixed_index"]

        for method, losses in (
            ("trellis", expected_loss),
            ("greedy", framewise_loss),
            ("best_fixed", costs[:, fixed_index]),
        ):
            analysis, filtering, path = case[method]
            reconstruction = inverse_universal_spectrogram(
                filtering, len(x), hop_size, n_fft, int(np.max(lengths))
            )
            valid_path = np.asarray(path)[warmup:]
            duration = len(valid_path) * hop_size / fs
            rows.append({
                "recording": Path(wav_path).stem,
                "sample_rate": fs,
                "method": method,
                "states": len(lengths),
                "normalized_renyi3": normalized_renyi_entropy(analysis[warmup:]),
                "mean_framewise_excess": float(
                    np.mean(losses - framewise_loss)
                ),
                "trajectory_variation_octaves_per_s": float(
                    np.sum(np.abs(np.diff(np.log2(valid_path)))) / max(duration, 1e-12)
                ),
                "switches_per_s": float(
                    np.count_nonzero(np.diff(valid_path)) / max(duration, 1e-12)
                ),
                "framewise_length_error_octaves": float(np.mean(
                    np.abs(np.log2(valid_path / framewise_lengths))
                )),
                "reconstruction_mse": float(np.mean((x - reconstruction) ** 2)),
                "runtime_ms_per_frame": (
                    _milliseconds_per_frame(case["elapsed"], len(path))
                    if method == "trellis" else np.nan
                ),
            })
    return (rows, cases) if return_cases else rows


def _runtime_scaling(seed=123, beta=1.5):
    """Measure complete forward-transform cost for dyadic and linear banks."""
    rng = np.random.default_rng(seed)
    rows = []
    hop_size = 32
    for n_fft in (256, 512, 1024, 2048, 4096):
        num_frames = max(192, n_fft // (2 * hop_size) + 96)
        x = rng.normal(size=num_frames * hop_size)
        for spacing in ("geometric", "linear"):
            lengths = candidate_window_lengths(n_fft, hop_size, spacing)
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                _run_grid_transform(x, n_fft, hop_size, beta, spacing)
                timings.append(time.perf_counter() - start)
            elapsed = float(np.median(timings))
            rows.append({
                "n_fft": n_fft,
                "hop_size": hop_size,
                "spacing": spacing,
                "states": len(lengths),
                "milliseconds_per_frame": _milliseconds_per_frame(
                    elapsed, num_frames
                ),
                "window_bank_kib": len(lengths) * n_fft * 8 / 1024.0,
            })
    return rows


def _audit_legacy_result_csvs(results_root):
    """Inventory what the repository's pre-existing benchmark CSVs establish."""
    rows = []
    for csv_path in sorted(Path(results_root).rglob("*_metrics.csv")):
        with csv_path.open(newline="") as stream:
            metrics = list(csv.DictReader(stream))
        if not metrics:
            continue
        universal = next((row for row in metrics if row["Model"] == "Universal STFT"), None)
        fixed = [row for row in metrics if row["Model"].startswith("Fixed L=")]
        if universal is None or not fixed:
            continue

        def numeric(row, key):
            try:
                return float(row[key])
            except (KeyError, TypeError, ValueError):
                return np.nan

        universal_kurtosis = numeric(universal, "Env Kurtosis")
        fixed_kurtosis = [numeric(row, "Env Kurtosis") for row in fixed]
        best_fixed_kurtosis = float(np.nanmax(fixed_kurtosis))
        rows.append({
            "result": csv_path.parent.name,
            "category": csv_path.parents[1].name,
            "universal_mse": numeric(universal, "MSE"),
            "best_fixed_mse": float(np.nanmin([numeric(row, "MSE") for row in fixed])),
            "universal_envelope_kurtosis": universal_kurtosis,
            "best_fixed_envelope_kurtosis": best_fixed_kurtosis,
            "universal_kurtosis_win": int(universal_kurtosis >= best_fixed_kurtosis),
        })
    return rows


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _median(rows, key):
    values = []
    for row in rows:
        value = row.get(key, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.median(values)) if values else np.nan


def _spectrogram_db(spectrograms, floor_db=-70.0):
    """Convert several complex spectrograms to one honest shared dB scale."""
    powers = [np.abs(np.asarray(S)) ** 2 for S in spectrograms]
    reference = max(float(np.max(power)) for power in powers)
    reference = max(reference, np.finfo(float).tiny)
    return [
        np.maximum(10.0 * np.log10((power + reference * 1e-14) / reference),
                   floor_db)
        for power in powers
    ]


def _rfft_full_frequency_multiplicities(n_fft):
    """Multiplicity of each rFFT bin in the corresponding full DFT grid."""
    n_fft = int(n_fft)
    multiplicity = np.full(n_fft // 2 + 1, 2.0)
    multiplicity[0] = 1.0
    if n_fft % 2 == 0:
        multiplicity[-1] = 1.0
    if int(np.sum(multiplicity)) != n_fft:
        raise AssertionError("rFFT multiplicities do not reproduce the full grid")
    return multiplicity


def _joint_spectrogram_renyi3_prefix(spectrogram, n_fft):
    """Rényi-3 of each raw 2-D spectrogram prefix, computed exactly once.

    The positive discrete density is the full-frequency spectrogram power on
    every frame observed so far.  The input is an rFFT spectrogram, so interior
    bins represent two equal-power full-DFT cells while DC and (for even
    ``n_fft``) Nyquist represent one.  For prefix ``t`` this computes

        H_3(P_t) = -1/2 log2 sum_{m,k} P_t[m,k]^3,

    where ``P_t`` is normalized over the complete t-by-n_fft grid.  The
    cumulative sums below are taken directly from raw power and raw cubed
    power.  They are not a cumulative sum of an already cumulative entropy.
    """
    spectrogram = np.asarray(spectrogram)
    n_fft = int(n_fft)
    if spectrogram.ndim != 2 or spectrogram.shape[1] != n_fft // 2 + 1:
        raise ValueError("spectrogram must have shape (frames, n_fft // 2 + 1)")
    if spectrogram.shape[0] == 0:
        raise ValueError("spectrogram must contain at least one frame")

    power = np.abs(spectrogram) ** 2
    if not np.all(np.isfinite(power)):
        raise ValueError("spectrogram power must be finite")
    scale = float(np.max(power))
    if scale <= np.finfo(float).tiny:
        raise ValueError("spectrogram region must have positive energy")
    power = power / scale
    multiplicity = _rfft_full_frequency_multiplicities(n_fft)
    frame_energy = np.sum(power * multiplicity[None, :], axis=1)
    frame_cubic_energy = np.sum(
        power ** 3 * multiplicity[None, :], axis=1
    )
    cumulative_energy = np.cumsum(frame_energy)
    cumulative_cubic_energy = np.cumsum(frame_cubic_energy)
    if np.any(cumulative_energy <= 0.0) or np.any(cumulative_cubic_energy <= 0.0):
        raise AssertionError("positive spectrogram prefixes must have positive moments")

    entropy_bits = 0.5 * (
        3.0 * np.log2(cumulative_energy)
        - np.log2(cumulative_cubic_energy)
    )
    full_grid_cells = n_fft * np.arange(1, len(entropy_bits) + 1)
    max_entropy_bits = np.log2(full_grid_cells)
    normalized_entropy = entropy_bits / max_entropy_bits
    tolerance = 2e-12
    if np.any(entropy_bits < -tolerance):
        raise AssertionError("Rényi-3 entropy cannot be negative on a PMF")
    if np.any(entropy_bits > max_entropy_bits + tolerance):
        raise AssertionError("Rényi-3 exceeds the uniform-grid maximum")
    if np.any((normalized_entropy < -tolerance)
              | (normalized_entropy > 1.0 + tolerance)):
        raise AssertionError("normalized Rényi-3 must lie in [0, 1]")

    final_energy = cumulative_energy[-1]
    pmf_sum = float(np.sum(power * multiplicity[None, :]) / final_energy)
    direct_sum_p_cubed = float(
        np.sum(power ** 3 * multiplicity[None, :]) / final_energy ** 3
    )
    direct_entropy_bits = -0.5 * np.log2(direct_sum_p_cubed)
    if not np.isclose(entropy_bits[-1], direct_entropy_bits, rtol=2e-13,
                      atol=2e-13):
        raise AssertionError("prefix and direct full-region Rényi-3 disagree")
    return {
        "entropy_bits": entropy_bits,
        "normalized_entropy": normalized_entropy,
        "effective_cells": 2.0 ** entropy_bits,
        "full_grid_cells": full_grid_cells,
        "pmf_sum_error": abs(pmf_sum - 1.0),
    }


def _joint_renyi3_case_record(label, short_label, slug, case, fs, n_fft,
                              hop_size, case_type, seed=None, snr_db=None):
    """Evaluate four released-analysis variants on one identical 2-D grid."""
    warmup = int(case["warmup"])
    n_fft = int(n_fft)
    hop_size = int(hop_size)
    fs = float(fs)
    candidate_specs = np.asarray(case["candidate_specs"])
    candidate_metrics = [
        _joint_spectrogram_renyi3_prefix(
            candidate_specs[state, warmup:], n_fft
        )
        for state in range(len(case["lengths"]))
    ]
    joint_optimal_fixed_index = int(np.argmin([
        metric["entropy_bits"][-1] for metric in candidate_metrics
    ]))
    fixed_minimum = candidate_metrics[joint_optimal_fixed_index]["entropy_bits"][-1]
    if np.any([
            metric["entropy_bits"][-1] < fixed_minimum - 2e-12
            for metric in candidate_metrics]):
        raise AssertionError("selected fixed support is not the finite-bank minimum")
    fixed_index = int(case["fixed_index"])
    fixed_spectrum = np.asarray(case["best_fixed"][0])[warmup:]
    if not np.allclose(
            fixed_spectrum, candidate_specs[fixed_index, warmup:],
            rtol=2e-13, atol=2e-13):
        raise AssertionError(
            "fixed comparator differs from the selected candidate transform"
        )

    spectra = {
        "posterior_blend": np.asarray(case["trellis"][0])[warmup:],
        "constrained_schedule": np.asarray(case["constrained"][0])[warmup:],
        "framewise_selection": np.asarray(case["greedy"][0])[warmup:],
        "best_fixed_support": fixed_spectrum,
    }
    metrics = {
        method: _joint_spectrogram_renyi3_prefix(values, n_fft)
        for method, values in spectra.items()
    }
    for method, metric in metrics.items():
        if metric["pmf_sum_error"] > 5e-14:
            raise AssertionError(f"{method} joint PMF does not sum to one")

    # Explicit scale-invariance check on the primary controlled output.
    scaled = _joint_spectrogram_renyi3_prefix(
        7.25 * spectra["posterior_blend"], n_fft
    )
    if not np.allclose(
            metrics["posterior_blend"]["entropy_bits"],
            scaled["entropy_bits"], rtol=2e-13, atol=2e-13):
        raise AssertionError("joint Rényi-3 must be invariant to signal scale")

    # Independently expand a small rFFT probe to explicit full-DFT cells.
    # This guards against the common but incorrect operation (2E)^3: each
    # conjugate interior cell must be duplicated before taking the third moment.
    probe = np.abs(spectra["posterior_blend"][:3]) ** 2
    if n_fft % 2 == 0:
        conjugate_cells = probe[:, 1:-1][:, ::-1]
    else:
        conjugate_cells = probe[:, 1:][:, ::-1]
    explicit_full_power = np.concatenate((probe, conjugate_cells), axis=1)
    if explicit_full_power.shape[1] != n_fft:
        raise AssertionError("explicit full-DFT probe has the wrong width")
    explicit_energy = np.cumsum(np.sum(explicit_full_power, axis=1))
    explicit_cubic = np.cumsum(np.sum(explicit_full_power ** 3, axis=1))
    explicit_entropy = 0.5 * (
        3.0 * np.log2(explicit_energy) - np.log2(explicit_cubic)
    )
    if not np.allclose(
            metrics["posterior_blend"]["entropy_bits"][:len(probe)],
            explicit_entropy, rtol=2e-13, atol=2e-13):
        raise AssertionError("rFFT multiplicities differ from explicit duplication")

    num_frames = spectra["posterior_blend"].shape[0]
    return {
        "label": label,
        "short_label": short_label,
        "slug": slug,
        "case_type": case_type,
        "seed": seed,
        "snr_db": snr_db,
        "metrics": metrics,
        "candidate_metrics": candidate_metrics,
        "best_fixed_length": int(case["lengths"][fixed_index]),
        "best_fixed_index": fixed_index,
        "joint_optimal_fixed_length": int(
            case["lengths"][joint_optimal_fixed_index]
        ),
        "joint_optimal_fixed_h3_bits": float(fixed_minimum),
        "fixed_selection_rule": (
            "minimum cumulative scaled Gibbs emission beta*d on the common "
            "post-warm-up region"
        ),
        "candidate_lengths": np.asarray(case["lengths"], dtype=int),
        "n_fft": n_fft,
        "hop_size": hop_size,
        "sample_rate": fs,
        "num_frames": num_frames,
        "rfft_bins": n_fft // 2 + 1,
        "full_frequency_cells_per_frame": n_fft,
        "time_start_s": (warmup + 1) * hop_size / fs,
        "time_end_s": (warmup + num_frames) * hop_size / fs,
        "prefix_time_s": (
            (warmup + np.arange(1, num_frames + 1)) * hop_size / fs
        ),
    }


def _joint_renyi3_rows(records):
    """Flatten joint-entropy records for exact, inspectable CSV output."""
    rows = []
    for record in records:
        total_cells = record["num_frames"] * record["n_fft"]
        for method, metric in record["metrics"].items():
            rows.append({
                "case": record["slug"],
                "case_type": record["case_type"],
                "method": method,
                "seed": "" if record["seed"] is None else record["seed"],
                "snr_db": "" if record["snr_db"] is None else record["snr_db"],
                "renyi_order": 3,
                "h3_bits": float(metric["entropy_bits"][-1]),
                "h3_normalized_by_log2_cells": float(
                    metric["normalized_entropy"][-1]
                ),
                "effective_cells_2_to_h3": float(metric["effective_cells"][-1]),
                "pmf_sum_error": float(metric["pmf_sum_error"]),
                "region_frames": record["num_frames"],
                "rfft_bins_stored": record["rfft_bins"],
                "full_frequency_cells_per_frame": record[
                    "full_frequency_cells_per_frame"
                ],
                "region_cells": total_cells,
                "time_start_s": record["time_start_s"],
                "time_end_s": record["time_end_s"],
                "frequency_region": "full two-sided DFT grid",
                "sample_rate_hz": record["sample_rate"],
                "n_fft": record["n_fft"],
                "hop_size": record["hop_size"],
                "best_fixed_length": record["best_fixed_length"],
                "joint_optimal_fixed_length": record[
                    "joint_optimal_fixed_length"
                ],
                "joint_optimal_fixed_h3_bits": record[
                    "joint_optimal_fixed_h3_bits"
                ],
            })
    return rows


def _joint_renyi3_snr_summary(records):
    """Aggregate controlled Monte Carlo joint H3 values without cherry-picking."""
    method_order = (
        "posterior_blend", "framewise_selection",
        "best_fixed_support", "constrained_schedule",
    )
    snrs = sorted({float(record["snr_db"]) for record in records})
    rows = []
    for snr_db in snrs:
        selected = [
            record for record in records
            if float(record["snr_db"]) == snr_db
        ]
        values_by_method = {
            method: np.asarray([
                record["metrics"][method]["entropy_bits"][-1]
                for record in selected
            ], dtype=float)
            for method in method_order
        }
        normalized_by_method = {
            method: np.asarray([
                record["metrics"][method]["normalized_entropy"][-1]
                for record in selected
            ], dtype=float)
            for method in method_order
        }
        excess_by_method = {method: [] for method in method_order}
        for record in selected:
            run_values = {
                method: float(record["metrics"][method]["entropy_bits"][-1])
                for method in method_order
            }
            minimum = min(run_values.values())
            for method in method_order:
                excess_by_method[method].append(run_values[method] - minimum)
        excess_by_method = {
            method: np.asarray(values, dtype=float)
            for method, values in excess_by_method.items()
        }
        medians = {
            method: float(np.median(values))
            for method, values in values_by_method.items()
        }
        ranked = sorted(method_order, key=lambda method: medians[method])
        win_counts = {method: 0 for method in method_order}
        for record in selected:
            run_values = {
                method: float(record["metrics"][method]["entropy_bits"][-1])
                for method in method_order
            }
            minimum = min(run_values.values())
            winners = [
                method for method, value in run_values.items()
                if np.isclose(value, minimum, rtol=0.0, atol=2e-12)
            ]
            for method in winners:
                win_counts[method] += 1
        for method in method_order:
            values = values_by_method[method]
            normalized = normalized_by_method[method]
            rows.append({
                "snr_db": snr_db,
                "method": method,
                "runs": len(selected),
                "median_h3_bits": medians[method],
                "q1_h3_bits": float(np.percentile(values, 25)),
                "q3_h3_bits": float(np.percentile(values, 75)),
                "median_h3_normalized_by_log2_cells": float(
                    np.median(normalized)
                ),
                "median_excess_above_run_min_bits": float(
                    np.median(excess_by_method[method])
                ),
                "q1_excess_above_run_min_bits": float(
                    np.percentile(excess_by_method[method], 25)
                ),
                "q3_excess_above_run_min_bits": float(
                    np.percentile(excess_by_method[method], 75)
                ),
                "rank_by_median_h3": ranked.index(method) + 1,
                "finite_family_run_wins": win_counts[method],
            })
    return rows


def _plot_joint_renyi3_audit(records, snr_summary, figure_dir):
    """Plot raw Baraniuk-style information first and normalized values second."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    method_order = (
        "posterior_blend", "framewise_selection",
        "best_fixed_support", "constrained_schedule",
    )
    method_labels = {
        "posterior_blend": "Posterior-blended U-STFT",
        "constrained_schedule": "Constrained hindsight schedule",
        "framewise_selection": "Framewise selection",
        "best_fixed_support": "Final-horizon tracker-loss-optimal fixed STFT",
    }
    colors = {
        "posterior_blend": "#0072B2",
        "constrained_schedule": "#D55E00",
        "framewise_selection": "#009E73",
        "best_fixed_support": "#CC79A7",
    }
    controlled = next(record for record in records
                      if record["slug"] == "tone-impulse-conflict")

    with plt.rc_context({
        "font.size": 7.5,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.7,
    }):
        fig = plt.figure(figsize=(7.16, 5.8))
        grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
        snr_axis = fig.add_subplot(grid[0, 0])
        excess_axis = fig.add_subplot(grid[0, 1])
        prefix_axis = fig.add_subplot(grid[1, 0])
        raw_axis = fig.add_subplot(grid[1, 1])

        condition_labels = [
            f"{float(value):g} dB"
            for value in sorted({row["snr_db"] for row in snr_summary})
        ] + ["Clean"]
        condition_x = np.arange(len(condition_labels), dtype=float)
        clean_values = {
            method: float(controlled["metrics"][method]["entropy_bits"][-1])
            for method in method_order
        }
        clean_minimum = min(clean_values.values())
        for method in method_order:
            selected = sorted(
                [row for row in snr_summary if row["method"] == method],
                key=lambda row: float(row["snr_db"]),
            )
            medians = np.asarray(
                [row["median_h3_bits"] for row in selected]
                + [clean_values[method]]
            )
            q1 = np.asarray(
                [row["q1_h3_bits"] for row in selected]
                + [clean_values[method]]
            )
            q3 = np.asarray(
                [row["q3_h3_bits"] for row in selected]
                + [clean_values[method]]
            )
            snr_axis.plot(
                condition_x, medians, color=colors[method], marker="o",
                ms=3.2, lw=1.7,
                ls="--" if method == "constrained_schedule" else "-",
                label=method_labels[method],
            )
            snr_axis.fill_between(
                condition_x, q1, q3, color=colors[method], alpha=0.11,
                linewidth=0,
            )
            median_excess = np.asarray(
                [row["median_excess_above_run_min_bits"] for row in selected]
                + [clean_values[method] - clean_minimum]
            )
            q1_excess = np.asarray(
                [row["q1_excess_above_run_min_bits"] for row in selected]
                + [clean_values[method] - clean_minimum]
            )
            q3_excess = np.asarray(
                [row["q3_excess_above_run_min_bits"] for row in selected]
                + [clean_values[method] - clean_minimum]
            )
            excess_axis.plot(
                condition_x, median_excess, color=colors[method], marker="o",
                ms=3.2, lw=1.7,
                ls="--" if method == "constrained_schedule" else "-",
            )
            excess_axis.fill_between(
                condition_x, q1_excess, q3_excess,
                color=colors[method], alpha=0.11, linewidth=0,
            )
        snr_axis.set_xlabel("Condition")
        snr_axis.set_ylabel(r"Full-region $H_3(P)$ (bits)")
        snr_axis.set_title("(a) Absolute information: median and IQR")
        snr_axis.grid(alpha=0.20)
        snr_axis.set_xticks(condition_x, condition_labels)

        excess_axis.axhline(0.0, color="black", lw=0.65)
        excess_axis.set_xlabel("Condition")
        excess_axis.set_ylabel("Excess above per-run minimum (bits)")
        excess_axis.set_title("(b) Four-method finite-family excess")
        excess_axis.grid(alpha=0.20)
        excess_axis.set_xticks(condition_x, condition_labels)
        excess_axis.set_ylim(bottom=-0.005)

        for method in method_order:
            prefix_axis.plot(
                controlled["prefix_time_s"],
                controlled["metrics"][method]["entropy_bits"],
                color=colors[method], lw=1.9,
                ls="--" if method == "constrained_schedule" else "-",
                label=method_labels[method],
            )
        prefix_axis.set_xlabel("Time (s)")
        prefix_axis.set_ylabel(r"Prefix $H_3(P_t)$ (bits)")
        prefix_axis.set_title(
            "(c) Clean controlled record: one PMF per prefix"
        )
        prefix_axis.grid(alpha=0.20)
        prefix_axis.set_xlim(
            controlled["prefix_time_s"][0], controlled["prefix_time_s"][-1]
        )

        x = np.arange(len(records), dtype=float)
        width = 0.19
        for offset, method in enumerate(method_order):
            positions = x + (offset - 1.5) * width
            raw_values = [
                record["metrics"][method]["entropy_bits"][-1]
                for record in records
            ]
            raw_axis.bar(
                positions, raw_values, width=width, color=colors[method],
                label=method_labels[method],
            )

        tick_labels = [
            f"{record['short_label']}\nfixed $L={record['best_fixed_length']}$"
            for record in records
        ]
        raw_axis.set_xticks(x, tick_labels)
        raw_axis.grid(axis="y", alpha=0.20)
        raw_axis.set_ylabel(r"Full-region $H_3(P)$ (bits)")
        raw_axis.set_title("(d) Complete-record values (within case)")

        fig.suptitle(
            "Joint spectrogram Rényi-3 concentration (lower is more concentrated)",
            y=0.995, fontsize=10.2,
        )
        handles, legend_labels = snr_axis.get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.935), frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.865))
        save_figure(fig,
            figure_dir / "joint_renyi3_concentration.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

        # A compact controlled-only replacement for the historical invalid
        # double-cumulative plot.  This uses the same single-prefix PMFs as (a).
        fig, axis = plt.subplots(figsize=(7.16, 3.25))
        for method in method_order:
            axis.plot(
                controlled["prefix_time_s"],
                controlled["metrics"][method]["entropy_bits"],
                color=colors[method], lw=2.0,
                ls="--" if method == "constrained_schedule" else "-",
                label=method_labels[method],
            )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(r"Joint Rényi information $H_3(P_t)$ (bits)")
        axis.set_title(
            "Controlled full-frequency spectrogram PMF: one calculation per prefix"
        )
        axis.grid(alpha=0.20)
        axis.legend(loc="upper left", ncol=2, framealpha=0.9)
        axis.set_xlim(
            controlled["prefix_time_s"][0], controlled["prefix_time_s"][-1]
        )
        fig.tight_layout()
        save_figure(fig,
            figure_dir / "joint_renyi3_controlled_prefix.png",
            dpi=300, bbox_inches="tight",
        )
        save_figure(fig,
            figure_dir.parent / "conflict_renyi_cumulative.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)


def _plot_controlled_snr_joint_renyi3(snr_summary, path):
    """Plot the controlled Monte Carlo 2-D H3 sweep without other records."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    order = (
        "posterior_blend", "framewise_selection",
        "best_fixed_support", "constrained_schedule",
    )
    labels = {
        "posterior_blend": "U-STFT (causal)",
        "framewise_selection": "Framewise selection (causal)",
        "best_fixed_support": "Fixed causal STFT (hindsight support)",
        "constrained_schedule": "Constrained (noncausal comparator)",
    }
    colors = {
        "posterior_blend": "#0072B2",
        "framewise_selection": "#009E73",
        "best_fixed_support": "#CC79A7",
        "constrained_schedule": "#D55E00",
    }
    snrs = sorted({float(row["snr_db"]) for row in snr_summary})
    if snrs != [-10.0, 0.0, 10.0, 20.0]:
        raise AssertionError("controlled SNR figure requires the full four-SNR sweep")

    with plt.rc_context({
        "font.size": 6.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 5.7,
    }):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.75), sharex=True)
        for method in order:
            selected = sorted(
                (row for row in snr_summary if row["method"] == method),
                key=lambda row: float(row["snr_db"]),
            )
            if len(selected) != len(snrs):
                raise AssertionError(f"incomplete SNR summary for {method}")
            medians = np.asarray([row["median_h3_bits"] for row in selected])
            q1 = np.asarray([row["q1_h3_bits"] for row in selected])
            q3 = np.asarray([row["q3_h3_bits"] for row in selected])
            axes[0].plot(
                snrs, medians, color=colors[method], marker="o", ms=3.5,
                lw=1.8, ls="--" if method == "constrained_schedule" else "-",
                label=labels[method],
            )
            axes[0].fill_between(
                snrs, q1, q3, color=colors[method], alpha=0.12, linewidth=0,
            )

            excess = np.asarray([
                row["median_excess_above_run_min_bits"] for row in selected
            ])
            excess_q1 = np.asarray([
                row["q1_excess_above_run_min_bits"] for row in selected
            ])
            excess_q3 = np.asarray([
                row["q3_excess_above_run_min_bits"] for row in selected
            ])
            axes[1].plot(
                snrs, excess, color=colors[method], marker="o", ms=3.5,
                lw=1.8, ls="--" if method == "constrained_schedule" else "-",
                label=labels[method],
            )
            axes[1].fill_between(
                snrs, excess_q1, excess_q3,
                color=colors[method], alpha=0.12, linewidth=0,
            )
            if method == "posterior_blend":
                for row, x_value, y_value in zip(selected, snrs, excess):
                    axes[1].annotate(
                        f"{int(row['finite_family_run_wins'])}/{int(row['runs'])}",
                        xy=(x_value, y_value), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        color=colors[method], fontsize=5.8,
                    )

        axes[0].set_ylabel(r"Joint $H_3(P)$ (bits)")
        axes[0].set_title("(a) Median and interquartile range")
        axes[0].legend(loc="best", ncol=1, framealpha=0.9)
        axes[1].axhline(0.0, color="black", lw=0.65)
        axes[1].set_ylabel("Excess above run\nminimum (bits)")
        axes[1].set_title("(b) Median excess; blue labels are wins/runs")
        axes[1].set_ylim(bottom=-0.004)
        for axis in axes:
            axis.set_xticks(snrs)
            axis.grid(alpha=0.20)
        axes[-1].set_xlabel("SNR (dB)")
        fig.tight_layout()
        save_figure(fig, Path(path), dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_spectrogram_evidence(clean_case, posterior_case, figure_dir,
                               n_fft=1024, hop_size=32, fs=4096):
    """Write publication-scale STFT comparisons from repository diagnostics.

    Every panel within a figure shares one physical time/frequency coordinate
    system and one global power reference.  No display-only renormalization is
    applied per method or per candidate.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    floor_db = -70.0
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # ------------------------------------------------------------------
    # Direct method comparison: full-band behavior plus a close-tone crop.
    # ------------------------------------------------------------------
    fixed_length = int(clean_case["lengths"][clean_case["fixed_index"]])
    method_specs = [
        np.asarray(clean_case[name][0])
        for name in ("trellis", "constrained", "greedy", "best_fixed")
    ]
    method_db = _spectrogram_db(method_specs, floor_db=floor_db)
    times = (np.arange(method_specs[0].shape[0]) + 1) * hop_size / fs
    method_labels = (
        "U-STFT",
        "Best constrained schedule",
        "Framewise selector",
        f"Best fixed ($L={fixed_length}$)",
    )

    with plt.rc_context({
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
    }):
        fig, axes = plt.subplots(
            2, 4, figsize=(7.16, 4.35), sharex="row", sharey="row",
        )
        image = None
        for column, (values, label) in enumerate(zip(method_db, method_labels)):
            axes[0, column].set_title(label, pad=3.0)
            for row, axis in enumerate(axes[:, column]):
                image = axis.pcolormesh(
                    times, frequencies, values.T, shading="auto",
                    cmap="magma", vmin=floor_db, vmax=0.0, rasterized=True,
                )
                axis.text(
                    0.018, 0.96,
                    f"({chr(ord('a') + 4 * row + column)})",
                    transform=axis.transAxes, ha="left", va="top",
                    color="white", fontweight="bold",
                    bbox={"facecolor": "black", "alpha": 0.42,
                          "edgecolor": "none", "pad": 1.5},
                )
                if row == 0:
                    axis.set_xlim(0.0, times[-1])
                    axis.set_ylim(0.0, 2000.0)
                    axis.axvspan(0.0, 1.0, color="white", alpha=0.055)
                    for event_time in (1.0, 2.0, 3.0):
                        axis.axvline(
                            event_time, color="white", lw=0.65,
                            linestyle="--", alpha=0.7,
                        )
                else:
                    axis.set_xlim(1.10, times[-1])
                    axis.set_ylim(450.0, 580.0)
                    for tone in (500.0, 532.0):
                        axis.axhline(
                            tone, color="#56B4E9", lw=0.65,
                            linestyle=":", alpha=0.9,
                        )
                    for click_time in (2.0, 3.0):
                        axis.axvline(
                            click_time, color="white", lw=0.55,
                            linestyle="--", alpha=0.55,
                        )
        axes[0, 0].set_ylabel("Full band\nFrequency (Hz)")
        axes[1, 0].set_ylabel("Close-tone detail\nFrequency (Hz)")
        for axis in axes[-1]:
            axis.set_xlabel("Time (s)")
        fig.subplots_adjust(
            left=0.10, right=0.90, bottom=0.105, top=0.90,
            hspace=0.18, wspace=0.10,
        )
        color_axis = fig.add_axes((0.92, 0.105, 0.018, 0.795))
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label("Power (dB relative to shared maximum)")
        save_figure(fig,
            figure_dir / "spectrogram_method_comparison.png", dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Candidate-bank context: the mixture and every dyadic resolution.
    # This uses the exact candidate spectra returned by the repository run.
    # ------------------------------------------------------------------
    candidate_specs = [
        np.asarray(clean_case["candidate_specs"])[state]
        for state in range(len(clean_case["lengths"]))
    ]
    bank_specs = [method_specs[0], *candidate_specs]
    bank_db = _spectrogram_db(bank_specs, floor_db=floor_db)
    bank_labels = [
        "U-STFT posterior blend (causal)",
        *[f"Fixed causal kernel $L={int(length)}$"
          for length in clean_case["lengths"]],
    ]
    with plt.rc_context({
        "font.size": 6.2,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 5.9,
    }):
        fig, axes = plt.subplots(
            len(bank_db), 1, figsize=(3.5, 8.35), sharex=True, sharey=True,
        )
        image = None
        for panel, (axis, values, label) in enumerate(
                zip(axes, bank_db, bank_labels)):
            image = axis.pcolormesh(
                times, frequencies, values.T, shading="auto", cmap="magma",
                vmin=floor_db, vmax=0.0, rasterized=True,
            )
            axis.set_xlim(0.0, times[-1])
            axis.set_ylim(0.0, 2000.0)
            for event_time in (1.0, 2.0, 3.0):
                axis.axvline(
                    event_time, color="white", lw=0.5,
                    linestyle="--", alpha=0.6,
                )
            axis.set_title(f"({chr(ord('a') + panel)}) {label}", pad=2.5)
        axes[-1].set_xlabel("Time (s)")
        fig.supylabel("Frequency (Hz)", x=0.015, fontsize=6.8)
        fig.subplots_adjust(
            left=0.18, right=0.97, bottom=0.08, top=0.98,
            hspace=0.23,
        )
        color_axis = fig.add_axes((0.21, 0.027, 0.70, 0.009))
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.set_label("Power (dB; shared reference)")
        save_figure(fig,
            figure_dir / "spectrogram_dyadic_bank.png", dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # One coherent causal narrative at 0 dB: output, uncertainty, schedule.
    # ------------------------------------------------------------------
    posterior = np.asarray(posterior_case["pmf"], dtype=float)
    lengths = np.asarray(posterior_case["lengths"], dtype=int)
    noisy_spec = np.asarray(posterior_case["trellis"][0])
    noisy_db = _spectrogram_db([noisy_spec], floor_db=floor_db)[0]
    noisy_times = (np.arange(len(posterior)) + 1) * hop_size / fs
    expected_level = posterior @ np.arange(len(lengths), dtype=float)
    constrained_level = np.asarray(
        posterior_case["hmm_comparator_path"], dtype=float
    )

    with plt.rc_context({
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
    }):
        fig, axes = plt.subplots(
            3, 1, figsize=(7.16, 5.65), sharex=True,
            gridspec_kw={"height_ratios": (2.0, 1.35, 0.95)},
        )
        spectrogram_image = axes[0].pcolormesh(
            noisy_times, frequencies, noisy_db.T, shading="auto",
            cmap="magma", vmin=floor_db, vmax=0.0, rasterized=True,
        )
        axes[0].set_ylim(0.0, 2000.0)
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_title("(a) Dyadic soft-trellis analysis at 0 dB", loc="left")
        for event_time in (1.0, 2.0, 3.0):
            axes[0].axvline(
                event_time, color="white", lw=0.6,
                linestyle="--", alpha=0.7,
            )
        spec_colorbar = fig.colorbar(
            spectrogram_image, ax=axes[0], pad=0.012, aspect=18,
        )
        spec_colorbar.set_label("Power (dB)")

        posterior_image = axes[1].imshow(
            posterior.T, origin="lower", aspect="auto", interpolation="nearest",
            extent=(noisy_times[0], noisy_times[-1], -0.5, len(lengths) - 0.5),
            vmin=0.0, vmax=1.0, cmap="Blues", rasterized=True,
        )
        axes[1].plot(
            noisy_times, expected_level, color="#D55E00", lw=1.15,
            label="Posterior mean level",
        )
        axes[1].step(
            noisy_times, constrained_level, where="post", color="black", lw=0.9,
            linestyle="--", alpha=0.8,
            label="Best constrained hindsight schedule",
        )
        axes[1].set_yticks(np.arange(len(lengths)), [str(L) for L in lengths])
        axes[1].set_ylabel("Window length")
        axes[1].set_title("(b) Complete state posterior", loc="left")
        axes[1].legend(loc="lower right", ncol=2)
        pmf_colorbar = fig.colorbar(
            posterior_image, ax=axes[1], pad=0.012, aspect=12,
        )
        pmf_colorbar.set_label("Probability")

        gap = np.asarray(posterior_case["hmm_curves"]["average_gap"])
        bound = np.asarray(posterior_case["hmm_curves"]["path_code_bound"])
        axes[2].plot(
            noisy_times, gap, color="#0072B2", lw=1.6,
            label="Gap to constrained hindsight schedule",
        )
        axes[2].plot(
            noisy_times, bound, color="#CC79A7", lw=1.35,
            linestyle="--", label="Exact path-code bound",
        )
        axes[2].axhline(0.0, color="black", lw=0.6)
        axes[2].set_ylabel("Nats/frame")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("(c) Average HMM mix-loss gap", loc="left")
        axes[2].legend(loc="upper right", ncol=2)
        axes[2].grid(alpha=0.2)
        axes[2].set_xlim(0.0, noisy_times[-1])

        fig.subplots_adjust(
            left=0.10, right=0.92, bottom=0.085, top=0.96,
            hspace=0.30,
        )
        save_figure(fig,
            figure_dir / "spectrogram_posterior_regret.png", dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _plot_field_recording_spectrograms(real_cases, figure_dir):
    """Compare all paper estimators on two unlabelled field recordings."""
    if not real_cases:
        return

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ordered = sorted(
        real_cases.items(),
        key=lambda item: ("humpback" not in item[0].lower(), item[0]),
    )
    ordered = ordered[:2]
    floor_db = -75.0

    with plt.rc_context({
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
    }):
        fig, axes = plt.subplots(
            len(ordered), 4, figsize=(7.16, 4.65), squeeze=False,
            sharex="row", sharey="row",
        )
        image = None
        for row, (recording, payload) in enumerate(ordered):
            case = payload["case"]
            fs = payload["fs"]
            n_fft = payload["n_fft"]
            hop_size = payload["hop_size"]
            specs = [
                np.asarray(case[name][0])
                for name in ("trellis", "constrained", "greedy", "best_fixed")
            ]
            values_db = _spectrogram_db(specs, floor_db=floor_db)
            times = (np.arange(specs[0].shape[0]) + 1) * hop_size / fs
            frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
            name_lower = recording.lower()
            if "humpback" in name_lower:
                recording_label = "Humpback whale"
                upper_frequency = min(4000.0, fs / 2.0)
            elif "shrimp" in name_lower:
                recording_label = "Snapping shrimp"
                upper_frequency = fs / 2.0
            else:
                recording_label = recording.replace("_", " ")
                upper_frequency = fs / 2.0
            fixed_length = int(case["lengths"][case["fixed_index"]])

            for column, (axis, values) in enumerate(zip(axes[row], values_db)):
                image = axis.pcolormesh(
                    times, frequencies, values.T, shading="auto",
                    cmap="magma", vmin=floor_db, vmax=0.0, rasterized=True,
                )
                axis.set_xlim(0.0, times[-1])
                axis.set_ylim(0.0, upper_frequency)
                method_names = ("U-STFT", "Hindsight schedule", "Framewise selection", "Best fixed window")
                axis.set_title(f"{recording_label}: {method_names[column]}")
                axis.set_ylabel("Frequency (Hz)")
                axis.text(
                    0.015, 0.96,
                    f"({chr(ord('a') + 4 * row + column)})",
                    transform=axis.transAxes, ha="left", va="top",
                    color="white", fontweight="bold",
                    bbox={"facecolor": "black", "alpha": 0.40,
                          "edgecolor": "none", "pad": 1.3},
                )
                if column == 3:
                    axis.text(
                        0.98, 0.96, f"$L={fixed_length}$",
                        transform=axis.transAxes, ha="right", va="top",
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.40,
                              "edgecolor": "none", "pad": 1.3},
                    )
                if row == len(ordered) - 1:
                    axis.set_xlabel("Time (s)")
            axes[row, 0].set_ylabel(f"{recording_label}\nFrequency (Hz)")


        fig.subplots_adjust(
            left=0.105, right=0.90, bottom=0.09, top=0.94,
            hspace=0.20, wspace=0.10,
        )
        color_axis = fig.add_axes((0.92, 0.09, 0.018, 0.85))
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label("Power (dB relative to each row maximum)")
        save_figure(fig,
            Path(figure_dir) / "spectrogram_field_recordings.png", dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_paper_figure(fig, path, dpi=350):
    """Export paired figures using the shared chapter renderer."""
    return save_figure(fig, path)


def _plot_window_length_tradeoff(case, metadata, path, n_fft=1024,
                                  hop_size=32):
    """Show the fixed-window time--frequency trade-off on known targets."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fs = float(metadata["fs"])
    lengths = np.asarray(case["lengths"], dtype=int)
    spectra = np.asarray(case["candidate_specs"])
    if spectra.shape[0] != len(lengths):
        raise AssertionError("candidate spectra and lengths disagree")
    short_index, long_index = 0, len(lengths) - 1
    frame_time = (np.arange(spectra.shape[1]) + 1) * hop_size / fs
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    stable = (frame_time >= 1.25) & (frame_time <= 1.80)
    tone_power = np.mean(np.abs(spectra[:, stable]) ** 2, axis=1)
    tone_reference = max(
        float(np.max(tone_power[short_index])),
        float(np.max(tone_power[long_index])),
    )
    tone_db = 10.0 * np.log10(
        np.maximum(tone_power, tone_reference * 1e-8) / tone_reference
    )

    # Isolate the temporal point-spread function so tonal leakage cannot
    # obscure the short-versus-long comparison.  Curves are centered on their
    # own causal response peak; only their widths are being compared here.
    impulse = np.zeros(int(fs), dtype=float)
    impulse[len(impulse) // 2] = 1.0
    _, _, impulse_frames = _repository_stft._padded_signal(
        impulse, n_fft, hop_size
    )
    impulse_db = {}
    impulse_time_ms = {}
    for index in (short_index, long_index):
        state_path = np.full(impulse_frames, index, dtype=int)
        analysis, _ = blend_guided_stft(
            impulse, _one_hot_blend_info(lengths, state_path),
            n_fft, hop_size,
        )
        energy = np.sum(np.abs(analysis) ** 2, axis=1)
        peak = int(np.argmax(energy))
        reference = float(energy[peak])
        impulse_db[index] = 10.0 * np.log10(
            np.maximum(energy, reference * 1e-8) / reference
        )
        impulse_time_ms[index] = (
            1000.0 * (np.arange(impulse_frames) - peak) * hop_size / fs
        )

    styles = (
        (short_index, "#0072B2", "-", f"short $L={lengths[short_index]}$"),
        (long_index, "#D55E00", "--", f"long $L={lengths[long_index]}$"),
    )
    with plt.rc_context({
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(3.1, 2.05))
        for index, color, linestyle, label in styles:
            axes[0].plot(
                frequencies, tone_db[index], color=color, ls=linestyle,
                lw=1.45, label=label,
            )
            axes[1].plot(
                impulse_time_ms[index], impulse_db[index],
                color=color, ls=linestyle, lw=1.45,
            )
        f1, f2 = metadata["tone_frequencies"]
        axes[0].axvline(f1, color="0.55", lw=0.55, ls=":")
        axes[0].axvline(f2, color="0.55", lw=0.55, ls=":")
        axes[0].set_xlim(f1 - 55.0, f2 + 55.0)
        axes[0].set_ylim(-35.0, 1.0)
        axes[0].set_title("(a) Close tones")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Relative power (dB)")
        axes[1].axvline(0.0, color="0.55", lw=0.55, ls=":")
        axes[1].set_xlim(-180.0, 180.0)
        axes[1].set_ylim(-45.0, 1.0)
        axes[1].set_title("(b) One-sample impulse")
        axes[1].set_xlabel("Peak-relative time (ms)")
        for axis in axes:
            axis.grid(alpha=0.18)
        fig.legend(
            loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
            frameon=False,
        )
        fig.subplots_adjust(
            left=0.13, right=0.985, bottom=0.19, top=0.80, wspace=0.32,
        )
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_window_length_trellis(path, num_frames=6):
    """Draw the ideal linear stay/grow/reset window-length trellis."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"stay": "0.25", "grow": "#0072B2", "reset": "#D55E00"}
    with plt.rc_context({
        "font.size": 7.0,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
    }):
        fig, axis = plt.subplots(1, 1, figsize=(3.1, 2.15))
        for frame in range(1, num_frames):
            for run_length in range(1, frame + 1):
                axis.annotate(
                    "", xy=(frame + 1, run_length),
                    xytext=(frame, run_length),
                    arrowprops={
                        "arrowstyle": "-|>", "mutation_scale": 6.0,
                        "lw": 0.7, "color": colors["stay"],
                    },
                    zorder=1,
                )
                axis.annotate(
                    "", xy=(frame + 1, run_length + 1),
                    xytext=(frame, run_length),
                    arrowprops={
                        "arrowstyle": "-|>", "mutation_scale": 6.0,
                        "lw": 0.7, "ls": ":", "color": colors["grow"],
                    },
                    zorder=1,
                )
                if run_length > 1:
                    axis.annotate(
                        "", xy=(frame + 1, 1),
                        xytext=(frame, run_length),
                        arrowprops={
                            "arrowstyle": "-|>", "mutation_scale": 6.0,
                            "lw": 0.65, "ls": "--",
                            "color": colors["reset"], "alpha": 0.85,
                        },
                        zorder=1,
                    )
        for frame in range(1, num_frames + 1):
            for run_length in range(1, frame + 1):
                axis.scatter(
                    frame, run_length, s=55.0, facecolor="white",
                    edgecolor="0.35", linewidth=0.7, zorder=3,
                )
                axis.text(
                    frame, run_length, str(run_length), ha="center",
                    va="center", fontsize=5.8, color="0.2", zorder=4,
                )
        axis.set_xlim(0.65, num_frames + 0.35)
        axis.set_ylim(0.55, num_frames + 0.45)
        axis.set_xticks(np.arange(1, num_frames + 1))
        axis.set_yticks(
            np.arange(1, num_frames + 1),
            [r"$H$"] + [rf"${value}H$" for value in range(2, num_frames + 1)],
        )
        axis.set_xlabel("Frame $m$")
        axis.set_ylabel("Window length")
        axis.spines[["top", "right"]].set_visible(False)
        legend_handles = (
            Line2D([], [], color=colors["stay"], lw=0.8, label="stay"),
            Line2D([], [], color=colors["grow"], lw=0.8, ls=":", label="grow"),
            Line2D([], [], color=colors["reset"], lw=0.8, ls="--", label="reset"),
        )
        axis.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False,
            handlelength=2.0, columnspacing=1.0,
        )
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.88)
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_case_spectrogram_grid(case, fs, n_fft, hop_size, title,
                                upper_frequency, path, floor_db=-70.0):
    """Compare all four transforms on one shared power scale."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names = ("constrained", "best_fixed", "trellis", "greedy")
    spectra = [np.asarray(case[name][0]) for name in names]
    values_db = _spectrogram_db(spectra, floor_db=floor_db)
    times = (np.arange(spectra[0].shape[0]) + 1) * hop_size / fs
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    fixed_length = int(case["lengths"][case["fixed_index"]])
    labels = (
        "Hindsight schedule", "Best fixed window", "U-STFT", "Framewise selection",
    )

    with plt.rc_context({
        "font.size": 7.5,
        "axes.titlesize": 8.3,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
    }):
        fig, axes = plt.subplots(
            1, len(names), figsize=(7.16, 2.75), sharex=True, sharey=True,
        )
        image = None
        for panel, (axis, values, label) in enumerate(
                zip(axes, values_db, labels)):
            image = axis.pcolormesh(
                times, frequencies, values.T, shading="auto", cmap="magma",
                vmin=floor_db, vmax=0.0, rasterized=True,
            )
            axis.set_xlim(0.0, times[-1])
            axis.set_ylim(0.0, min(float(upper_frequency), fs / 2.0))
            axis.set_title(f"({chr(ord('a') + panel)}) {label}", pad=2.5)
            axis.set_xlabel("Time (s)")
        axes[0].set_ylabel("Frequency (Hz)")
        fig.subplots_adjust(
            left=0.075, right=0.992, bottom=0.245, top=0.91,
            wspace=0.08,
        )
        color_axis = fig.add_axes((0.26, 0.075, 0.48, 0.025))
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.set_label("Power (dB relative to shared maximum)")
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_chirp_spectrogram_teaching(case, fs, n_fft, hop_size, path,
                                     floor_db=-70.0):
    """Shared-scale maps for all methods and the complete fixed-window bank."""
    from benchmarks.plot_suites import power_panels
    from benchmarks.figure_export import save_spec
    _plot_case_spectrogram_grid(case, fs, n_fft, hop_size, "Hyperbolic chirp",
                                fs / 2.0, path, floor_db=floor_db)
    spectra = {f"fixed_{index}": spectrum for index, spectrum in enumerate(case["candidate_specs"])}
    metadata = {"dataset": "hyperbolic-chirp", "trials": "single run", "hop": hop_size,
                "source": "Existing chirp candidate spectra from the released transform"}
    save_spec({"metadata": metadata, "panels": power_panels(spectra, fs, hop_size, floor_db)},
              Path(path).with_name(Path(path).stem + "_fixed_bank.png"))


def _plot_chirp_score_posterior(case, fs, hop_size, path):
    """Show how the two normalized score terms become posterior weights."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    lengths = np.asarray(case["lengths"], dtype=int)
    h_time = np.asarray(case["candidate_h_time"], dtype=float)
    h_frequency = np.asarray(case["candidate_h_spec"], dtype=float)
    total_score = np.asarray(case["candidate_costs"], dtype=float)
    posterior = np.asarray(case["pmf"], dtype=float)
    expected_shape = (posterior.shape[0], len(lengths))
    for name, values in (
            ("time score", h_time),
            ("frequency score", h_frequency),
            ("total score", total_score)):
        if values.shape != expected_shape:
            raise AssertionError(f"{name} and posterior shapes disagree")
    finite = np.isfinite(total_score)
    if not np.allclose(
            total_score[finite], (h_time + h_frequency)[finite],
            rtol=2e-12, atol=2e-12):
        raise AssertionError("candidate score is not the sum of its Rényi terms")
    if np.any(posterior < -2e-12) or not np.allclose(
            np.sum(posterior, axis=1), 1.0, rtol=2e-12, atol=2e-12):
        raise AssertionError("posterior weights must form a PMF at every frame")

    times = (
        (np.arange(posterior.shape[0]) + 1)
        * float(hop_size) / float(fs)
    )
    levels = np.arange(len(lengths), dtype=float)
    expected_level = posterior @ levels
    masked_time = np.ma.masked_where(~finite, h_time)
    masked_frequency = np.ma.masked_where(~finite, h_frequency)
    masked_total = np.ma.masked_where(~finite, total_score)
    finite_total = total_score[finite]
    score_min, score_max = np.percentile(finite_total, (1.0, 99.0))
    if not score_max > score_min:
        score_min, score_max = float(np.min(finite_total)), float(np.max(finite_total))
        score_max = score_min + max(1e-9, abs(score_min) * 1e-6)

    with plt.rc_context({
        "font.size": 7.3,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.5,
    }):
        fig, axes = plt.subplots(
            2, 2, figsize=(7.16, 3.75), sharex=True, sharey=True,
        )
        axes = axes.ravel()
        extent = (times[0], times[-1], -0.5, len(lengths) - 0.5)
        viridis = plt.get_cmap("viridis").copy()
        viridis.set_bad("0.88")
        magma = plt.get_cmap("magma").copy()
        magma.set_bad("0.88")
        images = (
            axes[0].imshow(
                masked_time.T, origin="lower", aspect="auto",
                interpolation="nearest", extent=extent, cmap=viridis,
                vmin=0.0, vmax=1.0, rasterized=True,
            ),
            axes[1].imshow(
                masked_frequency.T, origin="lower", aspect="auto",
                interpolation="nearest", extent=extent, cmap=viridis,
                vmin=0.0, vmax=1.0, rasterized=True,
            ),
            axes[2].imshow(
                masked_total.T, origin="lower", aspect="auto",
                interpolation="nearest", extent=extent, cmap=magma,
                vmin=score_min, vmax=score_max, rasterized=True,
            ),
            axes[3].imshow(
                posterior.T, origin="lower", aspect="auto",
                interpolation="nearest", extent=extent, cmap="Blues",
                vmin=0.0, vmax=1.0, rasterized=True,
            ),
        )
        titles = (
            r"(a) Normalized time $H_3$",
            r"(b) Normalized frequency $H_3$",
            r"(c) Total score $\ell=H_{3,\mathrm{time}}+H_{3,\mathrm{frequency}}$",
            "(d) Posterior and geometric-mean support",
        )
        colorbar_labels = (
            "Normalized entropy", "Normalized entropy", "Score", "Probability",
        )
        for axis, image, title, colorbar_label in zip(
                axes, images, titles, colorbar_labels):
            axis.set_yticks(levels, lengths)
            axis.set_title(title, pad=2.5)
            colorbar = fig.colorbar(image, ax=axis, pad=0.012, fraction=0.035)
            colorbar.set_label(colorbar_label)
        axes[3].plot(
            times, expected_level, color="#D55E00", lw=1.7,
            label="Posterior mean support",
        )
        axes[3].legend(loc="best", framealpha=0.9)
        for axis in axes[2:]:
            axis.set_xlabel("Time (s)")
        axes[0].set_ylabel("Window (samples)")
        axes[2].set_ylabel("Window (samples)")
        axes[3].set_xlim(times[0], times[-1])
        fig.subplots_adjust(
            left=0.075, right=0.965, bottom=0.12, top=0.96,
            hspace=0.34, wspace=0.19,
        )
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_case_trajectory_comparison(record, path, posterior_only=False):
    """Plot one record's posterior, optionally without comparator trajectories."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.asarray(record["time"], dtype=float)
    posterior = np.asarray(record["posterior"], dtype=float)
    lengths = np.asarray(record["candidate_lengths"], dtype=int)
    levels = np.arange(len(lengths), dtype=float)
    posterior_mean_level = posterior @ levels
    posterior_mean_length = lengths[0] * 2.0 ** posterior_mean_level

    if posterior_only:
        with plt.rc_context({
            "font.size": 6.5,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
        }):
            fig, axis = plt.subplots(1, 1, figsize=(3.5, 2.55))
            image = axis.imshow(
                posterior.T, origin="lower", aspect="auto",
                interpolation="nearest",
                extent=(t[0], t[-1], -0.5, len(lengths) - 0.5),
                cmap="Blues", vmin=0.0, vmax=1.0, rasterized=True,
            )
            axis.plot(
                t, posterior_mean_level, color="#D55E00", lw=1.8,
                label="Posterior geometric-mean support",
            )
            axis.set_yticks(levels, lengths)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Window (samples)")
            axis.set_title("Posterior over dyadic supports")
            axis.set_xlim(t[0], t[-1])
            axis.legend(loc="best", framealpha=0.9)
            colorbar = fig.colorbar(
                image, ax=axis, pad=0.015, fraction=0.045,
            )
            colorbar.set_label("Probability")
            fig.tight_layout()
            save_figure(fig, Path(path), dpi=300, bbox_inches="tight")
            plt.close(fig)
        return

    fixed_length = int(lengths[record["fixed_path"][0]])
    paths = (
        (posterior_mean_length, "Posterior geometric-mean support",
         "#0072B2", "-", 2.0),
        (lengths[record["framewise_path"]], "Framewise selection (causal)",
         "#009E73", "-", 1.35),
        (lengths[record["fixed_path"]],
         "Fixed causal STFT (hindsight)",
         "#CC79A7", ":", 1.8),
        (lengths[record["comparator_path"]], "Constrained (noncausal)",
         "#D55E00", "--", 1.45),
    )

    with plt.rc_context({
        "font.size": 6.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 5.7,
    }):
        fig, axes = plt.subplots(
            2, 1, figsize=(3.5, 4.35), sharex=True,
            gridspec_kw={"height_ratios": (1.35, 1.0)},
        )
        image = axes[0].imshow(
            posterior.T, origin="lower", aspect="auto",
            interpolation="nearest",
            extent=(t[0], t[-1], -0.5, len(lengths) - 0.5),
            cmap="Blues", vmin=0.0, vmax=1.0, rasterized=True,
        )
        axes[0].plot(
            t, posterior_mean_level, color="#0072B2", lw=1.8,
            label="Posterior geometric-mean support",
        )
        axes[0].step(
            t, record["comparator_path"], where="post", color="#D55E00",
            lw=1.25, ls="--", label="Noncausal comparator",
        )
        axes[0].set_yticks(levels, lengths)
        axes[0].set_ylabel("Window (samples)")
        axes[0].set_title("(a) Posterior over dyadic lengths")
        axes[0].legend(loc="best", ncol=1, framealpha=0.9)
        colorbar = fig.colorbar(image, ax=axes[0], pad=0.015, fraction=0.045)
        colorbar.set_label("Probability")

        for values, label, color, linestyle, linewidth in paths:
            axes[1].step(
                t, values, where="post", label=label, color=color,
                ls=linestyle, lw=linewidth,
            )
        axes[1].set_yscale("log", base=2)
        axes[1].set_yticks(lengths, lengths)
        axes[1].set_ylim(lengths[0] / np.sqrt(2.0), lengths[-1] * np.sqrt(2.0))
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Window (samples)")
        axes[1].set_title("(b) Support trajectories")
        axes[1].grid(alpha=0.20)
        axes[1].legend(loc="best", ncol=2, framealpha=0.9)
        axes[1].text(
            0.985, 0.04, f"fixed $L={fixed_length}$",
            transform=axes[1].transAxes, ha="right", va="bottom",
            fontsize=5.5, color="#AA3377",
        )
        axes[1].set_xlim(t[0], t[-1])
        fig.tight_layout()
        save_figure(fig, Path(path), dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_case_joint_renyi3_comparison(record, path):
    """Plot one record's genuine joint time-frequency Rényi-3 comparison."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    order = (
        "posterior_blend", "framewise_selection",
        "best_fixed_support", "constrained_schedule",
    )
    labels = {
        "posterior_blend": "U-STFT (causal)",
        "framewise_selection": "Framewise selection (causal)",
        "best_fixed_support": "Fixed causal STFT (hindsight support)",
        "constrained_schedule": "Constrained (noncausal comparator)",
    }
    colors = {
        "posterior_blend": "#0072B2",
        "framewise_selection": "#009E73",
        "best_fixed_support": "#CC79A7",
        "constrained_schedule": "#D55E00",
    }
    if set(record["metrics"]) != set(order):
        raise AssertionError("joint Rényi record does not contain the four baselines")

    with plt.rc_context({
        "font.size": 6.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 5.7,
    }):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.85))
        for method in order:
            axes[0].plot(
                record["prefix_time_s"],
                record["metrics"][method]["entropy_bits"],
                color=colors[method], lw=1.85,
                ls="--" if method == "constrained_schedule" else "-",
                label=labels[method],
            )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel(r"Joint $H_3(P_t)$ (bits)")
        axes[0].set_title("(a) One time--frequency PMF per prefix")
        axes[0].grid(alpha=0.20)
        axes[0].set_xlim(
            record["prefix_time_s"][0], record["prefix_time_s"][-1]
        )
        axes[0].legend(loc="best", framealpha=0.9)

        final_values = np.asarray([
            record["metrics"][method]["entropy_bits"][-1]
            for method in order
        ])
        bars = axes[1].bar(
            np.arange(len(order)), final_values,
            color=[colors[method] for method in order], width=0.72,
        )
        joint_fixed = float(record["joint_optimal_fixed_h3_bits"])
        axes[1].axhline(
            joint_fixed, color="0.25", lw=1.1, ls=":",
            label=("2-D-optimal fixed reference "
                   f"($L={record['joint_optimal_fixed_length']}$)"),
        )
        inset = max(0.08, 0.018 * float(np.max(final_values)))
        for bar in bars:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() - inset,
                f"{bar.get_height():.3f}",
                ha="center", va="top", fontsize=7.0,
                color="white", fontweight="bold",
            )
        axes[1].set_xticks(
            np.arange(len(order)),
            ("U-STFT", "Framewise", "Fixed STFT", "Constrained"),
        )
        axes[1].set_ylabel(r"Complete-region $H_3(P)$ (bits)")
        axes[1].set_title("(b) Common grid (lower is more concentrated)")
        axes[1].grid(axis="y", alpha=0.20)
        axes[1].legend(loc="lower right", fontsize=6.6, framealpha=0.9)
        axes[1].set_ylim(top=max(final_values) * 1.035)
        fig.tight_layout()
        save_figure(fig, Path(path), dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_chirp_joint_renyi3_comparison(record, path):
    """Plot the chirp's joint Rényi-3 prefixes as a wide metric figure."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    candidate_lengths = np.asarray(record["candidate_lengths"], dtype=int)
    candidate_metrics = record.get("candidate_metrics")
    if candidate_metrics is None or len(candidate_metrics) != len(candidate_lengths):
        raise AssertionError("joint record lacks the complete fixed-window bank")
    fixed_index = int(record["best_fixed_index"])
    fixed_prefix = np.asarray(candidate_metrics[fixed_index]["entropy_bits"])
    if not np.allclose(
            fixed_prefix,
            record["metrics"]["best_fixed_support"]["entropy_bits"],
            rtol=2e-13, atol=2e-13):
        raise AssertionError("highlighted fixed curve differs from the tracker baseline")
    prefix_time = np.asarray(record["prefix_time_s"], dtype=float)
    for metric in candidate_metrics:
        if len(metric["entropy_bits"]) != len(prefix_time):
            raise AssertionError("fixed-window and adaptive prefixes are misaligned")

    with plt.rc_context({
        "font.size": 7.5,
        "axes.titlesize": 8.3,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "legend.fontsize": 6.5,
    }):
        fig, axis = plt.subplots(1, 1, figsize=(7.16, 2.65))
        for metric in candidate_metrics:
            axis.plot(
                prefix_time, metric["entropy_bits"], color="0.62",
                lw=0.75, alpha=0.28, zorder=1,
            )
        axis.plot(
            prefix_time,
            record["metrics"]["posterior_blend"]["entropy_bits"],
            color="#0072B2", lw=2.0, label="U-STFT", zorder=4,
        )
        axis.plot(
            prefix_time,
            record["metrics"]["framewise_selection"]["entropy_bits"],
            color="#009E73", lw=1.6,
            label="Framewise score minimum", zorder=4,
        )
        axis.plot(
            prefix_time,
            record["metrics"]["constrained_schedule"]["entropy_bits"],
            color="#D55E00", lw=1.6, ls="--",
            label="Best window schedule", zorder=4,
        )
        axis.plot(
            prefix_time, fixed_prefix, color="#CC79A7", lw=2.0, ls=":",
            label=("Fixed STFT selected by tracking loss ("
                   f"$L={candidate_lengths[fixed_index]}$)"),
            zorder=5,
        )
        axis.plot(
            [], [], color="0.62", lw=0.9, alpha=0.55,
            label="Fixed-window bank",
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(r"Joint $H_3(P_r)$ (bits)")
        axis.set_title("Joint time--frequency Rényi-3 on common prefixes")
        axis.set_xlim(prefix_time[0], prefix_time[-1])
        axis.grid(alpha=0.20)
        axis.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3,
            framealpha=0.92,
        )
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.88)
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_case_loss_regret_comparison(record, path, plain_scientific=False):
    """Compare scaled losses and then show the exact path-code certificate."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.asarray(record["time"], dtype=float)
    warmup = int(record["warmup"])
    common_time = t[warmup:]
    if len(common_time) == 0:
        raise AssertionError("loss comparison has no common valid region")
    emission = np.asarray(record["emission_loss"], dtype=float)[warmup:]
    costs = np.asarray(record["candidate_costs"], dtype=float)[warmup:]
    transition = np.asarray(record["transition_matrix"], dtype=float)
    frame_index = np.arange(len(common_time))

    def path_curve(indices):
        values = emission[frame_index, np.asarray(indices)[warmup:]]
        if not np.all(np.isfinite(values)):
            raise AssertionError("common-region path contains an invalid support")
        return np.cumsum(values)

    framewise_curve = path_curve(record["framewise_path"])
    fixed_curve = path_curve(record["fixed_path"])
    if not np.allclose(framewise_curve, 0.0, rtol=0.0, atol=2e-12):
        raise AssertionError("framewise greedy must minimize every plotted emission")
    mix = np.asarray(record["mix_curve"], dtype=float)
    mix_baseline = mix[warmup - 1] if warmup else 0.0
    mix_common = mix[warmup:] - mix_baseline
    denominator = np.arange(1, len(common_time) + 1, dtype=float)

    # Recompute the best admissible schedule for every common-region prefix.
    # The conditional start prior is the transform's predicted distribution
    # immediately before the first common-region observation.  The DP minimizes
    # emission only; the selected path's negative log probability is its exact
    # code length under that conditional prior and the released transitions.
    if warmup:
        start_prior = np.asarray(record["posterior"])[warmup - 1] @ transition
    else:
        start_prior = np.zeros(emission.shape[1])
        start_prior[0] = 1.0
    dp = np.full_like(emission, np.inf)
    back = np.full(emission.shape, -1, dtype=int)
    start_valid = (start_prior > 0.0) & np.isfinite(emission[0])
    dp[0, start_valid] = emission[0, start_valid]
    for frame in range(1, len(emission)):
        for state in range(emission.shape[1]):
            if not np.isfinite(emission[frame, state]):
                continue
            predecessors = np.flatnonzero(transition[:, state] > 0.0)
            predecessors = predecessors[np.isfinite(dp[frame - 1, predecessors])]
            if len(predecessors) == 0:
                continue
            previous = predecessors[np.argmin(dp[frame - 1, predecessors])]
            dp[frame, state] = dp[frame - 1, previous] + emission[frame, state]
            back[frame, state] = previous
    prefix_schedule_curve = np.min(dp, axis=1)
    suffix_final_path = np.zeros(len(emission), dtype=int)
    suffix_final_path[-1] = int(np.argmin(dp[-1]))
    for frame in range(len(emission) - 1, 0, -1):
        suffix_final_path[frame - 1] = back[frame, suffix_final_path[frame]]
    stored_suffix_path = np.asarray(record["comparator_path"])[warmup:]
    if not np.array_equal(suffix_final_path, stored_suffix_path):
        differing = int(np.count_nonzero(suffix_final_path != stored_suffix_path))
        raise AssertionError(
            "conditional and full-horizon comparator paths differ in "
            f"{differing} frames"
        )
    prefix_code_curve = np.zeros(len(emission))
    for end_frame in range(len(emission)):
        prefix_path = np.zeros(end_frame + 1, dtype=int)
        prefix_path[-1] = int(np.argmin(dp[end_frame]))
        for frame in range(end_frame, 0, -1):
            prefix_path[frame - 1] = back[frame, prefix_path[frame]]
        initial_probability = start_prior[prefix_path[0]]
        if initial_probability <= 0.0:
            raise AssertionError("prefix comparator starts outside conditional prior")
        prefix_code_curve[end_frame] = -np.log(initial_probability)
        if end_frame:
            probabilities = transition[prefix_path[:-1], prefix_path[1:]]
            prefix_code_curve[end_frame] -= np.sum(np.log(probabilities))

    # Restore the frame-dependent offset removed by the Gibbs normalization:
    # b_m = eta_m min_j ell_mj.  It changes the visual origin but cancels from
    # every gap and from the theorem.
    eta = float(record["beta"]) / (np.std(costs, axis=1) + 1e-6)
    offset_curve = np.cumsum(eta * np.min(costs, axis=1))

    tolerance = 2e-9 * max(1.0, float(np.max(prefix_code_curve + prefix_schedule_curve)))
    if np.any(prefix_schedule_curve < -tolerance):
        raise AssertionError("standardized schedule loss must be nonnegative")
    if np.any(prefix_schedule_curve > fixed_curve + tolerance):
        raise AssertionError("best prefix schedule exceeds the fixed path")
    if np.any(prefix_schedule_curve > mix_common + tolerance):
        raise AssertionError("mixture loss falls below the best schedule emission")
    if np.any(mix_common > prefix_schedule_curve + prefix_code_curve + tolerance):
        raise AssertionError("conditional suffix path-code certificate failed")
    fixed_state = int(record["fixed_path"][0])
    if start_prior[fixed_state] <= 0.0:
        raise AssertionError("fixed support has zero conditional start probability")
    if transition[fixed_state, fixed_state] <= 0.0:
        raise AssertionError("fixed support has zero stay probability")
    fixed_code_curve = (
        -np.log(start_prior[fixed_state])
        - np.arange(len(emission), dtype=float)
        * np.log(transition[fixed_state, fixed_state])
    )
    if np.any(mix_common > fixed_curve + fixed_code_curve + tolerance):
        raise AssertionError("fixed-support path-code certificate failed")
    fixed_bank_curves = np.cumsum(emission, axis=0)
    if not np.all(np.isfinite(fixed_bank_curves)):
        raise AssertionError("fixed-window bank is invalid on the common suffix")
    if not np.allclose(
            fixed_curve, fixed_bank_curves[:, fixed_state],
            rtol=2e-12, atol=2e-12):
        raise AssertionError("highlighted fixed curve differs from its bank member")

    with plt.rc_context({
        "font.size": 7.3,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.2,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
        if plain_scientific:
            mixture_label = "U-STFT mixture"
            framewise_label = "Framewise score minimum"
            fixed_label = (
                "Fixed STFT selected by tracking loss "
                f"($L={record['candidate_lengths'][record['fixed_path'][0]]}$)"
            )
            schedule_label = "Best window schedule"
            regret_label = "Regret to best window schedule"
            bound_label = "Path-code bound"
        else:
            mixture_label = r"U-STFT causal mixture $B_{a:t}+\Lambda_{a:t}$"
            framewise_label = r"Framewise selection $B_{a:t}$"
            fixed_label = (
                "Fixed causal STFT; hindsight support "
                f"($L={record['candidate_lengths'][record['fixed_path'][0]]}$)"
            )
            schedule_label = (
                r"Noncausal comparator $B_{a:t}+D^*_{a:t}$"
            )
            regret_label = (
                r"Certified schedule regret "
                r"$(\Lambda_{a:t}-D^*_{a:t})/n$"
            )
            bound_label = r"Conditional path-code bound $C^*_{a:t}/n$"
        top_offset = 0.0 if plain_scientific else offset_curve
        if plain_scientific:
            for state in range(fixed_bank_curves.shape[1]):
                axes[0].plot(
                    common_time, fixed_bank_curves[:, state], color="0.62",
                    lw=0.75, alpha=0.28, zorder=1,
                )
        axes[0].plot(
            common_time, top_offset + mix_common,
            color="#0072B2", lw=2.0,
            label=mixture_label, zorder=4,
        )
        axes[0].plot(
            common_time, top_offset + framewise_curve,
            color="#009E73", lw=1.5,
            label=framewise_label, zorder=4,
        )
        axes[0].plot(
            common_time, top_offset + fixed_curve,
            color="#CC79A7", lw=1.7, ls=":",
            label=fixed_label, zorder=5,
        )
        axes[0].plot(
            common_time, top_offset + prefix_schedule_curve,
            color="#D55E00", lw=1.6, ls="--",
            label=schedule_label, zorder=4,
        )
        if plain_scientific:
            axes[0].plot(
                [], [], color="0.62", lw=0.9, alpha=0.55,
                label="Fixed-window bank",
            )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel(
            "Cumulative excess (nats)" if plain_scientific
            else "Cumulative scaled loss (nats)"
        )
        axes[0].set_title(
            "(a) Loss above framewise minimum"
            if plain_scientific else "(a) Scaled emissions and mixture loss"
        )
        axes[0].grid(alpha=0.20)
        axes[0].set_xlim(common_time[0], common_time[-1])

        regret = (mix_common - prefix_schedule_curve) / denominator
        bound = prefix_code_curve / denominator
        axes[1].fill_between(
            common_time, regret, bound, where=bound >= regret,
            color="#CC79A7", alpha=0.10, linewidth=0,
        )
        axes[1].plot(
            common_time, regret, color="#0072B2", lw=2.05,
            label=regret_label,
        )
        axes[1].plot(
            common_time, bound, color="#CC79A7", lw=1.6, ls="--",
            label=bound_label,
        )
        if not plain_scientific:
            axes[1].plot(
                common_time, (mix_common - fixed_curve) / denominator,
                color="#CC79A7", lw=1.05, ls=":",
                label="Mixture $-$ fixed (descriptive)",
            )
            axes[1].plot(
                common_time, (mix_common - framewise_curve) / denominator,
                color="#009E73", lw=1.05, ls=":",
                label="Mixture $-$ framewise minimum (descriptive)",
            )
        axes[1].axhline(0.0, color="black", lw=0.65)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel(
            "Nats/frame" if plain_scientific else "Average gap (nats/frame)"
        )
        axes[1].set_title(
            "(b) Regret to the best window schedule"
            if plain_scientific
            else r"(b) Certified regret and gaps ($n=t-a+1$)"
        )
        axes[1].grid(alpha=0.20)
        axes[1].set_xlim(common_time[0], common_time[-1])
        handles = []
        labels = []
        for axis in axes:
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            handles.extend(axis_handles)
            labels.extend(axis_labels)
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
            ncol=4 if plain_scientific else 3, framealpha=0.92,
            columnspacing=1.0, handlelength=2.2,
        )
        fig.subplots_adjust(
            left=0.085, right=0.985, bottom=0.19,
            top=0.73 if plain_scientific else 0.69, wspace=0.28,
        )
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_chirp_diagnostics(case, record, joint_record, fs, hop_size, path):
    """Collect the chirp's inference, regret, and 2-D entropy in one plate."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lengths = np.asarray(case["lengths"], dtype=int)
    h_time = np.asarray(case["candidate_h_time"], dtype=float)
    h_frequency = np.asarray(case["candidate_h_spec"], dtype=float)
    total_score = np.asarray(case["candidate_costs"], dtype=float)
    posterior = np.asarray(case["pmf"], dtype=float)
    expected_shape = (posterior.shape[0], len(lengths))
    for name, values in (
            ("time score", h_time),
            ("frequency score", h_frequency),
            ("total score", total_score)):
        if values.shape != expected_shape:
            raise AssertionError(f"{name} and posterior shapes disagree")
    finite = np.isfinite(total_score)
    if not np.allclose(
            total_score[finite], (h_time + h_frequency)[finite],
            rtol=2e-12, atol=2e-12):
        raise AssertionError("candidate score is not the sum of its Rényi terms")
    if np.any(posterior < -2e-12) or not np.allclose(
            np.sum(posterior, axis=1), 1.0, rtol=2e-12, atol=2e-12):
        raise AssertionError("posterior weights must form a PMF at every frame")

    score_time = (
        (np.arange(posterior.shape[0]) + 1)
        * float(hop_size) / float(fs)
    )
    levels = np.arange(len(lengths), dtype=float)
    expected_level = posterior @ levels
    extent = (
        score_time[0], score_time[-1], -0.5, len(lengths) - 0.5,
    )
    finite_total = total_score[finite]
    score_min, score_max = np.percentile(finite_total, (1.0, 99.0))
    if not score_max > score_min:
        score_min = float(np.min(finite_total))
        score_max = score_min + max(1e-9, abs(score_min) * 1e-6)

    # Recompute the exact common-region comparator and conditional path code.
    # These are the same quantities plotted by _plot_case_loss_regret_comparison.
    t = np.asarray(record["time"], dtype=float)
    warmup = int(record["warmup"])
    common_time = t[warmup:]
    emission = np.asarray(record["emission_loss"], dtype=float)[warmup:]
    transition = np.asarray(record["transition_matrix"], dtype=float)
    frame_index = np.arange(len(common_time))
    if len(common_time) == 0:
        raise AssertionError("chirp diagnostic has no common valid region")

    def path_curve(indices):
        values = emission[frame_index, np.asarray(indices)[warmup:]]
        if not np.all(np.isfinite(values)):
            raise AssertionError("common-region path contains an invalid support")
        return np.cumsum(values)

    framewise_curve = path_curve(record["framewise_path"])
    fixed_curve = path_curve(record["fixed_path"])
    mix = np.asarray(record["mix_curve"], dtype=float)
    mix_baseline = mix[warmup - 1] if warmup else 0.0
    mix_common = mix[warmup:] - mix_baseline
    if warmup:
        start_prior = np.asarray(record["posterior"])[warmup - 1] @ transition
    else:
        start_prior = np.zeros(emission.shape[1])
        start_prior[0] = 1.0
    dp = np.full_like(emission, np.inf)
    back = np.full(emission.shape, -1, dtype=int)
    start_valid = (start_prior > 0.0) & np.isfinite(emission[0])
    dp[0, start_valid] = emission[0, start_valid]
    for frame in range(1, len(emission)):
        for state in range(emission.shape[1]):
            if not np.isfinite(emission[frame, state]):
                continue
            predecessors = np.flatnonzero(transition[:, state] > 0.0)
            predecessors = predecessors[np.isfinite(dp[frame - 1, predecessors])]
            if len(predecessors) == 0:
                continue
            previous = predecessors[np.argmin(dp[frame - 1, predecessors])]
            dp[frame, state] = dp[frame - 1, previous] + emission[frame, state]
            back[frame, state] = previous
    schedule_curve = np.min(dp, axis=1)
    code_curve = np.zeros(len(emission))
    for end_frame in range(len(emission)):
        prefix_path = np.zeros(end_frame + 1, dtype=int)
        prefix_path[-1] = int(np.argmin(dp[end_frame]))
        for frame in range(end_frame, 0, -1):
            prefix_path[frame - 1] = back[frame, prefix_path[frame]]
        initial_probability = start_prior[prefix_path[0]]
        if initial_probability <= 0.0:
            raise AssertionError("prefix comparator starts outside conditional prior")
        code_curve[end_frame] = -np.log(initial_probability)
        if end_frame:
            probabilities = transition[prefix_path[:-1], prefix_path[1:]]
            code_curve[end_frame] -= np.sum(np.log(probabilities))
    fixed_bank_curves = np.cumsum(emission, axis=0)
    denominator = np.arange(1, len(common_time) + 1, dtype=float)
    regret = (mix_common - schedule_curve) / denominator
    bound = code_curve / denominator
    tolerance = 2e-9 * max(1.0, float(np.max(code_curve + schedule_curve)))
    if np.any(regret > bound + tolerance):
        raise AssertionError("conditional suffix path-code certificate failed")

    candidate_lengths = np.asarray(joint_record["candidate_lengths"], dtype=int)
    candidate_metrics = joint_record.get("candidate_metrics")
    if candidate_metrics is None or len(candidate_metrics) != len(candidate_lengths):
        raise AssertionError("joint record lacks the complete fixed-window bank")
    fixed_index = int(joint_record["best_fixed_index"])
    prefix_time = np.asarray(joint_record["prefix_time_s"], dtype=float)

    colors = {
        "ustft": "#0072B2",
        "framewise": "#009E73",
        "fixed": "#CC79A7",
        "schedule": "#D55E00",
        "bank": "0.62",
    }
    with plt.rc_context({
        "font.size": 5.8,
        "axes.titlesize": 6.5,
        "axes.labelsize": 6.2,
        "xtick.labelsize": 5.3,
        "ytick.labelsize": 5.3,
        "legend.fontsize": 5.2,
    }):
        fig = plt.figure(figsize=(3.1, 5.65))
        grid = fig.add_gridspec(
            4, 2, height_ratios=(1.0, 1.0, 1.05, 1.12),
            hspace=0.64, wspace=0.50,
        )
        top_axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
        ]
        bottom_axes = [
            fig.add_subplot(grid[2, 0]),
            fig.add_subplot(grid[2, 1]),
            fig.add_subplot(grid[3, :]),
        ]
        viridis = plt.get_cmap("viridis").copy()
        viridis.set_bad("0.88")
        magma = plt.get_cmap("magma").copy()
        magma.set_bad("0.88")
        heatmaps = (
            top_axes[0].imshow(
                np.ma.masked_where(~finite, h_time).T,
                origin="lower", aspect="auto", interpolation="nearest",
                extent=extent, cmap=viridis, vmin=0.0, vmax=1.0,
                rasterized=True,
            ),
            top_axes[1].imshow(
                np.ma.masked_where(~finite, h_frequency).T,
                origin="lower", aspect="auto", interpolation="nearest",
                extent=extent, cmap=viridis, vmin=0.0, vmax=1.0,
                rasterized=True,
            ),
            top_axes[2].imshow(
                np.ma.masked_where(~finite, total_score).T,
                origin="lower", aspect="auto", interpolation="nearest",
                extent=extent, cmap=magma, vmin=score_min, vmax=score_max,
                rasterized=True,
            ),
            top_axes[3].imshow(
                posterior.T, origin="lower", aspect="auto",
                interpolation="nearest", extent=extent, cmap="Blues",
                vmin=0.0, vmax=1.0, rasterized=True,
            ),
        )
        top_titles = (
            r"(a) Time $H_3$", r"(b) Frequency $H_3$",
            r"(c) Sum $\ell$", "(d) Posterior",
        )
        colorbar_titles = ("norm.", "norm.", "score", "prob.")
        for index, (axis, image, axis_title, colorbar_title) in enumerate(zip(
                top_axes, heatmaps, top_titles, colorbar_titles)):
            axis.set_title(axis_title, pad=2.0)
            axis.set_yticks(levels, lengths)
            axis.set_xlim(score_time[0], score_time[-1])
            if index % 2:
                axis.tick_params(labelleft=False)
            else:
                axis.set_ylabel("Window")
            colorbar = fig.colorbar(image, ax=axis, pad=0.012, fraction=0.055)
            colorbar.ax.set_title(colorbar_title, fontsize=4.8, pad=1.0)
            colorbar.ax.tick_params(labelsize=4.8)
        top_axes[3].plot(
            score_time, expected_level, color="#D55E00", lw=1.05,
        )

        loss_axis, regret_axis, joint_axis = bottom_axes
        for state in range(fixed_bank_curves.shape[1]):
            loss_axis.plot(
                common_time, fixed_bank_curves[:, state], color=colors["bank"],
                lw=0.45, alpha=0.28, zorder=1,
            )
        loss_axis.plot(common_time, mix_common, color=colors["ustft"], lw=1.25)
        loss_axis.plot(
            common_time, framewise_curve, color=colors["framewise"], lw=0.95,
        )
        loss_axis.plot(
            common_time, fixed_curve, color=colors["fixed"], lw=1.05, ls=":",
        )
        loss_axis.plot(
            common_time, schedule_curve, color=colors["schedule"],
            lw=1.0, ls="--",
        )
        loss_axis.set_title("(e) Cumulative excess")
        loss_axis.set_ylabel("Nats")

        regret_axis.fill_between(
            common_time, regret, bound, where=bound >= regret,
            color=colors["fixed"], alpha=0.10, linewidth=0,
        )
        regret_axis.plot(common_time, regret, color=colors["ustft"], lw=1.25)
        regret_axis.plot(
            common_time, bound, color=colors["fixed"], lw=1.0, ls="--",
        )
        regret_axis.axhline(0.0, color="black", lw=0.55)
        regret_axis.set_title("(f) Schedule regret")
        regret_axis.set_ylabel("Nats/frame")

        for metric in candidate_metrics:
            joint_axis.plot(
                prefix_time, metric["entropy_bits"], color=colors["bank"],
                lw=0.45, alpha=0.28, zorder=1,
            )
        joint_axis.plot(
            prefix_time,
            joint_record["metrics"]["posterior_blend"]["entropy_bits"],
            color=colors["ustft"], lw=1.25, zorder=4,
        )
        joint_axis.plot(
            prefix_time,
            joint_record["metrics"]["framewise_selection"]["entropy_bits"],
            color=colors["framewise"], lw=0.95, zorder=4,
        )
        joint_axis.plot(
            prefix_time,
            joint_record["metrics"]["constrained_schedule"]["entropy_bits"],
            color=colors["schedule"], lw=1.0, ls="--", zorder=4,
        )
        joint_axis.plot(
            prefix_time, candidate_metrics[fixed_index]["entropy_bits"],
            color=colors["fixed"], lw=1.05, ls=":", zorder=5,
        )
        joint_axis.set_title(r"(g) Joint 2-D $H_3$ (bits)")

        for axis in bottom_axes:
            axis.set_xlabel("Time (s)")
            axis.grid(alpha=0.18)
        loss_axis.set_xlim(common_time[0], common_time[-1])
        regret_axis.set_xlim(common_time[0], common_time[-1])
        joint_axis.set_xlim(prefix_time[0], prefix_time[-1])
        legend_handles = (
            Line2D([], [], color=colors["ustft"], lw=1.25, label="U-STFT"),
            Line2D([], [], color=colors["framewise"], lw=0.95,
                   label="Framewise minimum"),
            Line2D([], [], color=colors["fixed"], lw=1.05, ls=":",
                   label=f"Fixed STFT ($L={candidate_lengths[fixed_index]}$)"),
            Line2D([], [], color=colors["schedule"], lw=1.0, ls="--",
                   label="Best window schedule"),
            Line2D([], [], color=colors["bank"], lw=0.8, alpha=0.55,
                   label="Fixed-window bank"),
            Line2D([], [], color=colors["fixed"], lw=1.0, ls="--",
                   label="Path-code bound"),
        )
        fig.legend(
            handles=legend_handles, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.008), framealpha=0.92,
            columnspacing=0.9, handlelength=1.8,
        )
        fig.subplots_adjust(
            left=0.15, right=0.96, bottom=0.125, top=0.975,
        )
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_controlled_walkthrough(x, metadata, case, path, n_fft=1024,
                                 hop_size=32):
    """Explain the controlled example with four compact, time-aligned views."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fs = float(metadata["fs"])
    sample_time = np.arange(len(x), dtype=float) / fs
    frame_time = (np.arange(len(case["pmf"])) + 1) * hop_size / fs
    lengths = np.asarray(case["lengths"], dtype=int)
    costs = np.ma.masked_invalid(np.asarray(case["candidate_costs"], dtype=float).T)
    posterior = np.asarray(case["pmf"], dtype=float).T
    spectrogram = np.asarray(case["trellis"][0])
    spectrogram_db = _spectrogram_db([spectrogram], floor_db=-70.0)[0]
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    finite_costs = costs.compressed()
    score_vmin, score_vmax = np.percentile(finite_costs, (2.0, 98.0))

    with plt.rc_context({
        "font.size": 7.3,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
    }):
        fig, axes = plt.subplots(
            2, 2, figsize=(7.16, 3.85), sharex=True,
        )
        axes = axes.ravel()
        axes[0].plot(sample_time, x, color="0.15", lw=0.75)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("(a) Signal and known events")
        axes[0].grid(alpha=0.16)

        score_image = axes[1].imshow(
            costs, origin="lower", aspect="auto", interpolation="nearest",
            extent=(frame_time[0], frame_time[-1], -0.5, len(lengths) - 0.5),
            cmap="viridis_r", vmin=score_vmin, vmax=score_vmax,
            rasterized=True,
        )
        axes[1].set_yticks(np.arange(len(lengths)), lengths)
        axes[1].set_ylabel("Window (samples)")
        axes[1].set_title("(b) Candidate score (lower is preferred)")
        colorbar = fig.colorbar(score_image, ax=axes[1], pad=0.012,
                                fraction=0.04)
        colorbar.set_label("Score")

        posterior_image = axes[2].imshow(
            posterior, origin="lower", aspect="auto", interpolation="nearest",
            extent=(frame_time[0], frame_time[-1], -0.5, len(lengths) - 0.5),
            cmap="Blues", vmin=0.0, vmax=1.0, rasterized=True,
        )
        posterior_mean_level = np.asarray(case["pmf"]) @ np.arange(len(lengths))
        axes[2].plot(frame_time, posterior_mean_level, color="#D55E00", lw=1.2)
        axes[2].set_yticks(np.arange(len(lengths)), lengths)
        axes[2].set_ylabel("Window (samples)")
        axes[2].set_title("(c) Causal posterior over supports")
        colorbar = fig.colorbar(
            posterior_image, ax=axes[2], pad=0.012, fraction=0.04,
        )
        colorbar.set_label("Probability")

        spectral_image = axes[3].pcolormesh(
            frame_time, frequencies, spectrogram_db.T, shading="auto",
            cmap="magma", vmin=-70.0, vmax=0.0, rasterized=True,
        )
        axes[3].set_ylim(0.0, 2000.0)
        axes[3].set_ylabel("Frequency (Hz)")
        axes[3].set_title("(d) Posterior-blended U-STFT")
        colorbar = fig.colorbar(
            spectral_image, ax=axes[3], pad=0.012, fraction=0.04,
        )
        colorbar.set_label("Power (shared dB)")

        for axis in axes:
            axis.axvline(1.0, color="white" if axis is not axes[0] else "0.4",
                         lw=0.7, ls="--", alpha=0.8)
            for click_time in metadata["click_times"]:
                axis.axvline(
                    click_time,
                    color="white" if axis is not axes[0] else "#D55E00",
                    lw=0.7, ls=":", alpha=0.8,
                )
            axis.set_xlim(0.0, frame_time[-1])
        for axis in axes[2:]:
            axis.set_xlabel("Time (s)")
        fig.subplots_adjust(
            left=0.075, right=0.965, bottom=0.13, top=0.95,
            hspace=0.34, wspace=0.24,
        )
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_controlled_evidence(x, metadata, case, path, n_fft=1024,
                              hop_size=32):
    """Combine the controlled inference chain and method comparison."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fs = float(metadata["fs"])
    sample_time = np.arange(len(x), dtype=float) / fs
    frame_time = (np.arange(len(case["pmf"])) + 1) * hop_size / fs
    lengths = np.asarray(case["lengths"], dtype=int)
    costs = np.ma.masked_invalid(
        np.asarray(case["candidate_costs"], dtype=float).T
    )
    posterior = np.asarray(case["pmf"], dtype=float).T
    finite_costs = costs.compressed()
    score_vmin, score_vmax = np.percentile(finite_costs, (2.0, 98.0))
    names = ("constrained", "best_fixed", "trellis", "greedy")
    spectra = [np.asarray(case[name][0]) for name in names]
    spectra_db = _spectrogram_db(spectra, floor_db=-70.0)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    fixed_length = int(lengths[case["fixed_index"]])
    posterior_mean_level = np.asarray(case["pmf"]) @ np.arange(len(lengths))

    with plt.rc_context({
        "font.size": 5.9,
        "axes.titlesize": 6.6,
        "axes.labelsize": 6.3,
        "xtick.labelsize": 5.4,
        "ytick.labelsize": 5.4,
    }):
        fig, axes = plt.subplots(4, 2, figsize=(3.1, 4.65), sharex=True)
        axes[0, 0].plot(sample_time, x, color="0.15", lw=0.65)
        axes[0, 0].set_title("(a) Signal and events")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(alpha=0.15)

        score_image = axes[0, 1].imshow(
            costs, origin="lower", aspect="auto", interpolation="nearest",
            extent=(frame_time[0], frame_time[-1], -0.5, len(lengths) - 0.5),
            cmap="viridis_r", vmin=score_vmin, vmax=score_vmax,
            rasterized=True,
        )
        axes[0, 1].set_title("(b) Candidate score")
        axes[0, 1].set_yticks(np.arange(len(lengths)), lengths)
        axes[0, 1].set_ylabel("Window")
        score_bar = fig.colorbar(
            score_image, ax=axes[0, 1], pad=0.012, fraction=0.043,
        )
        score_bar.set_label("score", fontsize=4.8, labelpad=1.0)
        score_bar.ax.tick_params(labelsize=4.8)

        posterior_image = axes[1, 0].imshow(
            posterior, origin="lower", aspect="auto", interpolation="nearest",
            extent=(frame_time[0], frame_time[-1], -0.5, len(lengths) - 0.5),
            cmap="Blues", vmin=0.0, vmax=1.0, rasterized=True,
        )
        axes[1, 0].plot(
            frame_time, posterior_mean_level, color="#D55E00", lw=1.15,
        )
        axes[1, 0].set_title("(c) Posterior support")
        axes[1, 0].set_yticks(np.arange(len(lengths)), lengths)
        axes[1, 0].set_ylabel("Window")
        posterior_bar = fig.colorbar(
            posterior_image, ax=axes[1, 0], pad=0.012, fraction=0.043,
        )
        posterior_bar.set_label("prob.", fontsize=4.8, labelpad=1.0)
        posterior_bar.ax.tick_params(labelsize=4.8)

        method_titles = (
            "Hindsight schedule", "Best fixed window", "U-STFT", "Framewise selection",
        )
        spectral_axes = (axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 0])
        spectral_image = None
        for axis, values, axis_title in zip(
                spectral_axes, spectra_db, method_titles):
            spectral_image = axis.pcolormesh(
                frame_time, frequencies, values.T, shading="auto",
                cmap="magma", vmin=-70.0, vmax=0.0, rasterized=True,
            )
            axis.set_ylim(0.0, 2000.0)
            axis.set_title(axis_title)
        axes[1, 1].tick_params(labelleft=False)
        axes[2, 0].set_ylabel("Frequency (Hz)")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 1].set_xlabel("Time (s)")

        for axis in axes.ravel():
            axis.axvline(
                1.0, color="white" if axis is not axes[0, 0] else "0.4",
                lw=0.6, ls="--", alpha=0.8,
            )
            for click_time in metadata["click_times"]:
                axis.axvline(
                    click_time,
                    color="white" if axis is not axes[0, 0] else "#D55E00",
                    lw=0.6, ls=":", alpha=0.8,
                )
            axis.set_xlim(0.0, frame_time[-1])
        fig.subplots_adjust(
            left=0.15, right=0.96, bottom=0.125, top=0.97,
            hspace=0.48, wspace=0.52,
        )
        color_axis = fig.add_axes((0.25, 0.035, 0.50, 0.012))
        colorbar = fig.colorbar(
            spectral_image, cax=color_axis, orientation="horizontal",
        )
        colorbar.set_label("Relative power (dB)")
        _save_paper_figure(fig, path)
        plt.close(fig)


def _plot_field_spectrogram_comparison(real_cases, path):
    """Place the field records in a column-native 3-by-2 method plate."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    records = []
    for recording, payload in real_cases.items():
        lower_name = recording.lower()
        if "humpback" in lower_name:
            records.append((0, "Humpback", payload))
        elif "shrimp" in lower_name:
            records.append((1, "Shrimp", payload))
    records.sort(key=lambda item: item[0])
    if len(records) != 2:
        raise RuntimeError("field plate requires one humpback and one shrimp record")

    with plt.rc_context({
        "font.size": 5.9,
        "axes.titlesize": 6.4,
        "axes.labelsize": 6.3,
        "xtick.labelsize": 5.4,
        "ytick.labelsize": 5.4,
    }):
        fig, axes = plt.subplots(
            4, 2, figsize=(3.1, 4.35), squeeze=False, sharex=True,
        )
        image = None
        for column, (_, record_label, payload) in enumerate(records):
            case = payload["case"]
            fs = float(payload["fs"])
            n_fft = int(payload["n_fft"])
            hop_size = int(payload["hop_size"])
            names = ("constrained", "best_fixed", "trellis", "greedy")
            spectra = [np.asarray(case[name][0]) for name in names]
            values_db = _spectrogram_db(spectra, floor_db=-75.0)
            times = (np.arange(spectra[0].shape[0]) + 1) * hop_size / fs
            frequencies_khz = np.fft.rfftfreq(n_fft, d=1.0 / fs) / 1000.0
            fixed_length = int(case["lengths"][case["fixed_index"]])
            method_titles = (
                "Hindsight schedule", "Best fixed window", "U-STFT", "Framewise selection",
            )
            upper_hz = min(4000.0, fs / 2.0) if column == 0 else fs / 2.0
            for method_row, (values, method_title) in enumerate(zip(
                    values_db, method_titles)):
                axis = axes[method_row, column]
                panel = 2 * method_row + column
                image = axis.pcolormesh(
                    times, frequencies_khz, values.T, shading="auto",
                    cmap="magma", vmin=-75.0, vmax=0.0, rasterized=True,
                )
                axis.set_xlim(0.0, times[-1])
                axis.set_ylim(0.0, upper_hz / 1000.0)
                axis.set_title(
                    f"({chr(ord('a') + panel)}) {record_label}: {method_title}",
                    pad=1.5,
                )
                if column == 0:
                    axis.set_ylabel("Frequency (kHz)")
                if method_row == 3:
                    axis.set_xlabel("Time (s)")
        fig.subplots_adjust(
            left=0.15, right=0.97, bottom=0.12, top=0.975,
            hspace=0.48, wspace=0.32,
        )
        color_axis = fig.add_axes((0.25, 0.035, 0.50, 0.012))
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.set_label("Relative power (dB)")
        _save_paper_figure(fig, path)
        plt.close(fig)


def _regret_record(label, slug, case, fs, hop_size, beta):
    """Package one valid hindsight-schedule regret trace for plotting.

    At every prefix t, dynamic programming recomputes the emission-minimizing
    schedule on exactly the nonzero stay/grow/reset edges used by the causal
    tracker.  This is not the framewise minimum, an HMM MAP path, or the
    entropy of the blended output.
    """
    transition_matrix = _dense_transition_matrix(case["lengths"], hop_size)
    audit = _hmm_path_code_audit(
        case["candidate_costs"], beta, transition_matrix
    )
    # The audit independently replays the Gibbs/HMM recursion.  Equality with
    # the repository PMF guards the plotted loss against implementation drift.
    posterior_error = float(np.max(np.abs(
        np.asarray(case["pmf"], dtype=float) - audit["posterior_curve"]
    )))
    if posterior_error > 1e-6:
        raise AssertionError(
            f"posterior replay differs from repository output by {posterior_error}"
        )
    spectrogram = np.asarray(case["trellis"][0])
    n_fft = 2 * (spectrogram.shape[1] - 1)
    if slug == "tone-impulse-conflict":
        upper_frequency = min(2000.0, fs / 2.0)
        floor_db = -70.0
    elif slug == "humpback-whale":
        upper_frequency = min(4000.0, fs / 2.0)
        floor_db = -75.0
    else:
        upper_frequency = fs / 2.0
        floor_db = -75.0 if slug == "snapping-shrimp" else -70.0

    posterior = audit["posterior_curve"]
    comparator_path = audit["comparator_path"]
    frame_index = np.arange(len(comparator_path))
    candidate_costs = np.asarray(case["candidate_costs"], dtype=float)
    emission_loss = np.full_like(candidate_costs, np.inf)
    for frame in range(len(candidate_costs)):
        valid = np.isfinite(candidate_costs[frame])
        valid_costs = candidate_costs[frame, valid]
        dynamic_beta = beta / (
            (np.std(valid_costs) if len(valid_costs) > 1 else 0.0) + 1e-6
        )
        emission_loss[frame, valid] = dynamic_beta * (
            valid_costs - np.min(valid_costs)
        )
    framewise_path = np.nanargmin(candidate_costs, axis=1)
    if not np.array_equal(framewise_path, case["framewise_indices_full"]):
        raise AssertionError(
            "framewise plot path must equal the per-frame metric argmin"
        )
    if not np.allclose(
            emission_loss[frame_index, framewise_path], 0.0,
            rtol=0.0, atol=2e-12):
        raise AssertionError(
            "framewise metric argmin must have zero standardized emission"
        )
    fixed_path = np.full(len(candidate_costs), int(case["fixed_index"]), dtype=int)
    candidate_h_time = np.asarray(case["candidate_h_time"], dtype=float)
    candidate_h_spec = np.asarray(case["candidate_h_spec"], dtype=float)
    component_curves = {
        "posterior_time": np.sum(posterior * candidate_h_time, axis=1),
        "posterior_frequency": np.sum(posterior * candidate_h_spec, axis=1),
        "comparator_time": candidate_h_time[frame_index, comparator_path],
        "comparator_frequency": candidate_h_spec[frame_index, comparator_path],
        "short_time": candidate_h_time[:, 0],
        "short_frequency": candidate_h_spec[:, 0],
        "long_time": candidate_h_time[:, -1],
        "long_frequency": candidate_h_spec[:, -1],
    }

    return {
        "label": label,
        "slug": slug,
        "time": (
            np.arange(1, len(audit["average_gap_curve"]) + 1)
            * float(hop_size) / float(fs)
        ),
        "average_regret": audit["average_gap_curve"],
        "average_path_code": audit["path_code_bound_curve"],
        "mix_curve": audit["mix_curve"],
        "comparator_emission_curve": audit["comparator_emission_curve"],
        "certificate_curve": audit["certificate_curve"],
        "posterior": posterior,
        "comparator_path": comparator_path,
        "causal_map_path": np.argmax(posterior, axis=1),
        "framewise_path": framewise_path,
        "fixed_path": fixed_path,
        "emission_loss": emission_loss,
        "candidate_costs": candidate_costs,
        "transition_matrix": transition_matrix,
        "component_curves": component_curves,
        "warmup": int(case["warmup"]),
        "candidate_lengths": np.asarray(case["lengths"], dtype=int),
        "spectrogram": spectrogram,
        "n_fft": n_fft,
        "fs": float(fs),
        "hop_size": int(hop_size),
        "beta": float(beta),
        "upper_frequency": float(upper_frequency),
        "floor_db": float(floor_db),
        "final_average_regret": audit["average_mix_gap"],
        "final_average_path_code": (
            audit["comparator_path_code"] / len(case["candidate_costs"])
        ),
        "posterior_replay_error": posterior_error,
        "reconstruction_mse": float(case["trellis_reconstruction_mse"]),
    }


def _plot_regret_suite(regret_records, figure_dir):
    """Restore the old multi-case story with a mathematically valid comparator."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    blue = "#0072B2"
    magenta = "#CC79A7"

    with plt.rc_context({
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.35))
        for index, (axis, record) in enumerate(zip(axes.flat, regret_records)):
            t = record["time"]
            regret = record["average_regret"]
            bound = record["average_path_code"]
            axis.plot(
                t, regret, color=blue, lw=2.0,
                label=r"Average regret $R_t/t$",
            )
            axis.plot(
                t, bound, color=magenta, lw=1.6, ls="--",
                label=r"Exact path-code bound $C_t^*/t$",
            )
            axis.fill_between(
                t, regret, bound, where=bound >= regret,
                color=magenta, alpha=0.10, linewidth=0,
            )
            axis.axhline(0.0, color="black", lw=0.65)
            axis.set_xlim(0.0, t[-1])
            axis.set_xlabel("Time (s)")
            if index % 2 == 0:
                axis.set_ylabel("Standardized log loss\n(nats/frame)")
            axis.set_title(f"({chr(ord('a') + index)}) {record['label']}")
            axis.grid(alpha=0.20)
            axis.text(
                0.97, 0.06,
                (f"final {record['final_average_regret']:.3f} "
                 f"$\\leq$ {record['final_average_path_code']:.3f}"),
                transform=axis.transAxes, ha="right", va="bottom", fontsize=7.2,
                bbox={"facecolor": "white", "edgecolor": "0.82",
                      "alpha": 0.88, "pad": 1.6},
            )

        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.988), frameon=False,
        )
        fig.suptitle(
            "Regret to the best constrained hindsight schedule",
            y=1.035, fontsize=10.5,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        save_figure(fig,
            figure_dir / "regret_suite.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # Per-record narratives restore the visual density of the original
        # manuscript without its invalid framewise-oracle or component-wise
        # regret statistics.  Every panel follows one causal repository run:
        # transform, state posterior, cumulative certificate, and average gap.
        story_names = {
            "hyperbolic-chirp": (
                "chirp_regret_story.png",
                ("chirp_regret.png", "chirp_oracle_regret.png"),
            ),
            "tone-impulse-conflict": (
                "conflict_regret_story.png",
                ("conflict_regret.png", "heisenberg_regret.png"),
            ),
            "humpback-whale": (
                "humpback_regret_story.png", ("humpback_regret.png",),
            ),
            "snapping-shrimp": (
                "shrimp_regret_story.png", ("shrimp_regret.png",),
            ),
        }
        orange = "#D55E00"
        green = "#009E73"
        for record in regret_records:
            t = np.asarray(record["time"])
            regret = np.asarray(record["average_regret"])
            bound = np.asarray(record["average_path_code"])
            posterior = np.asarray(record["posterior"])
            lengths = np.asarray(record["candidate_lengths"])
            is_field_recording = record["slug"] in {
                "humpback-whale", "snapping-shrimp"
            }
            frequencies = np.fft.rfftfreq(
                record["n_fft"], d=1.0 / record["fs"]
            )
            spectrogram_db = _spectrogram_db(
                [record["spectrogram"]], floor_db=record["floor_db"]
            )[0]

            fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.35))

            axis = axes[0, 0]
            spectrogram_image = axis.pcolormesh(
                t, frequencies, spectrogram_db.T, shading="auto",
                cmap="magma", vmin=record["floor_db"], vmax=0.0,
                rasterized=True,
            )
            axis.set_xlim(0.0, t[-1])
            axis.set_ylim(0.0, record["upper_frequency"])
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Frequency (Hz)")
            axis.set_title("(a) Dyadic U-STFT")
            colorbar = fig.colorbar(
                spectrogram_image, ax=axis, pad=0.015, fraction=0.047,
            )
            colorbar.set_label("Power (dB)")

            axis = axes[0, 1]
            posterior_image = axis.imshow(
                posterior.T, origin="lower", aspect="auto",
                interpolation="nearest",
                extent=(t[0], t[-1], -0.5, len(lengths) - 0.5),
                cmap="Blues", vmin=0.0, vmax=1.0, rasterized=True,
            )
            if is_field_recording:
                posterior_mean_level = (
                    posterior @ np.arange(len(lengths), dtype=float)
                )
                axis.plot(
                    t, posterior_mean_level, color=orange, lw=0.9,
                    label="Causal posterior mean level",
                )
            else:
                axis.step(
                    t, record["causal_map_path"], where="post", color=orange,
                    lw=1.15, label="Causal posterior MAP",
                )
                axis.step(
                    t, record["comparator_path"], where="post", color="black",
                    lw=1.05, ls="--",
                    label="Final-horizon hindsight schedule",
                )
            axis.set_xlim(0.0, t[-1])
            axis.set_ylim(-0.5, len(lengths) - 0.5)
            axis.set_yticks(np.arange(len(lengths)), lengths)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Window length (samples)")
            axis.set_title(
                "(b) Full causal posterior"
                if is_field_recording
                else "(b) Posterior and final-horizon path"
            )
            axis.legend(loc="best", fontsize=6.2, framealpha=0.88)
            colorbar = fig.colorbar(
                posterior_image, ax=axis, pad=0.015, fraction=0.047,
            )
            colorbar.set_label("Probability")

            axis = axes[1, 0]
            comparator = np.asarray(record["comparator_emission_curve"])
            certificate = np.asarray(record["certificate_curve"])
            mix = np.asarray(record["mix_curve"])
            axis.fill_between(
                t, comparator, certificate, color=magenta, alpha=0.10,
                linewidth=0,
            )
            axis.plot(
                t, comparator, color=orange, lw=1.65,
                label=r"Prefix-optimal emission $D_t^*$",
            )
            axis.plot(
                t, mix, color=blue, lw=1.9,
                label=r"Mixture log loss $\Lambda_t$",
            )
            axis.plot(
                t, certificate, color=magenta, lw=1.45, ls="--",
                label=r"Certificate $D_t^*+C_t^*$",
            )
            axis.set_xlim(0.0, t[-1])
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Cumulative standardized\nlog loss (nats)")
            axis.set_title("(c) Prefix-optimal path certificates")
            axis.grid(alpha=0.20)
            axis.legend(loc="best", fontsize=6.3, framealpha=0.88)

            axis = axes[1, 1]
            axis.fill_between(
                t, regret, bound, where=bound >= regret,
                color=magenta, alpha=0.10, linewidth=0,
            )
            axis.plot(
                t, regret, color=blue, lw=1.9,
                label=r"Average regret $R_t/t$",
            )
            axis.plot(
                t, bound, color=magenta, lw=1.45, ls="--",
                label=r"Exact bound $C_t^*/t$",
            )
            axis.axhline(0.0, color="black", lw=0.65)
            axis.set_xlim(0.0, t[-1])
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Standardized log loss\n(nats/frame)")
            axis.set_title("(d) Prefix regret and exact code bound")
            axis.grid(alpha=0.20)
            axis.legend(loc="best", fontsize=6.3, framealpha=0.88)
            axis.text(
                0.97, 0.07,
                (f"final {record['final_average_regret']:.3f} "
                 f"$\\leq$ {record['final_average_path_code']:.3f}"),
                transform=axis.transAxes, ha="right", va="bottom",
                fontsize=6.8,
                bbox={"facecolor": "white", "edgecolor": "0.82",
                      "alpha": 0.88, "pad": 1.4},
            )

            fig.suptitle(
                f"{record['label']}: transform, tracking, and schedule regret",
                y=0.995, fontsize=10.2,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
            story_filename, legacy_filenames = story_names[record["slug"]]
            save_figure(fig,
                figure_dir / story_filename, dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

            # The established three-panel visual grammar is retained here,
            # but only the top panel is regret.  The time and frequency panels
            # are descriptive cumulative-average candidate costs; they do not
            # claim separate component-wise regret bounds.
            warmup = int(record["warmup"])
            profile_time = t[warmup:]
            if len(profile_time) == 0:
                raise AssertionError("component profile has no post-warm-up frames")
            denominator = np.arange(1, len(profile_time) + 1, dtype=float)

            def cumulative_average(values):
                return np.cumsum(np.asarray(values)[warmup:]) / denominator

            fig, axes = plt.subplots(
                3, 1, figsize=(7.16, 6.95), sharex=True,
                gridspec_kw={"height_ratios": (1.05, 1.0, 1.0)},
            )

            axis = axes[0]
            axis.fill_between(
                t, regret, bound, where=bound >= regret,
                color=magenta, alpha=0.10, linewidth=0,
            )
            axis.plot(
                t, regret, color=blue, lw=2.25,
                label=r"Average regret $R_t/t$",
            )
            axis.plot(
                t, bound, color=magenta, lw=1.65, ls="--",
                label=r"Exact path-code bound $C_t^*/t$",
            )
            axis.axhline(0.0, color="black", lw=0.65)
            axis.set_ylabel("Standardized log loss\n(nats/frame)")
            axis.set_title("(a) Exact regret to the best allowed prefix schedule")
            axis.grid(alpha=0.20)
            axis.legend(loc="best", ncol=2, fontsize=6.7, framealpha=0.88)
            axis.text(
                0.98, 0.08,
                (f"final {record['final_average_regret']:.3f} "
                 f"$\\leq$ {record['final_average_path_code']:.3f}"),
                transform=axis.transAxes, ha="right", va="bottom",
                fontsize=6.9,
                bbox={"facecolor": "white", "edgecolor": "0.82",
                      "alpha": 0.88, "pad": 1.4},
            )

            component_styles = (
                ("posterior", "black", "-", 2.15,
                 "Posterior-expected candidate cost"),
                ("comparator", orange, "--", 1.65,
                 "Final-horizon constrained schedule"),
                ("short", green, ":", 1.8,
                 f"Fixed shortest ($L={lengths[0]}$)"),
                ("long", magenta, ":", 1.8,
                 f"Fixed longest ($L={lengths[-1]}$)"),
            )
            for component_axis, component, panel, name in (
                (axes[1], "time", "(b)", "Time"),
                (axes[2], "frequency", "(c)", "Frequency"),
            ):
                for prefix, color, linestyle, linewidth, label in component_styles:
                    component_axis.plot(
                        profile_time,
                        cumulative_average(
                            record["component_curves"][f"{prefix}_{component}"]
                        ),
                        color=color, ls=linestyle, lw=linewidth, label=label,
                    )
                component_axis.set_ylabel(
                    f"Normalized Rényi-3\n{name.lower()} cost"
                )
                component_axis.set_title(
                    f"{panel} {name} component: post-warm-up cumulative average"
                )
                component_axis.set_ylim(bottom=0.0)
                component_axis.grid(alpha=0.20)
            axes[1].legend(
                loc="best", ncol=2, fontsize=6.5, framealpha=0.88,
            )
            axes[-1].set_xlabel("Time (s)")
            axes[-1].set_xlim(0.0, t[-1])
            fig.suptitle(
                f"{record['label']}: exact schedule regret and concentration costs",
                y=0.995, fontsize=10.2,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.973))
            profile_filename = story_filename.replace(
                "_story.png", "_profile.png"
            )
            save_figure(fig,
                figure_dir / profile_filename, dpi=300, bbox_inches="tight"
            )
            for filename in legacy_filenames:
                save_figure(fig,
                    figure_dir.parent / filename, dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)


def _plot_experiment_summary(
        rows, clean_case, posterior_case, runtime_rows, figure_dir,
        real_cases=None):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_spectrogram_evidence(clean_case, posterior_case, figure_dir)
    _plot_field_recording_spectrograms(real_cases, figure_dir)
    colors = {"trellis": "#0072B2", "greedy": "#D55E00", "best_fixed": "#009E73"}
    labels = {"trellis": "U-STFT (causal)",
              "greedy": "Framewise selection (causal)",
              "best_fixed": "Fixed causal STFT (hindsight support)"}
    finite_snrs = sorted({float(row["snr_db"]) for row in rows
                          if row["spacing"] == "geometric" and row["snr_db"] != "inf"})

    fig, axes = plt.subplots(4, 1, figsize=(3.5, 6.75))
    panels = (
        ("normalized_renyi3", "Normalized Rényi-3", False),
        ("trajectory_variation_octaves_per_s", "Variation (octaves/s)", False),
        ("framewise_length_error_octaves",
         "Distance to framewise selection (octaves)", False),
    )
    for panel, (axis, (key, ylabel, _)) in enumerate(zip(axes[:3], panels)):
        for method in ("trellis", "greedy", "best_fixed"):
            medians = []
            q1 = []
            q3 = []
            for snr in finite_snrs:
                vals = [float(row[key]) for row in rows
                        if row["spacing"] == "geometric" and row["method"] == method
                        and row["snr_db"] != "inf" and float(row["snr_db"]) == snr
                        and np.isfinite(float(row[key]))]
                medians.append(np.median(vals))
                q1.append(np.percentile(vals, 25))
                q3.append(np.percentile(vals, 75))
            axis.plot(finite_snrs, medians, marker="o", color=colors[method],
                      label=labels[method])
            axis.fill_between(finite_snrs, q1, q3, color=colors[method], alpha=0.13)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"({chr(ord('a') + panel)})", loc="left", fontsize=7.0)
        axis.tick_params(labelsize=6.0)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=5.7, ncol=1)

    axis = axes[3]
    for spacing, style in (("geometric", "-o"), ("linear", "--s")):
        selected = [row for row in runtime_rows if row["spacing"] == spacing]
        bank_label = "Dyadic bank" if spacing == "geometric" else "Linear bank"
        axis.plot([row["n_fft"] for row in selected],
                  [row["milliseconds_per_frame"] for row in selected], style,
                  label=bank_label)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("FFT length")
    axis.set_ylabel("Runtime (ms/frame)")
    axis.set_title("(d) Candidate-bank runtime", loc="left", fontsize=7.0)
    axis.tick_params(labelsize=6.0)
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=5.7)
    fig.tight_layout()
    save_figure(fig, figure_dir / "dyadic_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Clean-signal trajectory: compare smoothing and response latency directly.
    warmup = clean_case["warmup"]
    hop_size = 32
    fs = 4096
    time_axis = (np.arange(len(clean_case["pmf"])) + 1) * hop_size / fs
    costs = clean_case["candidate_costs"]
    greedy_indices = np.zeros(len(costs), dtype=int)
    for frame in range(len(costs)):
        valid = np.flatnonzero(np.isfinite(costs[frame]))
        greedy_indices[frame] = valid[np.argmin(costs[frame, valid])]
    lengths = clean_case["lengths"]
    framewise_choice = lengths[greedy_indices]
    fig, axis = plt.subplots(figsize=(11, 4.2))
    axis.step(time_axis, clean_case["trellis"][2], where="post", lw=2.2,
              color=colors["trellis"], label=labels["trellis"])
    axis.step(time_axis, clean_case["greedy"][2], where="post", lw=1.2,
              color=colors["greedy"], alpha=0.8, label=labels["greedy"])
    axis.step(time_axis, framewise_choice, where="post", lw=1.0,
              color="black", alpha=0.45,
              linestyle=":", label="Framewise selection")
    for event in (1.0, 2.0, 3.0):
        axis.axvline(event, color="0.5", lw=0.8, linestyle="--")
    axis.axvspan(0, warmup * hop_size / fs, color="0.8", alpha=0.25,
                 label="causal warm-up")
    axis.set_yscale("log", base=2)
    axis.set_yticks(lengths, [str(value) for value in lengths])
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Window length (samples)")
    axis.set_title("Known transition tracking: fast chirp, close tones, and impulses")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figure_dir / "transition_tracking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Finite-horizon regret is computed against the best emission schedule in
    # hindsight under the *same* stay/grow/reset support.  At every prefix t,
    # dynamic programming gives D_t^*, and the mixture identity certifies
    #   R_t = Lambda_t - D_t^* <= C_t^*,
    # where C_t^* is that comparator's negative-log path probability.
    curve_time = (
        np.arange(len(clean_case["hmm_curves"]["average_gap"])) + 1
    ) * hop_size / fs
    mix_curve = clean_case["hmm_curves"]["mix"]
    comparator_curve = clean_case["hmm_curves"]["comparator_emission"]
    certificate_curve = clean_case["hmm_curves"]["certificate"]
    average_regret = clean_case["hmm_curves"]["average_gap"]
    average_code = clean_case["hmm_curves"]["path_code_bound"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.45))
    axis = axes[0]
    axis.plot(
        curve_time, comparator_curve, color=colors["greedy"], lw=2.1,
        label=r"Best constrained schedule $D_t^*$",
    )
    axis.fill_between(
        curve_time, comparator_curve, certificate_curve,
        color="#CC79A7", alpha=0.12, linewidth=0,
    )
    axis.plot(
        curve_time, certificate_curve, color="#CC79A7", lw=1.9, ls="--",
        label=r"Certificate $D_t^*+C_t^*$",
    )
    axis.plot(
        curve_time, mix_curve, color=colors["trellis"], lw=2.5,
        label=r"Mixture log loss $\Lambda_t$",
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Cumulative standardized log loss (nats)")
    axis.set_title("(a) Cumulative losses and certificate")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, loc="upper left")

    axis = axes[1]
    axis.plot(
        curve_time, average_regret, color=colors["trellis"], lw=2.5,
        label=r"Average regret $R_t/t$",
    )
    axis.plot(
        curve_time, average_code, color="#CC79A7", lw=1.9, ls="--",
        label=r"Exact bound $C_t^*/t$",
    )
    axis.fill_between(
        curve_time, average_regret, average_code,
        where=average_code >= average_regret,
        color="#CC79A7", alpha=0.10, linewidth=0,
    )
    axis.axhline(0.0, color="black", lw=0.8)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Average standardized log loss (nats/frame)")
    axis.set_title("(b) Regret to the hindsight schedule")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, loc="upper right")
    axis.text(
        0.97, 0.07,
        (f"Final: {average_regret[-1]:.3f} $\\leq$ "
         f"{average_code[-1]:.3f} nats/frame"),
        transform=axis.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9,
              "boxstyle": "round,pad=0.3"},
    )
    fig.suptitle(
        "Competitiveness against the best constrained hindsight schedule",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir / "hmm_path_bound.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    clean_rows = [row for row in rows if row["spacing"] == "geometric"
                  and row["seed"] == 0 and row["snr_db"] == "inf"]
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.75))
    methods = ["trellis", "greedy", "best_fixed"]
    axes[0].bar(range(3), [_median([r for r in clean_rows if r["method"] == m],
                                  "tone_separation_db") for m in methods],
                color=[colors[m] for m in methods])
    axes[0].set_ylabel("Tone separation (dB)\nhigher is better")
    axes[0].set_title("(a) Close-tone resolution", fontsize=7.2)
    axes[1].bar(range(3), [_median([r for r in clean_rows if r["method"] == m],
                                  "transient_width_ms") for m in methods],
                color=[colors[m] for m in methods])
    axes[1].set_ylabel("Impulse width (ms)\nlower is better")
    axes[1].set_title("(b) Transient concentration", fontsize=7.2)
    axes[2].bar(range(3), [_median([r for r in clean_rows if r["method"] == m],
                                  "transient_peak_delay_ms") for m in methods],
                color=[colors[m] for m in methods])
    axes[2].set_ylabel("Peak delay (ms)\nlower is better")
    axes[2].set_title("(c) Transient timing", fontsize=7.2)
    for axis in axes:
        axis.set_xticks(
            range(3),
            ["U-STFT\n(causal)", "Framewise\nselection",
             "Fixed causal STFT\n(hindsight support)"],
        )
        axis.tick_params(labelsize=6.2)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figure_dir / "ground_truth_resolution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Ground-truth metrics across noise levels show where smoothing helps and
    # where the framewise selector remains the stronger clean-signal baseline.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    resolution_metrics = (
        ("tone_separation_db", "Tone separation (dB; higher better)"),
        ("transient_width_ms", "Impulse width (ms; lower better)"),
        ("transient_peak_delay_ms", "Impulse delay (ms; lower better)"),
    )
    for axis, (key, ylabel) in zip(axes, resolution_metrics):
        for method in methods:
            medians = []
            q1 = []
            q3 = []
            for snr in finite_snrs:
                values = [float(row[key]) for row in rows
                          if row["spacing"] == "geometric"
                          and row["method"] == method
                          and row["snr_db"] != "inf"
                          and float(row["snr_db"]) == snr
                          and np.isfinite(float(row[key]))]
                medians.append(np.median(values))
                q1.append(np.percentile(values, 25))
                q3.append(np.percentile(values, 75))
            axis.plot(finite_snrs, medians, marker="o", color=colors[method],
                      label=labels[method])
            axis.fill_between(finite_snrs, q1, q3, color=colors[method], alpha=0.13)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Ground-truth resolution across additive noise")
    fig.tight_layout()
    save_figure(fig, figure_dir / "resolution_vs_snr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Preserve the complete posterior, rather than showing only its argmax.
    # The heatmap uses a representative noisy case (seed 0, SNR 0 dB), and the
    # lower panels expose distributions in four known signal regimes.
    pmf = np.asarray(posterior_case["pmf"], dtype=float)
    lengths = np.asarray(posterior_case["lengths"], dtype=int)
    frame_times = (np.arange(len(pmf)) + 1) * hop_size / fs
    representatives = (
        (0.40625, "fast chirp"),
        (1.1015625, "tone acquisition"),
        (2.0234375, "first click"),
        (2.328125, "post-click"),
    )

    fig = plt.figure(figsize=(12.0, 6.2))
    grid = fig.add_gridspec(
        2, 4, height_ratios=(2.25, 1.0), left=0.07, right=0.90,
        bottom=0.12, top=0.88, hspace=0.52, wspace=0.22,
    )
    heatmap_axis = fig.add_subplot(grid[0, :])
    image = heatmap_axis.imshow(
        pmf.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(frame_times[0], frame_times[-1], -0.5, len(lengths) - 0.5),
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )
    posterior_mean_level = pmf @ np.arange(len(lengths), dtype=float)
    hindsight_path = np.asarray(
        posterior_case["hmm_comparator_path"], dtype=int
    )
    heatmap_axis.plot(
        frame_times, posterior_mean_level, color="white", linewidth=1.5,
        label="Posterior mean dyadic level",
    )
    # White underlay keeps the offline comparator legible without obscuring
    # the PMF.  It is diagnostic only and never feeds the causal tracker.
    heatmap_axis.step(
        frame_times, hindsight_path, where="post", color="white",
        linewidth=3.3, alpha=0.95,
    )
    heatmap_axis.step(
        frame_times, hindsight_path, where="post", color="#D55E00",
        linewidth=1.7, label="Best constrained schedule (hindsight only)",
    )
    for event_time in (1.0, 2.0, 3.0):
        heatmap_axis.axvline(
            event_time, color="black", linewidth=0.8, linestyle="--", alpha=0.65
        )
    heatmap_axis.set_yticks(np.arange(len(lengths)), [str(L) for L in lengths])
    heatmap_axis.set_xlabel("Time (s)")
    heatmap_axis.set_ylabel("Window length (samples)")
    heatmap_axis.legend(loc="upper right", fontsize=8)
    colorbar = fig.colorbar(image, ax=heatmap_axis, pad=0.012, aspect=28)
    colorbar.set_label("Posterior probability")

    bar_axes = []
    for column, (target_time, regime) in enumerate(representatives):
        axis = fig.add_subplot(
            grid[1, column], sharey=bar_axes[0] if bar_axes else None
        )
        bar_axes.append(axis)
        frame = int(np.searchsorted(frame_times, target_time, side="left"))
        frame = min(frame, len(frame_times) - 1)
        bars = axis.bar(
            np.arange(len(lengths)), pmf[frame], color="#0072B2", width=0.78
        )
        bars[hindsight_path[frame]].set_edgecolor("#D55E00")
        bars[hindsight_path[frame]].set_linewidth(2.2)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(np.arange(len(lengths)), [str(L) for L in lengths],
                        rotation=35)
        axis.set_xlabel("Window length")
        if column == 0:
            axis.set_ylabel("Posterior probability")
        else:
            axis.tick_params(labelleft=False)
        axis.set_title(
            (f"{frame_times[frame]:.3f} s: {regime}\n"
             f"hindsight $L={lengths[hindsight_path[frame]]}$"),
            fontsize=9,
        )
        axis.grid(axis="y", alpha=0.22)

    fig.suptitle(
        "Dyadic posterior and constrained hindsight schedule at 0 dB (seed 0)",
        y=0.98,
    )
    save_figure(fig, figure_dir / "posterior_pmf.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_audit_report(path, rows, clean_case, runtime_rows, real_rows,
                        regret_records, legacy_rows, joint_renyi_rows,
                        joint_renyi_snr_summary):
    """Write a compact evidence ledger for paper authors and reviewers."""
    geometric = [row for row in rows if row["spacing"] == "geometric"]
    linear = [row for row in rows if row["spacing"] == "linear"]
    clean = [row for row in geometric if row["seed"] == 0 and row["snr_db"] == "inf"]
    trellis = [row for row in geometric if row["method"] == "trellis"]
    greedy = [row for row in geometric if row["method"] == "greedy"]
    fixed = [row for row in geometric if row["method"] == "best_fixed"]
    max_runtime = [row for row in runtime_rows if row["n_fft"] == 4096]
    geo_runtime = next(row for row in max_runtime if row["spacing"] == "geometric")
    lin_runtime = next(row for row in max_runtime if row["spacing"] == "linear")
    legacy_wins = sum(row["universal_kurtosis_win"] for row in legacy_rows)
    joint_clean = [
        row for row in joint_renyi_rows
        if row["case"] == "tone-impulse-conflict"
    ]
    joint_method_names = {
        "posterior_blend": "posterior blend",
        "constrained_schedule": "constrained schedule",
        "framewise_selection": "framewise selection",
        "best_fixed_support": "final-horizon tracker-loss-optimal fixed STFT",
    }
    snr_rankings = []
    for snr_db in sorted({row["snr_db"] for row in joint_renyi_snr_summary}):
        selected = [
            row for row in joint_renyi_snr_summary
            if row["snr_db"] == snr_db
        ]
        winner = next(row for row in selected if row["rank_by_median_h3"] == 1)
        snr_rankings.append(
            f"{snr_db:g} dB: {joint_method_names[winner['method']]} "
            f"({winner['median_h3_bits']:.4f} bits)"
        )
    snr_ranking_text = "; ".join(snr_rankings)
    posterior_sweep_wins = sum(
        row["finite_family_run_wins"] for row in joint_renyi_snr_summary
        if row["method"] == "posterior_blend"
    )
    sweep_runs = sum(
        row["runs"] for row in joint_renyi_snr_summary
        if row["method"] == "posterior_blend"
    )

    def clean_value(method, key):
        return next(row[key] for row in clean if row["method"] == method)

    def joint_clean_value(method, key):
        return next(row[key] for row in joint_clean if row["method"] == method)

    joint_clean_winner = min(joint_clean, key=lambda row: row["h3_bits"])
    chirp_record = next(
        record for record in regret_records
        if record["slug"] == "hyperbolic-chirp"
    )

    report = f"""# Universal STFT experimental evidence audit

Generated deterministically by `python -m benchmarks.run_paper --run-experiments`.

## What is measured now

- Monte Carlo conflict signal: 6 seeds at SNR -10, 0, 10, 20 dB plus one clean run, using a fast chirp, a known pair of tones at 500/532 Hz, and impulses at 2/3 s.
- Implementation source: the dyadic transform is imported directly from this package's `universal_stft/numpy/stft.py`; only the hop-linear grid ablation is implemented locally.
- Baselines: dyadic soft trellis, framewise greedy Rényi selection (a Liuni-style myopic baseline, not the regret comparator), the final-horizon fixed STFT minimizing cumulative scaled Gibbs emission on the common suffix, and a linear-grid trellis ablation.
- Metrics: full-frequency joint spectrogram Rényi-3 information from a normalized two-dimensional energy PMF, the exact HMM mix-loss/path-code diagnostic against the best constrained hindsight schedule, close-tone peak/valley separation, impulse response width, trajectory total variation, switch rate, transition latency, reconstruction MSE, and wall-clock runtime.
- Real audio: humpback and snapping-shrimp recordings are evaluated only with descriptive concentration, stability, distance to the myopic selector, reconstruction, and runtime metrics. No ground-truth resolution or information-optimality claim is made for them.

The repository's earlier {len(legacy_rows)} CSV files measure reconstruction MSE, spectral convergence, envelope-spectrum kurtosis, and phase jitter. They contain no cross-signal statistical summary and no ground-truth resolution metric; reconstruction is near machine precision for both adaptive and fixed transforms, while the adaptive transform has the highest listed kurtosis on only {legacy_wins} of {len(legacy_rows)} signals. Those artifacts are useful smoke tests, not evidence of across-the-board resolution gains.

## Quantitative headline (current implementation)

- Dyadic bank: {int(clean[0]['states'])} states. Linear ablation: {int(next(row['states'] for row in linear if row['seed'] == 0 and row['snr_db'] == 'inf'))} states.
- At NFFT=4096, dyadic runtime is {geo_runtime['milliseconds_per_frame']:.3f} ms/frame versus {lin_runtime['milliseconds_per_frame']:.3f} ms/frame for the linear bank ({lin_runtime['milliseconds_per_frame'] / geo_runtime['milliseconds_per_frame']:.2f}x ratio on this machine).
- Clean close-tone separation: trellis {clean_value('trellis', 'tone_separation_db'):.2f} dB; greedy {clean_value('greedy', 'tone_separation_db'):.2f} dB; best fixed {clean_value('best_fixed', 'tone_separation_db'):.2f} dB.
- Clean impulse width: trellis {clean_value('trellis', 'transient_width_ms'):.2f} ms; greedy {clean_value('greedy', 'transient_width_ms'):.2f} ms; best fixed {clean_value('best_fixed', 'transient_width_ms'):.2f} ms.
- Clean impulse peak delay: trellis {clean_value('trellis', 'transient_peak_delay_ms'):.2f} ms; greedy {clean_value('greedy', 'transient_peak_delay_ms'):.2f} ms; best fixed {clean_value('best_fixed', 'transient_peak_delay_ms'):.2f} ms.
- Hyperbolic-chirp U-STFT identity-reconstruction MSE: {chirp_record['reconstruction_mse']:.3e}.
- Clean full-frequency joint H3: posterior blend {joint_clean_value('posterior_blend', 'h3_bits'):.6f} bits; framewise selection {joint_clean_value('framewise_selection', 'h3_bits'):.6f}; final-horizon tracker-loss-optimal fixed STFT {joint_clean_value('best_fixed_support', 'h3_bits'):.6f} at L={int(joint_clean_value('best_fixed_support', 'best_fixed_length'))}; constrained schedule {joint_clean_value('constrained_schedule', 'h3_bits'):.6f}. The finite four-method minimum is {joint_method_names[joint_clean_winner['method']]}. The separately optimized fixed-bank 2-D reference is {joint_clean_value('best_fixed_support', 'joint_optimal_fixed_h3_bits'):.6f} bits at L={int(joint_clean_value('best_fixed_support', 'joint_optimal_fixed_length'))}.
- Controlled SNR-sweep lowest median H3 by condition: {snr_ranking_text}. The posterior blend is the finite-four-method minimum on {posterior_sweep_wins} of {sweep_runs} individual noisy runs; this count is reported rather than selecting only favorable cases.
- Median trajectory variation over noisy dyadic runs: trellis {_median(trellis, 'trajectory_variation_octaves_per_s'):.3f} octaves/s; greedy {_median(greedy, 'trajectory_variation_octaves_per_s'):.3f}; best fixed {_median(fixed, 'trajectory_variation_octaves_per_s'):.3f}.
- Median reconstruction MSE: trellis {_median(trellis, 'reconstruction_mse'):.3e}; greedy {_median(greedy, 'reconstruction_mse'):.3e}; best fixed {_median(fixed, 'reconstruction_mse'):.3e}.
- Legacy blend-minus-oracle is negative on {100 * clean_case['audit']['legacy_negative_fraction']:.1f}% of clean post-warm-up frames; it is therefore not regret. The posterior Jensen inequality used to justify that blended output is violated on {100 * clean_case['audit']['posterior_jensen_violation_fraction']:.1f}% of those frames.
- The exact HMM audit does hold: clean mix log loss {clean_case['audit']['hmm_mix_log_loss']:.2f} is below comparator emission loss plus path code ({clean_case['audit']['hmm_comparator_emission_log_loss']:.2f} + {clean_case['audit']['hmm_comparator_path_code']:.2f}) by {clean_case['audit']['hmm_bound_margin']:.2f} nats. The large, linearly accumulating path code is why this does not imply vanishing average regret.

## Claims supported by executed evidence

1. A dyadic bank materially reduces candidate count and runtime relative to a hop-linear bank while retaining a broad time-frequency span.
2. The envelope-normalized filtering stream reconstructs the tested synthetic and real signals to floating-point precision, including greedy and time-varying trajectories.
3. The soft trellis is measurably smoother than framewise greedy selection; report the actual variation/switch-rate numbers rather than saying it eliminates chattering.
4. The controlled close-tone and impulse metrics directly test the two resolution axes. Use them only where the measured trellis values are competitive; do not infer superiority from spectrogram appearance.
5. The implementation's normalized one-sided Rényi-3 score has a positive finite-dimensional floor, computed by `normalized_renyi3_lower_bound`; randomized and structured candidates in the repository tests satisfy it.
6. The new two-dimensional audit forms one positive full-frequency spectrogram PMF per region. Interior rFFT bins are duplicated through multiplicity two while DC and Nyquist have multiplicity one. Raw H3 in bits is primary; H3/log2(N) is secondary. Every displayed prefix is computed once from raw power and raw cubed power.
7. "Best fixed window" always denotes the single fixed support chosen in hindsight by minimum cumulative scaled Gibbs emission on the common suffix. It is not the support that minimizes global joint H3. That separate fixed-bank optimum is labeled explicitly as the joint-H3 reference, and the four-method minimum is reported separately for every complete record or Monte Carlo run.

## Claims not supported by the present algorithm/results

1. **"Exact finite-state KT" and O(k log M) regret.** The code now uses a row-stochastic, physical-run-length dyadic HMM, not a state-index shortcut, but it still does not retain Dirichlet action counts for every path. It has the exact mix-loss-versus-path-code inequality measured here; a fixed terminal stay probability nevertheless assigns exponentially decreasing mass to a constant path, so the generic path-code bound is linear in M, not logarithmic. Call this a run-length-aware finite-state prior and state its exact bound rather than a KT-rate guarantee.
2. **"Vanishing regret against an omniscient oracle."** The old plots subtract per-frame minimum candidate entropy from the entropy of a different object, the blended frame. That difference can be negative and is not regret. Framewise greedy selection is only a myopic smoothing baseline. The valid comparator starts at H and minimizes emission over allowed stay/grow/reset schedules; the new HMM plot reports that loss, the selected schedule's exact path code, and the empirical average mix-loss gap. With the fixed terminal prior, decay of the unpenalized gap is empirical only, not generically guaranteed.
3. **Jensen domination of the blended spectrum.** Rényi entropy composed with a complex spectral blend is not shown concave on the used domain; the code also replaces predictive weights by posterior weights and normalizes the blended time entropy by an expected, generally noninteger length. The measured inequality violations rule out presenting this step as a proof.
4. **Historical 2-D entropy figures.** `running_2d_renyi` was already cumulative, but the old plotting code cumulatively summed it a second time and then applied a dB transform to entropy values. Those historical curves remain inadmissible. The replacement `joint_renyi3_concentration.png` instead computes H3 directly from one normalized full-frequency PMF for each raw prefix, with no entropy-domain cumulative sum and no dB transform.
5. **Zero-latency exact inversion.** Analysis is causal with no delay. `mode="filtering"` builds the companion stream in the same pass, but each companion frame waits until its overlap envelope is final: a fixed delay of `max_L / H - 1` frames. Describe it as bounded-delay streaming synthesis, not zero-latency.
6. **O(1) memory.** The engine stores an NFFT buffer and a K-by-NFFT window bank; the array wrapper additionally returns a K-by-M-by-F candidate tensor. State complexity in the relevant variables instead of O(1).
7. **Real-audio superiority.** The recordings have no resolution ground truth. They support reconstruction and stability observations, not claims that the inferred length is objectively optimal.
8. **A common end anchor is not a common temporal center.** Every symmetric Gaussian is right-aligned, so its peak occurs roughly L/2 samples before the anchor. Different candidate lengths therefore have different effective delays on transients even though their buffers end together. Report the measured peak-delay metric and avoid claiming sample-level transient timing or universal phase coherence.
9. **CQT comparison is still outstanding.** A fair causal CQT comparison needs matched bandwidth, hop, latency, and inversion settings. Do not claim superiority to CQT until that controlled baseline is added.

## What belongs in the paper

- Main spectrograms: shared-dB controlled and field comparisons plus the complete dyadic candidate bank. Treat greedy strictly as a myopic baseline and label field cases qualitative.
- Posterior figure: the complete PMF and representative state distributions, with the offline emission-optimal constrained schedule clearly separated from the causal tracker.
- Theory/evidence figures: the four-case regret suite and the detailed HMM cumulative mix loss versus the best constrained hindsight schedule, both with the exact path-code bound. Label any unpenalized average-gap decay as empirical.
- Joint concentration figure: the full SNR sweep, clean prefix trace, and complete-record controlled/field values in `joint_renyi3_concentration.png`. Raw H3 bits are the Baraniuk-style information measure. Treat the field panels as descriptive and reserve "information-optimal" for an explicitly named finite family.
- Main text or a compact table: dyadic versus linear candidate count and measured runtime scaling.
- Appendix/repository: all SNR/seed rows, runtime scaling, exact commands, and the evidence limitations above.
- Do not reuse the old framewise-oracle statistic or the historical double-cumulative Rényi statistic.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/ustft-mpl OPENBLAS_NUM_THREADS=1 \\
  python -m benchmarks.run_paper --run-experiments
```
"""
    Path(path).write_text(report)


@paper_figure_build
def run_paper_experiments(output_dir=None, quick=False, wav_dir=None,
                          figure_dir=None):
    """Run the complete deterministic evaluation and write CSV/JSON/PNG outputs."""
    wav_root = Path(wav_dir) if wav_dir else _WAV_ROOT
    chirp_path, humpback_path, shrimp_path = _require_paper_audio(wav_root)
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir else _REPOSITORY_ROOT / "results" / "paper"
    )
    figure_dir = (
        Path(figure_dir).expanduser().resolve()
        if figure_dir else output_dir / "figures" / "audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    seeds = range(2) if quick else range(6)
    snrs = (0.0, 20.0) if quick else (-10.0, 0.0, 10.0, 20.0)
    rows = []
    clean_case = None
    posterior_case = None
    joint_renyi_snr_records = []
    for seed in seeds:
        for snr_db in snrs:
            x, metadata = _benchmark_signal(seed=seed, snr_db=snr_db)
            for spacing in ("geometric", "linear"):
                case_rows, case = _evaluate_case(
                    x, metadata, seed, snr_db, spacing=spacing
                )
                rows.extend(case_rows)
                if seed == 0 and snr_db == 0.0 and spacing == "geometric":
                    posterior_case = case
                if spacing == "geometric":
                    joint_renyi_snr_records.append(
                        _joint_renyi3_case_record(
                            f"Controlled, {snr_db:g} dB, seed {seed}",
                            f"{snr_db:g} dB", f"controlled-{snr_db:g}db-seed{seed}",
                            case, metadata["fs"], n_fft=1024, hop_size=32,
                            case_type="controlled_snr",
                            seed=int(seed), snr_db=float(snr_db),
                        )
                    )

    # A single noiseless row set anchors the ground-truth resolution figures.
    x_clean, metadata = _benchmark_signal(seed=0, snr_db=np.inf)
    for spacing in ("geometric", "linear"):
        case_rows, case = _evaluate_case(
            x_clean, metadata, 0, np.inf, spacing=spacing
        )
        rows.extend(case_rows)
        if spacing == "geometric":
            clean_case = case

    runtime_rows = _runtime_scaling()
    wav_paths = [humpback_path, shrimp_path]
    real_rows, real_cases = _evaluate_real_audio(
        wav_paths, return_cases=True
    )

    # Offline two-dimensional concentration audit.  This is deliberately
    # separate from the causal algorithm: every method is evaluated on the
    # same post-warm-up, full-frequency time-by-frequency grid.  The repeated
    # fixed-support baseline is selected once by cumulative scaled Gibbs
    # emission beta*d; the 2-D-optimal fixed support is a labeled audit only.
    joint_renyi_clean_record = _joint_renyi3_case_record(
            "Tone--impulse conflict", "Controlled",
            "tone-impulse-conflict", clean_case, metadata["fs"],
            n_fft=1024, hop_size=32, case_type="controlled_clean",
            seed=0, snr_db="inf",
        )
    joint_renyi_representative_records = [joint_renyi_clean_record]
    for recording, payload in sorted(real_cases.items()):
        lower_name = recording.lower()
        if "humpback" in lower_name:
            label, short_label, slug = (
                "Humpback whale", "Humpback", "humpback-whale"
            )
        elif "shrimp" in lower_name:
            label, short_label, slug = (
                "Snapping shrimp", "Shrimp", "snapping-shrimp"
            )
        else:
            continue
        joint_renyi_representative_records.append(_joint_renyi3_case_record(
            label, short_label, slug, payload["case"], payload["fs"],
            payload["n_fft"], payload["hop_size"], case_type="field",
        ))
    # Restore the paper's broad qualitative case coverage with one comparator
    # definition throughout: best constrained hindsight schedule on the same
    # stay/grow/reset trellis.  The hyperbolic chirp is the original 15 s WAV.
    regret_records = []
    chirp_fs, chirp = _read_wav_mono(chirp_path, max_seconds=15.0)
    chirp_case = _case_outputs(
        chirp, n_fft=4096, hop_size=128, beta=1.5,
        spacing="geometric",
    )
    regret_records.append(_regret_record(
        "Hyperbolic chirp", "hyperbolic-chirp", chirp_case,
        chirp_fs, hop_size=128, beta=1.5,
    ))
    joint_renyi_representative_records.insert(
        0,
        _joint_renyi3_case_record(
            "Hyperbolic chirp", "Chirp", "hyperbolic-chirp",
            chirp_case, chirp_fs, n_fft=4096, hop_size=128,
            case_type="synthetic_chirp",
        ),
    )
    regret_records.append(_regret_record(
        "Tone--impulse conflict", "tone-impulse-conflict", clean_case,
        metadata["fs"], hop_size=32, beta=1.5,
    ))
    for recording, payload in real_cases.items():
        lower_name = recording.lower()
        if "humpback" in lower_name:
            label, slug = "Humpback whale", "humpback-whale"
        elif "shrimp" in lower_name:
            label, slug = "Snapping shrimp", "snapping-shrimp"
        else:
            continue
        regret_records.append(_regret_record(
            label, slug, payload["case"], payload["fs"],
            payload["hop_size"], beta=1.5,
        ))
    regret_order = {
        "hyperbolic-chirp": 0,
        "tone-impulse-conflict": 1,
        "humpback-whale": 2,
        "snapping-shrimp": 3,
    }
    regret_records.sort(key=lambda record: regret_order[record["slug"]])
    if len(regret_records) != 4:
        raise RuntimeError(
            "The four-case regret suite requires chirp, conflict, humpback, and shrimp"
        )
    if [record["slug"] for record in joint_renyi_representative_records] != [
            record["slug"] for record in regret_records]:
        raise AssertionError(
            "per-record Rényi and regret suites must use the same case order"
        )
    joint_renyi_records = (
        joint_renyi_snr_records + joint_renyi_representative_records
    )
    joint_renyi_rows = _joint_renyi3_rows(joint_renyi_records)
    joint_renyi_snr_summary = _joint_renyi3_snr_summary(
        joint_renyi_snr_records
    )

    # Regenerate the paper's established spectrogram vocabulary from
    # the current dyadic repository implementation.  These intentionally
    # replace the old hop-linear images bearing the same manuscript names.
    legacy_figure_dir = figure_dir.parent
    _plot_window_length_tradeoff(
        clean_case, metadata, figure_dir / "window_length_tradeoff.png",
    )
    _plot_window_length_trellis(
        figure_dir / "window_length_trellis.png",
    )
    _plot_case_spectrogram_grid(
        clean_case, metadata["fs"], 1024, 32,
        "Chirp, close tones, and impulses (clean)", 2000.0,
        legacy_figure_dir / "conflict_spectrogram_grid.png",
    )
    _plot_case_spectrogram_grid(
        clean_case, metadata["fs"], 1024, 32,
        "Chirp, close tones, and impulses (clean)", 2000.0,
        figure_dir / "tone-impulse-conflict_spectrogram_comparison.png",
    )
    _plot_case_spectrogram_grid(
        clean_case, metadata["fs"], 1024, 32,
        "Close-tone and impulse resolution", 2000.0,
        legacy_figure_dir / "heisenberg_grid.png",
    )
    _plot_chirp_spectrogram_teaching(
        chirp_case, chirp_fs, 4096, 128,
        figure_dir / "hyperbolic-chirp_spectrogram_comparison.png",
    )
    for recording, payload in real_cases.items():
        lower_name = recording.lower()
        if "humpback" in lower_name:
            title = "Humpback whale song (qualitative)"
            upper_frequency = min(4000.0, payload["fs"] / 2.0)
            filename = "humpback_spectrogram_grid.png"
        elif "shrimp" in lower_name:
            title = "Snapping-shrimp soundscape (qualitative)"
            upper_frequency = payload["fs"] / 2.0
            filename = "shrimp_spectrogram_grid.png"
        else:
            continue
        _plot_case_spectrogram_grid(
            payload["case"], payload["fs"], payload["n_fft"],
            payload["hop_size"], title, upper_frequency,
            legacy_figure_dir / filename, floor_db=-75.0,
        )
        slug = (
            "humpback-whale" if "humpback" in lower_name
            else "snapping-shrimp"
        )
        _plot_case_spectrogram_grid(
            payload["case"], payload["fs"], payload["n_fft"],
            payload["hop_size"], title, upper_frequency,
            figure_dir / f"{slug}_spectrogram_comparison.png",
            floor_db=-75.0,
        )
    legacy_rows = _audit_legacy_result_csvs(_REPOSITORY_ROOT / "results")

    _write_csv(output_dir / "synthetic_ablation.csv", rows)
    _write_csv(output_dir / "runtime_scaling.csv", runtime_rows)
    _write_csv(output_dir / "real_audio.csv", real_rows)
    _write_csv(output_dir / "joint_renyi3.csv", joint_renyi_rows)
    _write_csv(
        output_dir / "joint_renyi3_snr_summary.csv",
        joint_renyi_snr_summary,
    )
    _write_csv(output_dir / "legacy_results_inventory.csv", legacy_rows)

    summary = {
        "configuration": {
            "seeds": list(seeds),
            "snr_db": list(snrs) + ["inf"],
            "n_fft": 1024,
            "hop_size": 32,
            "beta": 1.5,
            "candidate_grid": "H, 2H, 4H, ... up to NFFT/2",
            "implementation": "universal_stft/numpy/stft.py",
        },
        "clean_audit": clean_case["audit"],
        "regret_suite": {
            record["slug"]: {
                "final_average_regret": record["final_average_regret"],
                "final_average_path_code": record["final_average_path_code"],
                "posterior_replay_error": record["posterior_replay_error"],
                "reconstruction_mse": record["reconstruction_mse"],
            }
            for record in regret_records
        },
        "joint_renyi3": {
            record["slug"]: {
                "region": {
                    "time_start_s": record["time_start_s"],
                    "time_end_s": record["time_end_s"],
                    "frames": record["num_frames"],
                    "full_frequency_cells_per_frame": record["n_fft"],
                    "total_cells": record["num_frames"] * record["n_fft"],
                },
                "best_fixed_tracking_loss_length": record["best_fixed_length"],
                "joint_optimal_fixed_length": record[
                    "joint_optimal_fixed_length"
                ],
                "joint_optimal_fixed_h3_bits": record[
                    "joint_optimal_fixed_h3_bits"
                ],
                "fixed_selection_rule": record["fixed_selection_rule"],
                "methods": {
                    method: {
                        "h3_bits": float(metric["entropy_bits"][-1]),
                        "h3_normalized_by_log2_cells": float(
                            metric["normalized_entropy"][-1]
                        ),
                        "effective_cells_2_to_h3": float(
                            metric["effective_cells"][-1]
                        ),
                        "pmf_sum_error": float(metric["pmf_sum_error"]),
                    }
                    for method, metric in record["metrics"].items()
                },
            }
            for record in joint_renyi_representative_records
        },
        "joint_renyi3_snr_summary": joint_renyi_snr_summary,
        "median_metrics": {},
    }
    for spacing in ("geometric", "linear"):
        for method in ("trellis", "greedy", "best_fixed"):
            selected = [row for row in rows if row["spacing"] == spacing
                        and row["method"] == method]
            summary["median_metrics"][f"{spacing}:{method}"] = {
                key: _median(selected, key) for key in (
                    "normalized_renyi3", "mean_framewise_excess",
                    "tone_separation_db", "transient_width_ms", "transient_peak_delay_ms",
                    "trajectory_variation_octaves_per_s", "switches_per_s",
                    "framewise_length_error_octaves", "reconstruction_mse"
                )
            }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    joint_renyi_audit = {
        "definition": {
            "order": 3,
            "primary_measure": "H3(P) = -0.5 log2(sum P^3), in bits",
            "density": (
                "one normalized positive spectrogram-power PMF over the "
                "post-warm-up full time-frequency region"
            ),
            "rfft_to_full_frequency": (
                "duplicate interior bins via multiplicity 2; DC and Nyquist "
                "have multiplicity 1"
            ),
            "prefix_construction": (
                "one cumulative sum of raw power and raw cubed power; no "
                "cumulative sum of entropy values"
            ),
            "secondary_measure": "H3(P) / log2(number of full-grid cells)",
            "interpretation": (
                "lower means more concentrated; minima are asserted only "
                "within the explicitly compared finite family"
            ),
        },
        "representative_cases": summary["joint_renyi3"],
        "controlled_snr_summary": joint_renyi_snr_summary,
    }
    (output_dir / "joint_renyi3_summary.json").write_text(
        json.dumps(joint_renyi_audit, indent=2)
    )
    _plot_experiment_summary(
        rows, clean_case, posterior_case, runtime_rows, figure_dir, real_cases
    )
    _plot_regret_suite(regret_records, figure_dir)
    _plot_joint_renyi3_audit(
        joint_renyi_representative_records,
        joint_renyi_snr_summary,
        figure_dir,
    )
    if not quick:
        _plot_controlled_snr_joint_renyi3(
            joint_renyi_snr_summary,
            figure_dir / "controlled_snr_joint_renyi3.png",
        )
    joint_by_slug = {
        record["slug"]: record
        for record in joint_renyi_representative_records
    }
    for record in regret_records:
        slug = record["slug"]
        if slug == "hyperbolic-chirp":
            _plot_chirp_score_posterior(
                chirp_case, chirp_fs, 128,
                figure_dir / f"{slug}_trajectory_comparison.png",
            )
            _plot_case_loss_regret_comparison(
                record, figure_dir / f"{slug}_loss_regret_comparison.png",
                plain_scientific=True,
            )
            _plot_chirp_joint_renyi3_comparison(
                joint_by_slug[slug],
                figure_dir / f"{slug}_joint_renyi3_comparison.png",
            )
            _plot_chirp_diagnostics(
                chirp_case, record, joint_by_slug[slug], chirp_fs, 128,
                figure_dir / f"{slug}_diagnostics.png",
            )
        else:
            _plot_case_trajectory_comparison(
                record, figure_dir / f"{slug}_trajectory_comparison.png",
                posterior_only=(slug == "humpback-whale"),
            )
            _plot_case_loss_regret_comparison(
                record, figure_dir / f"{slug}_loss_regret_comparison.png"
            )
            _plot_case_joint_renyi3_comparison(
                joint_by_slug[slug],
                figure_dir / f"{slug}_joint_renyi3_comparison.png",
            )
    _plot_controlled_walkthrough(
        x_clean, metadata, clean_case,
        figure_dir / "controlled_walkthrough.png",
    )
    _plot_controlled_evidence(
        x_clean, metadata, clean_case,
        figure_dir / "controlled_evidence.png",
    )
    _plot_field_spectrogram_comparison(
        real_cases, figure_dir / "field_spectrogram_comparison.png",
    )
    if not quick:
        _require_canonical_paper_figures(figure_dir)
    if not quick:
        _write_audit_report(
            output_dir / "EVIDENCE_AUDIT.md", rows, clean_case, runtime_rows,
            real_rows, regret_records, legacy_rows, joint_renyi_rows,
            joint_renyi_snr_summary,
        )
    print(f"Wrote experiment data to {output_dir}")
    print(f"Wrote paper-ready audit figures to {figure_dir}")
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-experiments", action="store_true",
                        help="run the reproducible paper audit")
    parser.add_argument("--quick", action="store_true",
                        help="run a small smoke-test subset")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="override the default results/paper directory")
    parser.add_argument("--wav-dir", type=Path, default=None,
                        help="directory containing the three paper WAV files")
    parser.add_argument("--figure-dir", type=Path, default=None,
                        help="override the default output-dir/figures/audit directory")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.run_experiments:
        run_paper_experiments(
            arguments.output_dir,
            quick=arguments.quick,
            wav_dir=arguments.wav_dir,
            figure_dir=arguments.figure_dir,
        )
