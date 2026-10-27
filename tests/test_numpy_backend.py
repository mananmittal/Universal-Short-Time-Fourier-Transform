"""Tests for the NumPy backend (universal_stft.numpy.stft)."""

import numpy as np
import pytest

from universal_stft.numpy import stft as N
from conftest import CONFIGS


@pytest.mark.parametrize(
    "n_fft, hop, expected",
    [
        (256, 32, [32, 64, 128]),
        (512, 32, [32, 64, 128, 256]),
        (300, 32, [32, 64, 128]),
        (128, 80, [80]),
    ],
)
def test_candidate_lengths_are_strictly_dyadic(n_fft, hop, expected):
    lengths = N._candidate_lengths(n_fft, hop)
    assert np.array_equal(lengths, expected)
    if len(lengths) > 1:
        assert np.array_equal(lengths[1:], 2 * lengths[:-1])
    assert lengths[-1] <= max(n_fft // 2, hop)


def test_dyadic_trellis_uses_physical_run_length():
    lengths = np.array([32, 64, 128, 256])
    transitions = N._run_length_transition_matrix(lengths, hop_size=32)
    expected = np.array([
        [1 / 3, 1 / 3, 1 / 3],
        [13 / 54, 13 / 54, 14 / 27],
        [713 / 3402, 713 / 3402, 988 / 1701],
        [1 / 16, 15 / 16, 0],
    ])
    assert np.allclose(transitions, expected, atol=1e-7)
    assert np.allclose(transitions.sum(axis=1), 1.0, atol=1e-7)
    assert np.all(transitions >= 0)


def _direct_renyi3(pmf):
    return -0.5 * np.log2(np.sum(pmf ** 3))


@pytest.mark.parametrize("kind", ["random", "impulse", "tone"])
def test_normalized_renyi3_cost_obeys_one_sided_bound(kind):
    n_fft, L = 256, 128
    active = np.arange(L)
    if kind == "random":
        values = np.random.default_rng(42).standard_normal(L)
    elif kind == "impulse":
        values = np.zeros(L)
        values[-1] = 1.0
    else:
        values = np.cos(2 * np.pi * 11 * active / L)

    buffer = np.zeros(n_fft)
    buffer[-L:] = values
    rfft = np.fft.rfft(buffer)
    score, h_time, h_spec = N.calculate_physics_loss_renyi(buffer, rfft, L)

    p = np.abs(buffer) ** 2
    p /= p.sum()
    q = np.abs(rfft) ** 2
    q /= q.sum()
    expected_time = _direct_renyi3(p) / np.log2(L)
    expected_spec = _direct_renyi3(q) / np.log2(len(rfft))

    assert np.isclose(h_time, expected_time, atol=1e-12)
    assert np.isclose(h_spec, expected_spec, atol=1e-12)
    assert score >= N.normalized_renyi3_lower_bound(n_fft) - 1e-12


@pytest.mark.parametrize("n_fft", [2, 3, 8, 32, 256])
def test_normalized_renyi3_bound_closed_form_is_positive(n_fft):
    c = n_fft ** -0.5
    t = max(c, np.sqrt(2.0 + c) - 1.0)
    b_endpoint = 2.0 * t * min(1.0, 1.0 + c - t) / (1.0 + t)
    b_max = b_endpoint if n_fft == 2 else max(
        b_endpoint, ((1.0 + np.sqrt(2.0 / n_fft)) / 2.0) ** 2
    )
    expected = -np.log(b_max) / np.log(n_fft)
    assert N.normalized_renyi3_lower_bound(n_fft) == pytest.approx(expected)
    assert expected > 0


@pytest.mark.parametrize("n_fft, L", [(8, 4), (32, 16), (128, 64)])
def test_randomized_one_sided_scores_respect_bound(n_fft, L):
    rng = np.random.default_rng(n_fft)
    buffers = np.zeros((256, n_fft))
    buffers[:, -L:] = rng.standard_normal((len(buffers), L))
    scores = N.calculate_physics_loss_renyi(
        buffers, np.fft.rfft(buffers, axis=-1), L
    )[0]
    assert np.all(scores >= N.normalized_renyi3_lower_bound(n_fft) - 1e-10)


def test_brute_force_two_point_scores_respect_bound():
    angles = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    buffers = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    scores = N.calculate_physics_loss_renyi(
        buffers, np.fft.rfft(buffers, axis=-1), L=2
    )[0]
    assert np.min(scores) >= N.normalized_renyi3_lower_bound(2) - 1e-10


def test_transform_size_validation():
    with pytest.raises(ValueError, match="hop_size"):
        N._candidate_lengths(256, 1)
    with pytest.raises(ValueError, match="exceed"):
        N._candidate_lengths(64, 128)
    with pytest.raises(ValueError, match="n_fft"):
        N.normalized_renyi3_lower_bound(1)


def _transform(x, mode="filtering", n_fft=256, hop=32, beta=1.5):
    return N.universal_short_time_fourier_transform(
        x, n_fft=n_fft, hop_size=hop, beta=beta, mode=mode)


@pytest.mark.parametrize("n_fft, hop, beta", CONFIGS)
def test_output_contract(signal_1d, n_fft, hop, beta):
    out = _transform(signal_1d, n_fft=n_fft, hop=hop, beta=beta)
    assert isinstance(out, N.USTFTResult) and len(out) == 12

    num_frames = int(np.ceil(len(signal_1d) / hop))
    num_bins = n_fft // 2 + 1
    lengths = out.blend_info[0][0]

    assert out.analysis_spectrogram.shape == (num_frames, num_bins)
    assert out.filtering_spectrogram.shape == (num_frames, num_bins)
    assert np.iscomplexobj(out.analysis_spectrogram)
    assert np.iscomplexobj(out.filtering_spectrogram)
    assert out.dominant_lengths.shape == (num_frames,)
    assert len(out.blend_info) == num_frames
    assert out.all_candidate_spectrograms.shape == (len(lengths), num_frames, num_bins)

    # Posterior rows are probability distributions over candidate lengths.
    assert out.pmf_history.shape == (num_frames, len(lengths))
    assert np.allclose(out.pmf_history.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(out.pmf_history >= -1e-12)
    assert np.all(np.isin(out.dominant_lengths, lengths))

    # Every available nonzero candidate respects the normalized bound.
    available = lengths[None, :] <= (np.arange(num_frames)[:, None] + 1) * hop
    costs = out.candidate_entropies
    assert np.all(costs[~available] == 0)
    assert np.all(np.isfinite(costs[available]))
    assert np.all(costs[available] >= N.normalized_renyi3_lower_bound(n_fft) - 1e-10)


def test_analysis_mode_is_the_same_pass_without_the_companion(signal_1d):
    fast = _transform(signal_1d, mode="analysis")
    full = _transform(signal_1d, mode="filtering")
    assert fast.filtering_spectrogram is None
    for name in fast._fields:
        if name in ("filtering_spectrogram", "blend_info"):
            continue
        assert np.array_equal(getattr(fast, name), getattr(full, name)), name
    with pytest.raises(ValueError, match="mode='filtering'"):
        N.inverse_universal_spectrogram(fast.filtering_spectrogram, len(signal_1d))
    with pytest.raises(ValueError, match="mode"):
        _transform(signal_1d, mode="fast")


def test_analysis_is_the_fft_under_the_effective_window(signal_1d):
    """Mixing candidate spectra equals one FFT with the mixed window."""
    out = _transform(signal_1d, mode="analysis")
    direct = N.generate_analysis_spectrogram(signal_1d, out.blend_info, n_fft=256, hop_size=32)
    assert np.allclose(direct, out.analysis_spectrogram, atol=1e-12)


@pytest.mark.parametrize("n_fft, hop, beta", CONFIGS)
def test_perfect_reconstruction(signal_1d, n_fft, hop, beta):
    out = _transform(signal_1d, n_fft=n_fft, hop=hop, beta=beta)
    x_rec = N.inverse_universal_spectrogram(
        out.filtering_spectrogram, len(signal_1d), hop_size=hop, n_fft=n_fft)
    assert np.mean((signal_1d - x_rec) ** 2) < 1e-24


def test_companion_frames_only_wait_a_fixed_number_of_frames(signal_1d):
    """Companion frame m depends on samples up to frame m + D - 1 only."""
    n_fft, hop = 256, 32
    delay = int(N._candidate_lengths(n_fft, hop)[-1] // hop) - 1
    full = _transform(signal_1d, n_fft=n_fft, hop=hop).filtering_spectrogram
    for cut_frame in (5, 20, 47):
        prefix = signal_1d[:(cut_frame + 1) * hop]
        partial = _transform(prefix, n_fft=n_fft, hop=hop).filtering_spectrogram
        ready = cut_frame - delay
        assert np.allclose(partial[:ready + 1], full[:ready + 1], atol=1e-12)


def test_determinism(signal_1d):
    a = _transform(signal_1d)
    b = _transform(signal_1d)
    assert np.array_equal(a.analysis_spectrogram, b.analysis_spectrogram)
    assert np.array_equal(a.filtering_spectrogram, b.filtering_spectrogram)


def test_known_trajectory_reproduces_the_transform(signal_1d):
    out = _transform(signal_1d)
    analysis, filtering = N.blend_guided_stft(signal_1d, out.blend_info, n_fft=256, hop_size=32)
    assert np.allclose(analysis, out.analysis_spectrogram, atol=1e-12)
    assert np.array_equal(filtering, out.filtering_spectrogram)


def test_filtering_with_a_common_response(signal_1d):
    """An all-pass gain is the identity; a real gain filters every frame alike."""
    out = _transform(signal_1d)
    ones = np.ones(out.filtering_spectrogram.shape[-1])
    x_same = N.inverse_universal_spectrogram(out.filtering_spectrogram * ones, len(signal_1d))
    assert np.allclose(x_same, signal_1d, atol=1e-12)


def test_multichannel_shapes_and_reference(signal_multi):
    ref = 0
    out = N.multichannel_stft(signal_multi, reference_sensor_id=ref, n_fft=256, beta=1.5,
                              hop_size=32, mode="filtering")
    num_sensors = signal_multi.shape[0]
    num_frames = int(np.ceil(signal_multi.shape[1] / 32))
    assert out.analysis_spectrogram.shape == (num_sensors, num_frames, 129)
    assert out.filtering_spectrogram.shape == (num_sensors, num_frames, 129)

    single = _transform(signal_multi[ref], mode="analysis")
    assert np.allclose(out.analysis_spectrogram[ref], single.analysis_spectrogram, atol=1e-12)

    for ch in range(num_sensors):
        rec = N.inverse_universal_spectrogram(
            out.filtering_spectrogram[ch], signal_multi.shape[1], hop_size=32, n_fft=256)
        assert np.mean((signal_multi[ch] - rec) ** 2) < 1e-24

    fast = N.multichannel_stft(signal_multi, reference_sensor_id=ref, n_fft=256, beta=1.5,
                               hop_size=32)
    assert fast.filtering_spectrogram is None
    assert np.allclose(fast.analysis_spectrogram, out.analysis_spectrogram, atol=1e-12)


def test_running_renyi_is_finite(signal_1d):
    analysis = _transform(signal_1d, mode="analysis").analysis_spectrogram
    r = N.running_2d_renyi(analysis)
    assert r.shape == (analysis.shape[0],)
    assert np.all(np.isfinite(r))


def test_signal_shorter_than_one_hop():
    out = _transform(np.ones(5))
    assert out.analysis_spectrogram.shape == (1, 129)
    assert np.allclose(N.inverse_universal_spectrogram(out.filtering_spectrogram, 5), 1.0)
