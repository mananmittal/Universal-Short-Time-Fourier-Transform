"""Tests for the PyTorch backend (universal_stft.torch.stft)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from universal_stft.torch import stft as T
from conftest import CONFIGS


def _transform(x, mode="filtering", n_fft=256, hop=32, beta=1.5):
    return T.universal_short_time_fourier_transform(
        x, n_fft=n_fft, hop_size=hop, beta=beta, mode=mode)


@pytest.mark.parametrize(
    "n_fft, hop, expected",
    [(256, 32, [32, 64, 128]), (512, 32, [32, 64, 128, 256]), (300, 32, [32, 64, 128])],
)
def test_candidate_lengths_are_strictly_dyadic(n_fft, hop, expected):
    assert np.array_equal(T._candidate_lengths(n_fft, hop), expected)


@pytest.mark.parametrize(
    "real_dtype, complex_dtype",
    [(torch.float32, torch.complex64), (torch.float64, torch.complex128)],
)
def test_output_dtype_tracks_real_input(signal_1d, real_dtype, complex_dtype):
    x = torch.tensor(signal_1d, dtype=real_dtype)
    out = _transform(x)
    assert out.analysis_spectrogram.dtype == complex_dtype
    assert out.filtering_spectrogram.dtype == complex_dtype
    assert out.all_candidate_spectrograms.dtype == complex_dtype
    assert out.pmf_history.dtype == real_dtype
    rec = T.inverse_universal_spectrogram(out.filtering_spectrogram, len(x), n_fft=256, hop_size=32)
    assert rec.dtype == real_dtype


@pytest.mark.parametrize("n_fft, L", [(8, 4), (32, 16), (128, 64)])
def test_randomized_one_sided_scores_respect_bound(n_fft, L):
    generator = torch.Generator().manual_seed(n_fft)
    buffers = torch.zeros((128, n_fft), dtype=torch.float32)
    buffers[:, -L:] = torch.randn((len(buffers), L), generator=generator)
    scores = T.calculate_physics_loss_renyi(buffers, torch.fft.rfft(buffers, dim=-1), float(L))[0]
    assert torch.all(scores >= T.normalized_renyi3_lower_bound(n_fft) - 2e-6)


def test_analysis_mode_skips_only_the_companion(signal_1d):
    x = torch.tensor(signal_1d)
    fast = _transform(x, mode="analysis")
    full = _transform(x, mode="filtering")
    assert fast.filtering_spectrogram is None
    for name in fast._fields:
        if name in ("filtering_spectrogram", "blend_info"):
            continue
        assert torch.equal(getattr(fast, name), getattr(full, name)), name


@pytest.mark.parametrize("n_fft, hop, beta", CONFIGS)
def test_perfect_reconstruction_1d(signal_1d, n_fft, hop, beta):
    x = torch.tensor(signal_1d, dtype=torch.float32)
    out = _transform(x, n_fft=n_fft, hop=hop, beta=beta)
    x_rec = T.inverse_universal_spectrogram(
        out.filtering_spectrogram, len(signal_1d), hop_size=hop, n_fft=n_fft)
    assert torch.mean((x - x_rec) ** 2).item() < 1e-8


def test_batched_matches_single(signal_1d):
    x2 = torch.tensor(np.stack([signal_1d, np.roll(signal_1d, 40)]))
    batched = _transform(x2)
    num_frames = int(np.ceil(x2.shape[1] / 32))
    assert batched.analysis_spectrogram.shape == (2, num_frames, 129)
    assert batched.blend_info[0][1].shape == (2, 3)
    for row in range(2):
        single = _transform(x2[row])
        assert torch.allclose(batched.analysis_spectrogram[row], single.analysis_spectrogram, atol=1e-12)
        assert torch.allclose(batched.filtering_spectrogram[row], single.filtering_spectrogram, atol=1e-12)


def test_multichannel(signal_multi):
    ref = 0
    xm = torch.tensor(signal_multi)
    out = T.multichannel_stft(xm, reference_sensor_id=ref, n_fft=256, beta=1.5, hop_size=32,
                              mode="filtering")
    assert out.analysis_spectrogram.shape[0] == signal_multi.shape[0]
    single = _transform(xm[ref], mode="analysis")
    assert torch.allclose(out.analysis_spectrogram[ref], single.analysis_spectrogram, atol=1e-12)
    for ch in range(signal_multi.shape[0]):
        rec = T.inverse_universal_spectrogram(
            out.filtering_spectrogram[ch], signal_multi.shape[1], hop_size=32, n_fft=256)
        assert torch.mean((xm[ch] - rec) ** 2).item() < 1e-24
    assert T.multichannel_stft(xm, ref, n_fft=256, hop_size=32).filtering_spectrogram is None


@pytest.mark.parametrize("mode", ["analysis", "filtering"])
def test_differentiable(signal_1d, mode):
    """Gradients flow through the posterior blend (and the envelope) to the input."""
    x = torch.tensor(signal_1d[:600], requires_grad=True)
    out = _transform(x, mode=mode)
    loss = (out.analysis_spectrogram.abs() ** 2).sum()
    if mode == "filtering":
        loss = loss + (out.filtering_spectrogram.abs() ** 2).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_known_trajectory_reproduces_the_transform(signal_1d):
    x = torch.tensor(signal_1d)
    out = _transform(x)
    analysis, filtering = T.blend_guided_stft(x, out.blend_info, n_fft=256, hop_size=32)
    assert torch.allclose(analysis, out.analysis_spectrogram, atol=1e-12)
    assert torch.allclose(filtering, out.filtering_spectrogram, atol=1e-12)


def _reference_streams(x, trajectory, n_fft, hop_size):
    """Plain two-pass definition: whole envelope first, then divide and window."""
    x_padded, pad_len, num_frames = T._padded_signal(x, n_fft, hop_size)
    lengths, bank = T._window_bank(n_fft, hop_size, x.dtype, x.device)
    windows = [trajectory[m][1] @ bank for m in range(num_frames)]
    envelope = torch.zeros_like(x_padded)
    for m, window in enumerate(windows):
        end = pad_len + (m + 1) * hop_size
        envelope = envelope + torch.nn.functional.pad(
            window, (end - n_fft, x_padded.shape[1] - end))
    covered = envelope > 1e-9
    flat = torch.where(covered, x_padded / torch.where(covered, envelope, 1.0), 0.0)
    analysis, filtering = [], []
    for m, window in enumerate(windows):
        end = pad_len + (m + 1) * hop_size
        analysis.append(torch.fft.rfft(x_padded[:, end - n_fft:end] * window, dim=-1))
        filtering.append(torch.fft.rfft(flat[:, end - n_fft:end] * window, dim=-1))
    return torch.stack(analysis, 1), torch.stack(filtering, 1)


def test_streaming_matches_two_pass_definition_with_gradients():
    n_fft, hop_size, num_samples = 64, 16, 160
    lengths = T._candidate_lengths(n_fft, hop_size)
    num_frames = int(np.ceil(num_samples / hop_size))
    generator = torch.Generator().manual_seed(41)
    x_data = torch.randn((2, num_samples), generator=generator, dtype=torch.float64)
    logits_data = torch.randn((num_frames, 2, len(lengths)), generator=generator, dtype=torch.float64)

    def evaluate(function):
        x = x_data.clone().requires_grad_()
        logits = logits_data.clone().requires_grad_()
        trajectory = [(lengths, torch.softmax(logits[m], dim=-1)) for m in range(num_frames)]
        analysis, filtering = function(x, trajectory, n_fft, hop_size)
        loss = analysis.abs().square().mean() + filtering.abs().square().mean()
        return (analysis, filtering) + torch.autograd.grad(loss, (x, logits))

    streamed = evaluate(T.blend_guided_stft)
    reference = evaluate(_reference_streams)
    for a, b in zip(streamed, reference):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-10)


def test_inverse_preserves_gradient():
    n_fft, hop_size, num_frames, max_length = 64, 16, 7, 32
    num_samples = num_frames * hop_size
    spectra_data = torch.complex(torch.randn((2, num_frames, 33), dtype=torch.float64),
                                 torch.randn((2, num_frames, 33), dtype=torch.float64))

    def reference_inverse(spectra):
        pad_len = n_fft
        buffer_len = max(num_samples + 2 * pad_len, num_frames * hop_size + pad_len + n_fft)
        out = 0
        for m in range(num_frames):
            vector = torch.roll(torch.fft.irfft(spectra[:, m], n=n_fft, dim=-1), max_length, -1)
            start = (m + 1) * hop_size - max_length + pad_len
            out = out + torch.nn.functional.pad(vector, (start, buffer_len - start - n_fft))
        return out[:, pad_len:pad_len + num_samples]

    results = []
    for inverse in (lambda s: T.inverse_universal_spectrogram(s, num_samples, hop_size, n_fft),
                    reference_inverse):
        spectra = spectra_data.clone().requires_grad_()
        y = inverse(spectra)
        results.append((y, torch.autograd.grad(y.square().mean(), spectra)[0]))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-12)
    assert torch.allclose(results[0][1], results[1][1], atol=1e-12)
