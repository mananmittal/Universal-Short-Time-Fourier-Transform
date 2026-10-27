"""The NumPy and PyTorch backends must agree (to float32 precision)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from universal_stft.numpy import stft as N
from universal_stft.torch import stft as T
from conftest import CONFIGS


def _rel(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)


def test_dyadic_transition_matrices_match():
    lengths = np.array([32, 64, 128, 256])
    numpy_transitions = N._run_length_transition_matrix(lengths, hop_size=32)
    torch_transitions = T._run_length_transition_matrix(
        lengths, hop_size=32, dtype=torch.float64, device=torch.device("cpu")
    ).numpy()
    assert np.allclose(torch_transitions, numpy_transitions, atol=1e-7)
    assert np.allclose(torch_transitions.sum(axis=1), 1.0, atol=1e-12)


def test_one_sided_renyi_costs_and_bound_match():
    n_fft, L = 128, 64
    rng = np.random.default_rng(7)
    buffers = np.zeros((3, n_fft), dtype=np.float32)
    buffers[0, -L:] = rng.standard_normal(L)
    buffers[1, -1] = 1.0
    buffers[2, -L:] = np.cos(2 * np.pi * 7 * np.arange(L) / L)

    torch_buffers = torch.tensor(buffers)
    torch_fft = torch.fft.rfft(torch_buffers, dim=-1)
    torch_cost = T.calculate_physics_loss_renyi(
        torch_buffers, torch_fft, torch.full((3,), float(L))
    )[0].numpy()
    numpy_cost = np.array([
        N.calculate_physics_loss_renyi(row, np.fft.rfft(row), L)[0]
        for row in buffers
    ])

    assert np.allclose(torch_cost, numpy_cost, atol=2e-6)
    assert T.normalized_renyi3_lower_bound(n_fft) == N.normalized_renyi3_lower_bound(n_fft)
    assert np.all(torch_cost >= N.normalized_renyi3_lower_bound(n_fft) - 2e-6)


@pytest.mark.parametrize("n_fft, hop, beta", CONFIGS)
def test_backends_match(signal_1d, n_fft, hop, beta):
    no = N.universal_short_time_fourier_transform(signal_1d, n_fft=n_fft, hop_size=hop, beta=beta, mode="filtering")
    to = T.universal_short_time_fourier_transform(
        torch.tensor(signal_1d, dtype=torch.float32), n_fft=n_fft, hop_size=hop, beta=beta,
        mode="filtering",
    )

    A_n, F_n = no[0], no[1]
    A_t, F_t = to[0].numpy(), to[1].numpy()

    # The discrete window-length trajectory must be exactly identical.
    assert np.array_equal(no[2], to[2].numpy())

    # The complex spectrograms agree to float32 precision.
    assert _rel(A_n, A_t) < 1e-4
    assert _rel(F_n, F_t) < 1e-4


def test_reconstruction_matches(signal_1d):
    no = N.universal_short_time_fourier_transform(signal_1d, n_fft=256, hop_size=32, beta=1.5, mode="filtering")
    to = T.universal_short_time_fourier_transform(
        torch.tensor(signal_1d, dtype=torch.float32), n_fft=256, hop_size=32, beta=1.5,
        mode="filtering",
    )
    mL = int(np.max([np.max(i[0]) for i in no[4]]))
    xr_n = N.inverse_universal_spectrogram(no[1], len(signal_1d), 32, 256, max_L=mL)
    xr_t = T.inverse_universal_spectrogram(
        to[1], num_original_samples=len(signal_1d), hop_size=32, n_fft=256, max_L=mL
    ).numpy()
    assert _rel(xr_n, xr_t) < 1e-4


def test_float64_backend_preserves_precision_and_matches_numpy(signal_1d):
    x = np.asarray(signal_1d, dtype=np.float64)
    no = N.universal_short_time_fourier_transform(
        x, n_fft=256, hop_size=32, beta=1.5, mode="filtering"
    )
    to = T.universal_short_time_fourier_transform(
        torch.from_numpy(x), n_fft=256, hop_size=32, beta=1.5, mode="filtering"
    )

    assert to[0].dtype == torch.complex128
    assert to[1].dtype == torch.complex128
    assert to[-1].dtype == torch.complex128
    assert to[3].dtype == torch.float64
    assert np.array_equal(no[2], to[2].numpy())
    assert _rel(no[0], to[0].numpy()) < 1e-7
    assert _rel(no[1], to[1].numpy()) < 1e-7
    assert np.allclose(no[3], to[3].numpy(), atol=1e-7)

    max_L = int(to[4][0][0][-1])
    reconstructed = T.inverse_universal_spectrogram(
        to[1], len(x), n_fft=256, hop_size=32, max_L=max_L
    )
    assert reconstructed.dtype == torch.float64
    assert torch.mean((torch.from_numpy(x) - reconstructed) ** 2).item() < 1e-24


def test_backends_return_the_same_fields():
    x = np.random.default_rng(3).standard_normal(700)
    no = N.universal_short_time_fourier_transform(x, n_fft=128, hop_size=16)
    to = T.universal_short_time_fourier_transform(torch.from_numpy(x), n_fft=128, hop_size=16)
    assert type(no) is type(to)
    for name in no._fields:
        a, b = getattr(no, name), getattr(to, name)
        if name == "blend_info":
            assert all(np.allclose(wa, wb) for (_, wa), (_, wb) in zip(a, b))
        elif a is None:
            assert b is None
        else:
            assert np.allclose(a, b.numpy(), atol=1e-12), name
