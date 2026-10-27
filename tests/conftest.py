"""Shared fixtures for the Universal STFT test suite."""

import numpy as np
import pytest
from scipy.signal import chirp


# A short, deterministic, non-stationary test signal: a hyperbolic chirp plus a
# little broadband noise. Small enough that the whole suite runs in seconds.
FS = 2000
DURATION = 1.0


@pytest.fixture(scope="session")
def signal_1d():
    rng = np.random.default_rng(0)
    t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
    x = chirp(t, f0=800, t1=DURATION, f1=60, method="hyperbolic")
    x = x + 0.2 * rng.standard_normal(x.shape)
    return x.astype(np.float64)


@pytest.fixture(scope="session")
def signal_multi(signal_1d):
    # Three "sensors": the reference plus two delayed/scaled copies.
    return np.stack([signal_1d, np.roll(signal_1d, 17), 0.7 * np.roll(signal_1d, -23)])


# A couple of (n_fft, hop_size, beta) configurations exercised across tests.
CONFIGS = [
    (256, 32, 1.5),
    (128, 16, 0.5),
]
