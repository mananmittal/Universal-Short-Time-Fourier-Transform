import numpy as np

def renyi_entropy_2d(S, alpha=3.0):
    mag_sq = np.abs(S) ** 2
    mag_sq /= (np.sum(mag_sq) + 1e-12)
    return (1.0 / (1.0 - alpha)) * np.log2(np.sum(mag_sq ** alpha) + 1e-12)

def running_2d_renyi(S, alpha=3.0):
    """
    Compute the cumulative 2-D Rényi entropy used in benchmark reports.
    """
    mag_sq = np.abs(S) ** 2
    E = np.sum(mag_sq, axis=1)
    A = np.sum(mag_sq ** alpha, axis=1)

    cum_E = np.cumsum(E) + 1e-12
    cum_A = np.cumsum(A) + 1e-12

    return (1.0 / (1.0 - alpha)) * np.log2(cum_A / (cum_E ** alpha))

def sparsity_measure(S):
    mag = np.abs(S)
    return np.sum(mag) / (np.sqrt(np.sum(mag ** 2)) + 1e-12)

def envelope_spectrum_kurtosis(S):
    env = np.sum(np.abs(S) ** 2, axis=1)
    env_spec = np.abs(np.fft.rfft(env))
    mu = np.mean(env_spec)
    std = np.std(env_spec) + 1e-12
    return np.mean(((env_spec - mu) / std) ** 4)

def phase_jitter(x_orig, x_rec):
    from scipy.signal import hilbert
    trim = min(256, len(x_orig) // 10)
    if trim == 0: trim = 1
    phase_o = np.unwrap(np.angle(hilbert(x_orig[trim:-trim])))
    phase_r = np.unwrap(np.angle(hilbert(x_rec[trim:-trim])))
    return np.std(phase_r - phase_o)

def spectral_convergence(x_orig, x_rec):
    X_orig = np.abs(np.fft.rfft(x_orig))
    X_rec = np.abs(np.fft.rfft(x_rec))
    return np.linalg.norm(X_orig - X_rec) / (np.linalg.norm(X_orig) + 1e-12)
