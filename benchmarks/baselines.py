"""
Baseline time-frequency transforms used to benchmark the Universal STFT:
a fixed-window causal STFT (and its inverse) and a Constant-Q Transform. Only
``get_cqt`` needs the optional ``librosa`` dependency, and it is imported lazily
inside that function.
"""

import numpy as np
from scipy import signal


def get_cqt(x, fs, hop, n_bins=512):
    """
    Constant-Q Transform baseline computed natively on n_bins.
    Returns:
        C: Complex CQT matrix
        x_rec: Reconstructed signal
        freqs: Frequency vector for the CQT bins
    """
    import librosa
    fmin = librosa.note_to_hz('C1')
    nyquist = fs / 2.0
    
    # Dynamically calculate bins_per_octave to fit exactly n_bins before Nyquist
    octaves = np.log2(nyquist / fmin)
    bins_per_octave = int(np.ceil(n_bins / octaves))
    
    # Recalculate max freq to ensure it doesn't exceed Nyquist
    max_cqt_freq = fmin * (2.0 ** ((n_bins - 1) / bins_per_octave))
    while max_cqt_freq >= nyquist:
        bins_per_octave += 1
        max_cqt_freq = fmin * (2.0 ** ((n_bins - 1) / bins_per_octave))
        
    C = librosa.cqt(x, sr=fs, hop_length=hop, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    x_rec = librosa.icqt(C, sr=fs, hop_length=hop, fmin=fmin, bins_per_octave=bins_per_octave)
    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    
    return C, x_rec, freqs

def fixed_causal_spectrogram(x, L=128, n_fft=256, hop_size=32):
    num_samples = len(x)
    num_bins = n_fft // 2 + 1
    num_frames = int(np.ceil(num_samples / hop_size))

    # 1. Use the EXACT same padding structure as Universal
    pad_len = n_fft
    buffer_len = max(num_samples + 2 * pad_len, num_frames * hop_size + pad_len + n_fft)
    x_padded = np.zeros(buffer_len)
    x_padded[pad_len:pad_len + num_samples] = x

    spectrogram = np.zeros((num_frames, num_bins), dtype=complex)
    chosen_lengths = np.full(num_frames, L)

    # 2. Pre-compute the window exactly as Universal does: normalize by sum
    win = signal.windows.gaussian(L, std=L / 6.0)
    win /= np.sum(win)

    for frame_idx in range(num_frames):
        # 3. Use the exact same time indexing
        current_time_padded = pad_len + (frame_idx + 1) * hop_size
        
        # Pull the segment from the padded array
        segment = x_padded[current_time_padded - L: current_time_padded]

        # Pad to n_fft and multiply by window
        padded = np.zeros(n_fft)
        padded[-L:] = segment * win
        
        spectrogram[frame_idx, :] = np.fft.rfft(padded)

    return spectrogram, chosen_lengths

def inverse_spectrogram(spectrogram, num_original_samples, chosen_lengths, hop_size=32, n_fft=256):
    """
    Inverse transformation for fixed_causal_spectrogram
    Dynamically normalizes by the accumulated analysis window envelope (handles non-COLA).
    """
    num_frames = spectrogram.shape[0]
    pad_len = n_fft
    buffer_len = max(num_original_samples + 2 * pad_len, num_frames * hop_size + pad_len + n_fft)
    x_rec_padded = np.zeros(buffer_len)
    envelope_padded = np.zeros(buffer_len)

    for frame_idx in range(num_frames):
        current_time_padded = pad_len + (frame_idx + 1) * hop_size
        L = int(chosen_lengths[frame_idx])

        time_vector = np.fft.irfft(spectrogram[frame_idx, :], n=n_fft)

        # Roll to align data (same as inverse_universal_spectrogram)
        time_vector = np.roll(time_vector, L)

        # Build and roll the analysis window identically
        win = signal.windows.gaussian(L, std=L / 6.0)
        win /= np.sum(win)
        full_win = np.zeros(n_fft)
        full_win[-L:] = win
        full_win = np.roll(full_win, L)

        # Overlap-add the full rolled buffer
        start_idx = current_time_padded - L
        end_idx = start_idx + n_fft

        x_rec_padded[start_idx:end_idx] += time_vector
        envelope_padded[start_idx:end_idx] += full_win

    # Extract the valid signal range
    x_rec = x_rec_padded[pad_len: pad_len + num_original_samples]
    envelope = envelope_padded[pad_len: pad_len + num_original_samples]

    # Divide out the window envelope
    valid_mask = envelope > 1e-12
    x_rec[valid_mask] /= envelope[valid_mask]
    return x_rec

