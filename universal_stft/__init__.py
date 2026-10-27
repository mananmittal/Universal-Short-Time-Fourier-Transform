"""
Universal Short-Time Fourier Transform (U-STFT).

An STFT that picks its window length frame by frame. The NumPy backend is
available at the top level::

    from universal_stft import universal_short_time_fourier_transform

    result = universal_short_time_fourier_transform(x, n_fft=512, hop_size=32)
    result.analysis_spectrogram

For batched, differentiable, GPU-ready use, import the PyTorch backend::

    from universal_stft.torch import universal_short_time_fourier_transform
"""

from .numpy.stft import (
    USTFTResult,
    universal_short_time_fourier_transform,
    generate_analysis_spectrogram,
    generate_filtering_spectrogram,
    blend_guided_stft,
    inverse_universal_spectrogram,
    multichannel_stft,
    calculate_physics_loss_renyi,
    running_2d_renyi,
    normalized_renyi3_lower_bound,
)

__version__ = "1.0.0"

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
    "__version__",
]
