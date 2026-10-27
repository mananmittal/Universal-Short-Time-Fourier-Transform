"""
Benchmark / paper-reproduction suite for the Universal Short Time Fourier Transform.

This package is *not* part of the importable library; it holds the evaluation
pipeline used to compare the adaptive STFT against fixed-window baselines and to
render the figures used in the paper.

Modules
-------
baselines        Fixed-window STFT and CQT baselines.
metrics          Reconstruction / sparsity / entropy metrics.
plotting         Shared spectrogram plotting helpers.
evaluate         ``run_full_evaluation`` - the unified benchmarking pipeline.
run_synthetic    Driver: catalog of synthetic modulation/chirp/impulse signals.
run_complex      Driver: 20 extreme "conflict" signals.
run_real_audio   Driver: full evaluation of the bundled chirp, or of WAV files you pass in.

Run a driver from the repository root, e.g.::

    python -m benchmarks.run_synthetic

Outputs (per-signal PNGs, a combined PDF report, and a metrics CSV) are written
under ``results/``.
"""
