# Contributing

Thanks for helping improve the U-STFT. Bug reports, questions, examples and
fixes are all welcome.

## Report a problem or ask a question

Open an issue at
<https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/issues>.
For a bug, please include:

- a short script that reproduces it (a synthetic signal is fine);
- the full error message or a description of the wrong output;
- your Python, NumPy, SciPy and (if used) PyTorch versions, and the
  `universal_stft.__version__`.

Check the [FAQ](README.md#faq-and-troubleshooting) first: it covers the most
common surprises (inverting the wrong stream, mismatched `n_fft`/`hop_size`,
memory on long recordings).

## Set up and run the tests

```bash
git clone https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform.git
cd Universal-Short-Time-Fourier-Transform
python -m pip install -e ".[dev,examples]"
python -m pytest
python examples/example.py
python examples/example_torch.py
```

The same test suite runs on GitHub Actions for Python 3.10 to 3.13 on every push
and pull request.

## Propose a change

1. Fork the repository and create a branch.
2. Make the change, and add or update a test in `tests/` when behaviour changes.
3. Run `python -m pytest` and make sure it passes.
4. Open a pull request that says what changed and why.

Keep the NumPy (`universal_stft/numpy/stft.py`) and PyTorch
(`universal_stft/torch/stft.py`) backends in step: same function names,
arguments, output fields and field order. The PyTorch backend must keep the
input's device, gradients, batching and precision (`float32` to `complex64`,
`float64` to `complex128`). The paper scripts in `benchmarks/` import the
package directly, so there is a single implementation to change.

By contributing you agree that your contribution is released under the
[MIT License](LICENSE).
