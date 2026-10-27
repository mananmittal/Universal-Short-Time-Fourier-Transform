# Universal Short-Time Fourier Transform

[![tests](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/actions/workflows/tests.yml/badge.svg)](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/examples/quickstart.ipynb)
[![Project page](https://img.shields.io/badge/project-page-1f5fa8)](https://mananmittal.github.io/Universal-Short-Time-Fourier-Transform/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/LICENSE)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-2610.37795-b31b1b.svg)](https://arxiv.org/abs/2610.37795)

An STFT that chooses its window length as it goes, with matching NumPy and PyTorch implementations.

- **Try it in your browser:** the [Colab quickstart](https://colab.research.google.com/github/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/examples/quickstart.ipynb) installs the package and runs a full example in about a minute, nothing to set up.
- **Install:** `pip install git+https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform.git`
- **Paper:** [arXiv:2610.37795](https://arxiv.org/abs/2610.37795)
- **Project page with results:** https://mananmittal.github.io/Universal-Short-Time-Fourier-Transform/
- **Cite:** if you use the U-STFT, please cite the paper; BibTeX is in the [Citation](#citation) section.

## What it is

A normal STFT uses one window length for the whole signal. Short windows show
clicks and onsets sharply but blur frequency; long windows separate close
tones but blur time. The U-STFT keeps a bank of window lengths (`H, 2H, 4H, ...`)
and, at every frame, updates a probability for each one. The effective window
grows during steady tones and snaps back to the shortest window at transients.

Use it when your signal mixes steady and sudden parts (speech, music, animal
sounds, machinery) and no single window length suits all of it. It gives you a
spectrogram for plotting or as features, and, if you ask for it, a second
stream that can be filtered and turned back into a signal exactly.

**Contents:**
[Step 1 Install](#step-1-install) ·
[Step 2 First spectrogram](#step-2-your-first-spectrogram) ·
[Step 3 What it chose](#step-3-look-at-what-it-chose) ·
[Step 4 Filter and resynthesize](#step-4-filter-and-resynthesize) ·
[Step 5 PyTorch](#step-5-pytorch-batches-gpu-and-gradients) ·
[Step 6 Several channels](#step-6-several-channels) ·
[Choosing parameters](#choosing-parameters) ·
[How it works](#how-it-works) ·
[API](#api-reference) ·
[FAQ](#faq-and-troubleshooting) ·
[Reproduce the paper](#reproduce-the-paper) ·
[Data sources](#data-sources) ·
[Citation](#citation)

## Step 1: Install

You need Python 3.10 or newer and `pip`. The package depends only on NumPy and
SciPy; PyTorch is optional.

**Quickest: one line, no clone.**

```bash
python -m pip install "universal_stft[examples] @ git+https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform.git"
```

Leave out `[examples]` (which adds matplotlib and Jupyter for the plots below)
to install just the package:
`python -m pip install git+https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform.git`.

**From a clone**, if you want the examples, tests or paper scripts, or want to
change the code:

```bash
git clone https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform.git
cd Universal-Short-Time-Fourier-Transform
python -m pip install -e ".[examples]"    # the package plus matplotlib for the plots below
```

Other optional extras (from a clone):

```bash
python -m pip install -e ".[torch]"       # PyTorch backend (Step 5)
python -m pip install -e ".[benchmarks]"  # paper reproduction scripts
python -m pip install -e ".[dev]"         # tests
```

Check that it works:

```bash
python -c "import universal_stft; print(universal_stft.__version__)"
```

## Step 2: Your first spectrogram

Build a test signal with a steady part and a sudden part, then run the
transform. The default `mode="analysis"` makes one pass over the signal and
returns the adaptive spectrogram.

```python
import numpy as np
from universal_stft import universal_short_time_fourier_transform

# A test signal: 1 s of a steady 440 Hz tone, then 1 s of short clicks.
fs = 8000                                    # sample rate in Hz
t = np.arange(2 * fs) / fs                   # 2 seconds of time stamps
x = np.where(t < 1.0, np.sin(2 * np.pi * 440 * t), 0.0)
x[fs::800] += 1.0                            # a click every 0.1 s in the second half
x += 1e-3 * np.random.default_rng(0).standard_normal(len(x))   # a little noise

n_fft, hop_size = 512, 32
result = universal_short_time_fourier_transform(x, n_fft=n_fft, hop_size=hop_size)

S = result.analysis_spectrogram              # complex array, shape (frames, bins)
print(S.shape)                               # (500, 257)
```

`frames = ceil(len(x) / hop_size)` and `bins = n_fft // 2 + 1`. Each row is one
frame in time; each column is one frequency bin.

Plot it in decibels. Frame `m` ends at sample `(m + 1) * hop_size`, and bin `k`
is at frequency `k * fs / n_fft`:

```python
import matplotlib.pyplot as plt

frame_times = (np.arange(S.shape[0]) + 1) * hop_size / fs   # end time of each frame, seconds
freqs = np.arange(S.shape[1]) * fs / n_fft                    # bin frequencies, Hz
S_db = 20 * np.log10(np.abs(S.T) + 1e-12)                     # rows = frequency, columns = time

plt.imshow(S_db, origin="lower", aspect="auto", cmap="magma",
           extent=[frame_times[0], frame_times[-1], freqs[0], freqs[-1]],
           vmin=S_db.max() - 80, vmax=S_db.max())
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="dB")
plt.show()
```

You should see a thin horizontal line at 440 Hz for the first second (a long
window, sharp in frequency) and thin vertical lines for the clicks after that
(short windows, sharp in time).

## Step 3: Look at what it chose

The result also records which window it used. `dominant_lengths` is the most
likely window length at each frame. `pmf_history` is the full probability over
all window lengths at each frame (each row sums to 1). Continuing from Step 2:

```python
lengths = result.blend_info[0][0]            # the window bank: [ 32  64 128 256]
print(lengths)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.step(frame_times, result.dominant_lengths, where="post")
ax1.set_yscale("log", base=2)
ax1.set_yticks(lengths, [str(L) for L in lengths])
ax1.set_ylabel("Most likely window")

ax2.imshow(result.pmf_history.T, origin="lower", aspect="auto", cmap="Greys",
           interpolation="nearest",
           extent=[frame_times[0], frame_times[-1], -0.5, len(lengths) - 0.5])
ax2.set_yticks(range(len(lengths)), [str(L) for L in lengths])
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Posterior")
plt.show()
```

During the tone the window climbs to 256 samples within a few frames and stays
there. During the clicks it keeps dropping back to 32 or 64 samples. Window
lengths are in samples; divide by `fs` for seconds.

## Step 4: Filter and resynthesize

The spectrogram from Step 2 cannot be turned back into a signal, because its
window changes from frame to frame. For that, run with `mode="filtering"`. The
same pass then also builds a companion stream, `filtering_spectrogram`, that
inverts exactly.

```python
from universal_stft import inverse_universal_spectrogram

result = universal_short_time_fourier_transform(x, n_fft=n_fft, hop_size=hop_size,
                                                mode="filtering")
F = result.filtering_spectrogram             # same shape as the analysis spectrogram

# Unchanged coefficients give back the input.
x_hat = inverse_universal_spectrogram(F, len(x), n_fft=n_fft, hop_size=hop_size)
print(np.max(np.abs(x - x_hat)))             # below 1e-15

# Low-pass filter: keep bins below 1 kHz, remove the rest.
freqs = np.arange(F.shape[1]) * fs / n_fft
gain = (freqs < 1000).astype(float)          # one gain per frequency bin
x_low = inverse_universal_spectrogram(F * gain, len(x), n_fft=n_fft, hop_size=hop_size)
```

Three rules:

- Invert `filtering_spectrogram`, never `analysis_spectrogram`.
- Pass the same `n_fft` and `hop_size` to the inverse that you used for the
  transform.
- To filter, multiply every frame by the same frequency response (`F * gain`
  with `gain` of shape `(bins,)`). Changes that differ from frame to frame are
  outside the exact-reconstruction guarantee.

`x_low` keeps the 440 Hz tone and removes the energy of the clicks above 1 kHz.

## Step 5: PyTorch, batches, GPU and gradients

The PyTorch backend has the same function names, arguments and output fields.
Install it with `python -m pip install torch` (or `-e ".[torch]"` from a clone).

```python
import torch
from universal_stft.torch import universal_short_time_fourier_transform, inverse_universal_spectrogram

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(8, 16_000, device=device)    # (batch, samples); (samples,) also works

result = universal_short_time_fourier_transform(x, n_fft=512, hop_size=32)
print(result.analysis_spectrogram.shape)     # torch.Size([8, 500, 257])
print(result.analysis_spectrogram.dtype)     # torch.complex64 (float32 in, complex64 out)
print(result.pmf_history.shape)              # torch.Size([8, 500, 4])
```

- **Batches.** Input `(samples,)` or `(batch, samples)`. Batched input adds a
  leading batch axis to every output tensor.
- **Device and precision.** Outputs stay on the input's device. `float32` gives
  `complex64` and `float64` gives `complex128`.
- **Gradients.** The transform is differentiable, so it can be a layer inside a
  network:

```python
x = torch.randn(2, 4_000, requires_grad=True)
result = universal_short_time_fourier_transform(x, n_fft=256, hop_size=32)
loss = result.analysis_spectrogram.abs().pow(2).mean()
loss.backward()
print(x.grad.shape)                          # torch.Size([2, 4000])
```

- **No gradients needed?** Wrap calls in `torch.inference_mode()`. Filtering
  mode and the inverse work the same way as in NumPy:

```python
x = torch.randn(4, 16_000, dtype=torch.float64, device=device)
with torch.inference_mode():
    result = universal_short_time_fourier_transform(x, n_fft=512, hop_size=32, mode="filtering")
    x_hat = inverse_universal_spectrogram(result.filtering_spectrogram, x.shape[-1],
                                          n_fft=512, hop_size=32)
print((x - x_hat).abs().max())               # about 1e-15
```

For a full worked example with chirps and plots, run `python examples/example_torch.py`.

## Step 6: Several channels

For microphone or hydrophone arrays you usually want every channel analysed
with the same windows, so that phase differences between channels stay
meaningful. `multichannel_stft` picks the windows on one reference channel and
applies them to all channels. Continuing from Step 2 (NumPy):

```python
from universal_stft import multichannel_stft

# Two channels: the second is a delayed, quieter copy of the first.
x_multi = np.stack([x, 0.5 * np.roll(x, 3)])                 # shape (channels, samples)

result = multichannel_stft(x_multi, reference_sensor_id=0, n_fft=n_fft, hop_size=hop_size)
print(result.analysis_spectrogram.shape)     # (2, 500, 257)
print(result.dominant_lengths.shape)         # (500,): the reference channel's choices
```

The two spectrograms get a leading channel axis; the other fields describe the
reference channel. Pass `mode="filtering"` to get an invertible stream for each
channel. Note that `multichannel_stft` defaults to `beta=2.0`.

You can also reuse a window trajectory by hand. `blend_info` from one result
can be applied to any other signal of the same length:

```python
from universal_stft import generate_analysis_spectrogram

reference = universal_short_time_fourier_transform(x, n_fft=n_fft, hop_size=hop_size)
y = 0.5 * np.roll(x, 3)                      # another signal of the same length
S_y = generate_analysis_spectrogram(y, reference.blend_info, n_fft=n_fft, hop_size=hop_size)
print(S_y.shape)                             # (500, 257)
```

`generate_filtering_spectrogram` does the same for the invertible stream, and
`blend_guided_stft` returns both.

## Choosing parameters

There are three knobs. Both sizes are in samples and must be integers of at
least 2, with `hop_size <= n_fft`.

| Parameter | Default | What it controls |
|---|---|---|
| `hop_size` (`H`) | 32 | The step between frames **and** the shortest window. Smaller means finer time steps, more frames and more compute. |
| `n_fft` (`N`) | 256 | The FFT size, which sets the longest window: the bank is `H, 2H, 4H, ...` up to `max(N // 2, H)`. It also sets the number of bins, `N // 2 + 1`. |
| `beta` | 1.5 | How strongly the choice follows the per-frame scores. Larger values switch windows more readily; smaller values lean on the stay / grow / reset prior and give a smoother schedule. `beta=0` ignores the scores. |

Examples of the window bank:

| `n_fft` | `hop_size` | Windows (samples) |
|---|---|---|
| 256 | 32 | 32, 64, 128 |
| 512 | 32 | 32, 64, 128, 256 |
| 1024 | 64 | 64, 128, 256, 512 |
| 64 | 32 | 32 only (no adaptation) |

You can print the bank for any setting with `result.blend_info[0][0]`.

Sensible starting points:

1. Pick `hop_size` as the shortest window you want, for example about 4 ms:
   32 samples at 8 kHz, 64 at 16 kHz, 128 at 32 kHz.
2. Pick `n_fft` so that `n_fft // 2` is the longest window you want. Use
   `n_fft >= 4 * hop_size`, otherwise there is only one window and nothing to
   choose. `n_fft = 8 * hop_size` or `16 * hop_size` gives 3 or 4 windows.
3. Start with `beta` between 1 and 2. If the window flickers too much, lower it;
   if it reacts too slowly to transients, raise it.

Cost: each frame computes one FFT per window, so time and memory grow with the
number of frames and the number of windows. In filtering mode each companion
frame is finished `max_L / H - 1` frames after its analysis frame (`max_L` is
the longest window), for example 7 frames for `n_fft=512, hop_size=32`.

## How it works

At every frame (hop `H`, FFT size `N`):

1. Take the `N` most recent samples.
2. Window them with each candidate Gaussian window of length `H, 2H, 4H, ...`,
   all ending on the newest sample, and take the FFT. The longest window is the
   largest of these that is at most `max(N // 2, H)`. For `N=512, H=32` the bank
   is `[32, 64, 128, 256]`.
3. Score each candidate by its normalized order-3 Rényi entropy in time plus in
   frequency. Rényi entropy measures how spread out a distribution is; here it
   is computed on the normalized energy of the windowed frame (time) and of its
   one-sided FFT (frequency). Lower means more concentrated.
4. Turn the scores into a likelihood, `exp(-beta * (score - min) / std)`, and
   multiply by the prior to get a posterior over window lengths.
5. Output the posterior-weighted sum of the candidate spectra. By linearity of
   the FFT this equals one FFT with the effective window `sum_s posterior[s] * w_s`.
6. Move the posterior through the stay / grow / reset model to get the next prior.

Every step uses only the current and earlier samples, so the analysis is causal.

### Two modes

| | `mode="analysis"` (default) | `mode="filtering"` |
|---|---|---|
| Passes over the signal | one, causal | one, causal |
| Output | adaptive spectrogram + diagnostics | the same, plus the invertible companion stream |
| Delay | none | each companion frame is ready `max_L / H - 1` frames after its analysis frame |
| Use it for | visualization, features, neural networks | filtering and resynthesis |

The adaptive spectrogram is not directly invertible because its effective
window changes from frame to frame. The companion stream fixes that: it divides
the signal by the overlap envelope `E` (the sum of all effective windows placed
at their frames) and then applies the same effective windows. Overlap-adding the
inverse FFTs of those frames returns the input exactly. `E` at a sample is final
once every window that can reach it has been seen, so the companion frames are
built in the same pass, a fixed number of frames behind.

### Stay / grow / reset probabilities

The prior on window changes is defined one hop at a time: with a run of `u`
hops the window resets or stays with probability `1/(3u)` each, and grows by one
hop otherwise. The bank only keeps `H, 2H, 4H, ...`, so going from `L` to the next
length takes several growth steps in a row. With `r = L / H` and
`delta = (L_next - L) / H`:

```text
p_grow  = product over u = r, ..., r + delta - 1 of (1 - 2/(3u))
p_reset = p_stay = (1 - p_grow) / 2
```

The longest window cannot grow: it resets with probability `1/(2r)` and
otherwise stays. The product is computed with log-Gamma functions so long
steps do not underflow.

The paper's regret bound is proved for the ideal trellis, which keeps full
action counts for every path. This package implements the finite dyadic
version above; its exact guarantee is the path code of the realized finite
trellis, and the ideal asymptotic rate is not claimed for it.

### A floor on the score

No nonzero frame can be concentrated in one sample and one frequency bin at
once, so the score has a positive lower bound for every candidate:

```python
from universal_stft import normalized_renyi3_lower_bound
floor = normalized_renyi3_lower_bound(n_fft=512)
print(floor)                                 # about 0.16
```

It splits the rFFT bins into the end bins (one-dimensional real Fourier
subspaces) and interior bins (two-dimensional cosine/sine subspaces), bounds the
largest time and frequency probabilities, and applies `H3(p) >= -log(max p)`:

```text
c = N^(-1/2),  t = max(c, sqrt(2 + c) - 1)
B_end = 2t min(1, 1 + c - t) / (1 + t)
B_int = ((1 + sqrt(2/N)) / 2)^2                 (N > 2)
B(N)  = B_end for N = 2, otherwise max(B_end, B_int)
floor = -log(B(N)) / log(N)
```

## API reference

```python
universal_short_time_fourier_transform(x, n_fft=256, hop_size=32, beta=1.5, mode="analysis")
```

Returns a `USTFTResult` named tuple. Fields can be read by name or unpacked by
position, in this order:

| Field | Shape (single signal) | Meaning |
|---|---|---|
| `analysis_spectrogram` | `(frames, bins)` | adaptive spectrogram |
| `filtering_spectrogram` | `(frames, bins)` or `None` | invertible companion stream (`None` in analysis mode) |
| `dominant_lengths` | `(frames,)` | most probable window length |
| `pmf_history` | `(frames, states)` | posterior over window lengths |
| `blend_info` | list of `(lengths, weights)` | the trajectory, reusable on other signals |
| `candidate_entropies` | `(frames, states)` | score of each candidate (0 where unavailable) |
| `blended_entropies` | `(frames,)` | score of the effective-window frame |
| `blended_h_time`, `blended_h_spec` | `(frames,)` | its time and frequency parts |
| `candidate_h_time`, `candidate_h_spec` | `(frames, states)` | time and frequency parts per candidate |
| `all_candidate_spectrograms` | `(states, frames, bins)` | spectrum of every candidate |

`frames = ceil(len(x) / hop_size)`, `bins = n_fft // 2 + 1` and `states` is the
number of windows in the bank. The PyTorch backend adds a leading batch axis
for batched input. A candidate is "unavailable" in the first few frames, until
enough samples have arrived to fill it.

Other functions:

```python
generate_analysis_spectrogram(x, blend_info, n_fft=256, hop_size=32)
generate_filtering_spectrogram(x, blend_info, n_fft=256, hop_size=32)
blend_guided_stft(x, blend_info, n_fft=256, hop_size=32)        # both of the above
inverse_universal_spectrogram(filtering, num_samples, hop_size=32, n_fft=256)
multichannel_stft(x_multi, reference_sensor_id, n_fft=256, beta=2.0, hop_size=32, mode="analysis")
running_2d_renyi(spectrogram)
normalized_renyi3_lower_bound(n_fft)
```

| Function | What it does |
|---|---|
| `generate_analysis_spectrogram` | analysis spectrogram of `x` along a known trajectory `blend_info` |
| `generate_filtering_spectrogram` | invertible companion stream of `x` along a known trajectory |
| `blend_guided_stft` | returns `(analysis, filtering)` for a known trajectory |
| `inverse_universal_spectrogram` | overlap-add inverse of a filtering spectrogram |
| `multichannel_stft` | tracks the trajectory on one reference channel and applies it to every channel (`x_multi` is `(channels, samples)`) |
| `running_2d_renyi` | running Rényi-3 entropy of a spectrogram, used in the paper's regret plots |
| `normalized_renyi3_lower_bound` | the score floor described above |

NumPy functions are importable from `universal_stft`; the PyTorch versions,
with the same names, from `universal_stft.torch`.

## FAQ and troubleshooting

**`filtering_spectrogram` is `None`, or the inverse says "no filtering spectrogram".**
You ran the default `mode="analysis"`. Run again with `mode="filtering"`.

**I inverted the spectrogram and got noise or a badly distorted signal.**
Two common causes. (1) You inverted `analysis_spectrogram`; invert
`filtering_spectrogram` instead. (2) The inverse used different sizes from the
transform. `inverse_universal_spectrogram` defaults to `n_fft=256, hop_size=32`,
so always pass the same `n_fft` and `hop_size` you used for the transform.

**I edited the filtering spectrogram and the result is not what I expected.**
Exact reconstruction holds for unmodified coefficients, and filtering means
multiplying every frame by the same frequency response. Masks that change from
frame to frame will run, but are outside that guarantee.

**I run out of memory on long recordings.**
`all_candidate_spectrograms` stores the spectrum of every window at every frame:
`states x frames x bins` complex numbers (16 bytes each in NumPy). For 10 minutes
at 48 kHz with `n_fft=1024, hop_size=64` that is about 15 GB. Process long
recordings in pieces, use a larger `hop_size`, or use a smaller `n_fft`
(fewer windows and fewer bins).

**The window never changes.**
Check the bank with `result.blend_info[0][0]`. If it has only one length,
`n_fft` is too small for your `hop_size`; use `n_fft >= 4 * hop_size`. If it has
several lengths and the choice is still flat, try a larger `beta`.

**The first few frames always use short windows.**
That is expected. A window of length `L` needs `L` samples, so longer windows
become available only after `L / hop_size` frames. Until then their score is
reported as 0 in `candidate_entropies`.

**PyTorch: `'numpy.ndarray' object has no attribute 'dim'`.**
The PyTorch backend needs a tensor. Convert first, for example
`torch.as_tensor(x, dtype=torch.float32)`, or use the NumPy backend.

**Which mode should I use for training a network?**
`mode="analysis"`. It is one pass and differentiable. Use `mode="filtering"` only
when you need to resynthesize a signal.

**Is framewise selection a fair comparison?**
Picking the best-scoring window independently at each frame is a myopic
baseline, not the hindsight-schedule comparator used in the regret analysis.
The trellis smooths the window schedule; it is not meant to beat framewise
selection on that selector's own per-frame score.

## Tests and examples

```bash
python -m pip install -e ".[dev,examples]"
python -m pytest
python examples/example.py
python examples/example_torch.py
```

The `examples/` folder also has Jupyter notebook versions of both scripts, and
`examples/quickstart.ipynb`, the short notebook behind the "Open in Colab"
badge. Tests run on Python 3.10 to 3.13 on every push. See
[CONTRIBUTING.md](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/CONTRIBUTING.md)
to report a problem or propose a change.

## Repository layout

```text
universal_stft/
    _shared.py        window grid, transition table, frame geometry, result type
    numpy/stft.py     NumPy backend
    torch/stft.py     PyTorch backend
examples/             Colab quickstart, NumPy and PyTorch walk-throughs (scripts and notebooks)
benchmarks/           paper experiments, figure generation and audio inputs
tests/                pytest suite
docs/                 project web page (GitHub Pages)
```

## Reproduce the paper

The WAV inputs used in the paper are bundled and documented in
[`benchmarks/wav/README.md`](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/benchmarks/wav/README.md). From the repository root:

```bash
python -m pip install -e ".[benchmarks]"
python -m benchmarks.run_paper --run-experiments --quick   # short smoke run
python -m benchmarks.run_paper --run-experiments           # full run
```

The full run writes the CSV, JSON and figure set to `results/paper/`.
`--wav-dir PATH` points to another copy of the checksum-matched inputs. Smaller
drivers are `python -m benchmarks.run_synthetic`, `run_complex` and
`run_real_audio` (the bundled chirp by default, or WAV files you pass on the
command line); all outputs go under `results/`.

## Scope

- Analysis is causal and uses only the current and earlier samples.
- Exact reconstruction holds for unmodified companion coefficients. A single
  frequency response applied to every frame filters the signal; frame-dependent
  changes are outside that guarantee.
- The trellis smooths the window schedule. It is not meant to beat framewise
  selection on that selector's own per-frame score.

## Data sources

The paper's field recordings are 5 s excerpts from the NOAA Office of National
Marine Sanctuaries and U.S. Navy **SanctSound** passive acoustic collection:

> NOAA Office of National Marine Sanctuaries and U.S. Navy, *SanctSound Raw
> Passive Acoustic Data*, NOAA National Centers for Environmental Information,
> 2020. doi:[10.25921/saca-sp25](https://doi.org/10.25921/saca-sp25)

| Recording | SanctSound site |
|---|---|
| Humpback whale | Channel Islands, CI04 (2020-02-25) |
| Humpback whale and dolphin | Hawaiian Islands, HI04 (2020-01-29) |
| Snapping shrimp | Florida Keys, FK02 (2019-11-03) |
| Fish chorus | Gray's Reef, GR01 (2019-09-02) |

The speech recordings in the results gallery (speakers p227 and p228) come from
the VCTK corpus:

> J. Yamagishi, C. Veaux and K. MacDonald, *CSTR VCTK Corpus: English
> Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92),
> University of Edinburgh, Centre for Speech Technology Research, 2019.
> doi:[10.7488/ds/2645](https://doi.org/10.7488/ds/2645)

Cite the SanctSound collection when you reuse the excerpts. NOAA's disclaimer
and use constraints apply to them independently of this repository's license.
The hyperbolic chirp is a synthetic signal generated for this work. See
[`benchmarks/wav/README.md`](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/benchmarks/wav/README.md) for every file's source,
original file name and checksum.

## Citation

If the U-STFT or this code helps your work, please cite the paper:

Paper: [arXiv:2610.37795](https://arxiv.org/abs/2610.37795)

```bibtex
@misc{mittal2026ustft,
  title         = {Universal Short-Time Fourier Transform: Data-Adaptive Window-Length Selection for Time-Frequency Analysis},
  author        = {Mittal, Manan and Sarkar, Kanad and Corey, Ryan M. and Buck, John R. and Singer, Andrew C.},
  year          = {2026},
  eprint        = {2610.37795},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SP},
  url           = {https://arxiv.org/abs/2610.37795}
}
```

GitHub's "Cite this repository" button (from
[`CITATION.cff`](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/CITATION.cff))
gives the same reference in APA and BibTeX. Please cite the paper, also when you
use only the code.

Questions: open an issue or email `manan.mittal@stonybrook.edu`.

This work was supported in part by the Office of Naval Research under Award N00014-23-1-2133.

## License

MIT. See [LICENSE](https://github.com/mananmittal/Universal-Short-Time-Fourier-Transform/blob/main/LICENSE).
