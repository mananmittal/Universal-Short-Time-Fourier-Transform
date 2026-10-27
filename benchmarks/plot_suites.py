"""Shared presentation of existing example and benchmark arrays.

This module does not run, invert or filter any transform. Callers retain their
signal generation, transform parameters, causality masks and metric formulas.
"""

from functools import wraps
from pathlib import Path
import numpy as np

from benchmarks.figure_export import ROOT, figure_edition, save_spec, slug
from benchmarks.plot_style import LABELS


def panel(title, xlabel, ylabel, xlim, ylim):
    return {"title": title, "xlabel": xlabel, "ylabel": ylabel,
            "xlim": list(xlim), "ylim": list(ylim), "xscale": "linear", "yscale": "linear",
            "lines": [], "images": [], "bars": [], "scatter": [], "arrows": [],
            "xticks": [], "yticks": [], "xticklabels": [], "yticklabels": []}


def curve_panel(title, time, curves, ylabel, scale="linear"):
    time = np.asarray(time, dtype=float)
    finite = np.concatenate([np.asarray(y)[np.isfinite(y)] for y in curves.values()])
    bounds = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (0, 1)
    pad = max(0.05 * (bounds[1] - bounds[0]), 1e-8)
    result = panel(title, "Time (s)", ylabel, (time[0], time[-1]), (bounds[0] - pad, bounds[1] + pad))
    result["yscale"] = scale
    for key, values in curves.items():
        result["lines"].append({"x": time, "y": np.asarray(values, dtype=float),
            "method": key, "label": LABELS[key], "reference_axis": None, "linestyle": None})
    return result


def power_panels(spectra, fs, hop_size, floor_db=-70):
    """One pooled power reference for all maps, including method-only twins."""
    powers = {key: np.abs(np.asarray(value)) ** 2 for key, value in spectra.items()}
    finite = np.concatenate([value[np.isfinite(value)].ravel() for value in powers.values()])
    reference = max(float(np.percentile(finite, 99.8)), np.finfo(float).tiny)
    result = []
    for key, power in powers.items():
        duration = power.shape[0] * hop_size / fs
        values = 10 * np.log10(np.maximum(power, reference * 10 ** (floor_db / 10)) / reference)
        item = panel(LABELS[key], "Time (s)", "Frequency (Hz)", (0, duration), (0, fs / 2))
        item["images"] = [{"values": values.T, "extent": [0, duration, 0, fs / 2],
                            "clim": [floor_db, 0], "cmap": "magma", "label": "Power (dB)"}]
        result.append(item)
    return result


def posterior_panel(pmf, lengths, duration):
    item = panel("Posterior over dyadic supports", "Time (s)", "Window (samples)",
                 (0, duration), (-0.5, len(lengths) - 0.5))
    item["images"] = [{"values": np.asarray(pmf).T,
                        "extent": [0, duration, -0.5, len(lengths) - 0.5],
                        "clim": [0, 1], "cmap": "Blues", "label": "Probability"}]
    item["yticks"] = list(range(len(lengths)))
    item["yticklabels"] = [str(int(value)) for value in lengths]
    return item


def _metadata(title, hop_size, backend=None):
    return {"dataset": slug(title), "trials": "single run", "hop": int(hop_size),
            "source": f"Released {backend or 'NumPy'} transform; existing plotted arrays"}


def example_figure_build(backend):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with figure_edition(ROOT / "results" / "examples" / backend,
                                f"examples/example{'_torch' if backend == 'torch' else ''}.py; original signals and parameters") as edition:
                result = function(*args, **kwargs)
            print(f"Wrote paired figures to {edition.root}")
            return result
        return wrapped
    return decorate


def example_figures(title, fs, hop_size, duration, analysis, pmf, lengths,
                    candidate_specs, candidate_entropy, blend_entropy,
                    candidate_time, blend_time, candidate_frequency, blend_frequency,
                    prefix_candidates, prefix_blend, time, backend="numpy"):
    metadata = _metadata(title, hop_size, backend)
    spectra = {f"fixed_{index}": value for index, value in enumerate(candidate_specs)}
    spectra["posterior_blend"] = analysis
    save_spec({"metadata": metadata, "panels": power_panels(spectra, fs, hop_size)},
              ROOT / "results" / "examples" / backend / f"{slug(title)}_spectrogram.png")
    save_spec({"metadata": metadata, "panels": [posterior_panel(pmf, lengths, duration)]},
              ROOT / "results" / "examples" / backend / f"{slug(title)}_posterior.png")
    for name, candidates, blend in (
            ("total", candidate_entropy, blend_entropy),
            ("time", candidate_time, blend_time),
            ("frequency", candidate_frequency, blend_frequency)):
        cumulative = np.cumsum(blend)
        denominator = np.arange(1, len(blend) + 1)
        curves = {f"fixed_{index}": (np.cumsum(np.nan_to_num(candidates[:, index], nan=0.0)) - cumulative) / denominator
                  for index in range(candidates.shape[1])}
        curves["posterior_blend"] = cumulative - cumulative
        item = curve_panel(f"Average {name} entropy difference", time, curves,
                           "Entropy difference", "symlog")
        save_spec({"metadata": metadata, "panels": [item]},
                  ROOT / "results" / "examples" / backend / f"{slug(title)}_{name}_entropy_difference.png")
    curves = {f"fixed_{index}": value for index, value in enumerate(prefix_candidates)}
    curves["posterior_blend"] = prefix_blend
    item = curve_panel("Prefix 2-D Rényi Entropy", time, curves, "Rényi-3 (bits)")
    save_spec({"metadata": metadata, "panels": [item]},
              ROOT / "results" / "examples" / backend / f"{slug(title)}_renyi_prefix.png")


def evaluation_figures(title, fs, hop_size, analysis, pmf, lengths, candidate_specs,
                       prefix_candidates, prefix_blend, candidate_metrics, universal_metrics,
                       entropy_components, n_frames, output):
    """All candidates in fixed support order, independent of their performance."""
    metadata = _metadata(title, hop_size)
    spectra = {f"fixed_{index}": value[:n_frames] for index, value in enumerate(candidate_specs)}
    spectra["posterior_blend"] = analysis[:n_frames]
    output = Path(output)
    save_spec({"metadata": metadata, "panels": power_panels(spectra, fs, hop_size, -60)},
              output / "spectrogram_comparison.png")
    duration = n_frames * hop_size / fs
    save_spec({"metadata": metadata, "panels": [posterior_panel(pmf[:n_frames], lengths, duration)]},
              output / "posterior.png")
    time = np.linspace(0, duration, len(prefix_blend))
    curves = {f"fixed_{index}": np.asarray(value[:len(time)]) for index, value in enumerate(prefix_candidates)}
    curves["posterior_blend"] = prefix_blend
    save_spec({"metadata": metadata, "panels": [curve_panel("Prefix 2-D Rényi Entropy", time, curves, "Rényi-3 (bits)")]},
              output / "renyi_prefix.png")
    for candidates, blend, component in entropy_components:
        framewise_floor = np.nanmin(candidates, axis=1)
        cumulative_floor = np.cumsum(framewise_floor)
        denominator = np.arange(1, len(blend) + 1)
        curves = {f"fixed_{index}": (np.cumsum(candidates[:, index]) - cumulative_floor) / denominator
                  for index in range(candidates.shape[1])}
        curves["posterior_blend"] = (np.cumsum(blend) - cumulative_floor) / denominator
        time = np.linspace(0, duration, len(blend))
        item = curve_panel(f"Gap to framewise candidate lower envelope: {component}",
                           time, curves, "Entropy gap", "symlog")
        # A myopic baseline diagnostic; never labeled schedule regret.
        save_spec({"metadata": metadata, "panels": [item]}, output / f"{component.lower()}_framewise_entropy_gap.png")
    for index, (name, ylabel) in enumerate((
            ("reconstruction_mse", "MSE"), ("spectral_convergence", "Spectral convergence"),
            ("envelope_kurtosis", "Envelope kurtosis"), ("phase_jitter", "Phase jitter"))):
        item = panel(name.replace("_", " "), "Method", ylabel, (-0.6, len(lengths) + 0.6), (0, 1))
        keys = ["posterior_blend", *(f"fixed_{i}" for i in range(len(lengths)))]
        values = [universal_metrics[index], *(row[index] for row in candidate_metrics)]
        item["bars"] = [{"x": position - 0.3, "width": 0.6, "value": float(value), "bottom": 0, "method": key}
                         for position, (key, value) in enumerate(zip(keys, values))]
        item["xticks"] = list(range(len(keys)))
        item["xticklabels"] = [LABELS[key] for key in keys]
        save_spec({"metadata": metadata, "panels": [item]}, output / f"{name}.png")
