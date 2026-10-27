"""Build the complete WAV figure collection in paired dissertation/slide sets.

Run the full audit and all local WAV excerpts with one command::

    python -m benchmarks.wav_collection

Or reuse a completed full paper audit::

    python -m benchmarks.wav_collection --from-paper-edition PATH

The released transform and the paper's standardized-emission hindsight path
are imported directly. Entropy differences to that one path are diagnostics;
the path does not minimize joint spectrogram entropy or each component alone.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import wave

import numpy as np

from benchmarks import run_paper as paper
from benchmarks.figure_export import ROOT, figure_edition, load_evidence, save_spec, slug
from benchmarks.plot_style import LABELS, method_from_label
from benchmarks.plot_suites import curve_panel, power_panels, posterior_panel


N_FFT = 4096
HOP = 128
BETA = 1.5
CACHED_PAPER_WAVS = {
    "hyperbolic_chirp.wav": "hyperbolic-chirp",
    "humpback_whale_song.wav": "humpback-whale",
    "snapping_shrimp.wav": "snapping-shrimp",
}
FULL_TEST_WAVS = {"hyperbolic_chirp.wav", "mixed_test_signal.wav"}
# Display ranges reviewed against all ten recorded spectrograms. Full-band
# companions retain higher-frequency power whenever the main view is cropped.
FOCUS_HZ = {
    "humpback_whale_song.wav": 4000,
    "fish_chorus.wav": 4000,
    "humpback_whale_and_dolphins.wav": 4000,
    "humpback8kmono.wav": 4000,
    "english_speech_1.wav": 8000,
    "english_speech_2.wav": 8000,
    "cool_soliloquy_A.wav": 16000,
}
OBJECTIVE = (
    "Minimum full-horizon sum of beta * (ell - framewise minimum ell) / "
    "(framewise std(ell) + 1e-6), ell = normalized time H3 + normalized "
    "one-sided rFFT H3, on causally available stay/grow/reset paths starting at H. "
    "Transition magnitudes enter the path code, not this objective."
)
COMPACT_FIGURES = (
    "01_spectrograms", "02_regret_time_frequency_total",
    "03_regret_2d_renyi3", "04_blend_weights_pmf",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_excerpt(path):
    """Read only the authorized PCM excerpt, then use paper normalization."""
    path = Path(path)
    with wave.open(str(path), "rb") as stream:
        fs, channels, width, frames = (
            stream.getframerate(), stream.getnchannels(),
            stream.getsampwidth(), stream.getnframes(),
        )
        count = frames if path.name in FULL_TEST_WAVS else min(frames, 5 * fs)
        raw = stream.readframes(count)
    if width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        samples = octets[:, 0] | octets[:, 1] << 8 | octets[:, 2] << 16
        samples = ((samples ^ 0x800000) - 0x800000).astype(float)
    elif width in (1, 2, 4):
        samples = np.frombuffer(raw, dtype={1: "u1", 2: "<i2", 4: "<i4"}[width]).astype(float)
    else:
        raise ValueError(f"Unsupported PCM width {width}: {path}")
    samples = samples.reshape(-1, channels).mean(axis=1)
    if not len(samples):
        raise ValueError(f"Empty WAV excerpt: {path}")
    samples -= np.mean(samples)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    provenance = {
        "filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
        "sample_rate_hz": fs, "channels": channels, "pcm_bits": width * 8,
        "original_duration_s": frames / fs, "excerpt_start_s": 0.0,
        "excerpt_end_s": len(samples) / fs, "excerpt_samples": len(samples),
        "normalization": "channel mean, excerpt demean, excerpt peak normalization",
        "n_fft": N_FFT, "hop_samples": HOP, "beta": BETA,
    }
    return samples, provenance


def joint_prefix_bits(spectrogram, n_fft):
    """Reuse the paper's full-DFT H3; zero-energy prefixes are undefined."""
    spectrogram = np.asarray(spectrogram)
    if spectrogram.ndim != 2 or spectrogram.shape[1] != n_fft // 2 + 1:
        raise ValueError("Wrong spectrogram grid")
    if not np.all(np.isfinite(spectrogram)):
        raise ValueError("Spectrogram must be finite")
    positive = np.flatnonzero(np.any(np.abs(spectrogram) > 0, axis=1))
    result = np.full(spectrogram.shape[0], np.nan)
    if len(positive):
        first = int(positive[0])
        result[first:] = paper._joint_spectrogram_renyi3_prefix(
            spectrogram[first:], n_fft
        )["entropy_bits"]
    return result


def extract_case(case):
    """Retain existing diagnostics, without copying an STFT implementation."""
    return {
        "analysis": case["trellis"][0],
        "comparator_path": case["constrained_indices"],
        **{key: case[key] for key in (
            "candidate_specs", "lengths", "pmf", "candidate_costs",
            "candidate_h_time", "candidate_h_spec", "blended_costs",
            "blended_h_time", "blended_h_spec", "transition_matrix",
        )},
    }


def diagnostics(arrays, warmup, n_fft, hop, fs):
    """All entropy measures share one feasible full-horizon hindsight path."""
    candidates = np.asarray(arrays["candidate_specs"])
    path = np.asarray(arrays["comparator_path"], dtype=int)
    lengths = np.asarray(arrays["lengths"], dtype=int)
    pmf = np.asarray(arrays["pmf"])
    frames = len(path)
    if candidates.shape[:2] != (len(lengths), frames):
        raise ValueError("Candidate spectra and schedule disagree")
    if not np.array_equal(lengths, hop * 2 ** np.arange(len(lengths))):
        raise ValueError("Supports must be strictly dyadic")
    if path[0] != 0 or np.any(path < 0) or np.any(path >= len(lengths)):
        raise ValueError("Invalid hindsight start/state")
    if np.any(arrays["transition_matrix"][path[:-1], path[1:]] <= 0):
        raise ValueError("Hindsight path uses an impossible transition")
    rows = np.arange(frames)
    if np.any(~np.isfinite(arrays["candidate_costs"][rows, path])):
        raise ValueError("Hindsight path uses an unavailable candidate")
    if np.any(pmf < 0) or not np.allclose(pmf.sum(axis=1), 1, atol=1e-12):
        raise ValueError("Blend weights must be a probability distribution")
    unavailable = lengths[None, :] > (rows[:, None] + 1) * hop
    if np.any(pmf[unavailable] != 0):
        raise ValueError("Blend gives weight to an unavailable support")

    region = slice(warmup, None)
    comparator_spectrum = candidates[path[region], rows[region]]
    joint_u = joint_prefix_bits(arrays["analysis"][region], n_fft)
    joint_hindsight = joint_prefix_bits(comparator_spectrum, n_fft)
    fixed_joint = np.array([
        joint_prefix_bits(spectrum[region], n_fft) for spectrum in candidates
    ])
    fixed_entropy = fixed_joint[:, -1]
    if not np.all(np.isfinite(fixed_entropy)):
        raise ValueError("Fixed-window ranking requires a nonzero excerpt")
    ranking = np.argsort(fixed_entropy, kind="stable")
    selected = {"best": int(ranking[0]), "middle": int(ranking[len(ranking) // 2]),
                "worst": int(ranking[-1])}

    count = np.arange(1, frames - warmup + 1)
    joint_regrets = {"posterior_blend": joint_u - joint_hindsight,
        **{f"fixed_{index}": prefix - joint_hindsight
           for index, prefix in enumerate(fixed_joint)}}
    components = {}
    for name, blend_key, candidate_key in (
        ("time", "blended_h_time", "candidate_h_time"),
        ("frequency", "blended_h_spec", "candidate_h_spec"),
        ("total", "blended_costs", "candidate_costs"),
    ):
        universal = np.asarray(arrays[blend_key])[region]
        hindsight = np.asarray(arrays[candidate_key])[rows[region], path[region]]
        curves = {"posterior_blend": universal,
            **{f"fixed_{index}": np.asarray(arrays[candidate_key])[region, index]
               for index in range(len(lengths))}}
        cumulative = {key: np.cumsum(values - hindsight)
                      for key, values in curves.items()}
        components[name] = {"universal": universal, "hindsight": hindsight,
            "regret": cumulative["posterior_blend"], "curves": curves,
            "cumulative_regrets": cumulative,
            "average_regrets": {key: values / count for key, values in cumulative.items()}}
    for key in joint_regrets:
        if not np.allclose(components["total"]["cumulative_regrets"][key],
                           components["time"]["cumulative_regrets"][key]
                           + components["frequency"]["cumulative_regrets"][key],
                           rtol=1e-10, atol=1e-9):
            raise AssertionError("Total entropy regret differs from time + frequency")
    return {
        "time_s": (rows[region] + 1) * hop / fs,
        "joint_u": joint_u, "joint_hindsight": joint_hindsight,
        "joint_regret": joint_u - joint_hindsight, "components": components,
        "fixed_joint": fixed_joint, "joint_regrets": joint_regrets,
        "average_count": count, "fixed_h3_bits": fixed_entropy, "selected": selected,
    }


def regret_panel(title, time, values, ylabel):
    curves = values if isinstance(values, dict) else {"posterior_blend": values}
    item = curve_panel(title, time, curves, ylabel)
    # This is a diagnostic zero, not another algorithm or a theorem bound.
    item["lines"].append({"x": np.array([time[0], time[-1]]), "y": np.zeros(2),
        "method": None, "label": "Zero regret", "reference_axis": "horizontal", "linestyle": ":"})
    item["xlabel"] = "Time (s); hindsight: min scaled time + frequency H3"
    return item


def selected_windows(record):
    """Keep one common best/middle/worst selection across a WAV's figures."""
    lengths = list(record["candidate_lengths_samples"])
    chosen = record["selected_lengths_samples"]
    if len(set(chosen.values())) != 3:
        raise ValueError("The compact comparison needs three distinct fixed supports")
    return {f"fixed_{lengths.index(length)}": (rank, length)
            for rank, length in chosen.items()}


def compact_specs(specs, record):
    """Consolidate checked plotted arrays into exactly four figure specs."""
    windows = selected_windows(record)
    selected = {"posterior_blend", *windows}

    def find(*names):
        for name in names:
            if name in specs:
                return specs[name]
        raise ValueError(f"Missing cached figure: {names[0]}")

    maps = find("01_spectrograms", "spectrograms_u_all_fixed", "spectrograms_u_best_middle_worst")
    joint = find("03_regret_2d_renyi3", "regret_2d_renyi3")
    weights = find("04_blend_weights_pmf", "blend_weights_map")
    if "02_regret_time_frequency_total" in specs:
        components = specs["02_regret_time_frequency_total"]["panels"]
        component_template = specs["02_regret_time_frequency_total"]
    else:
        components = [find(f"regret_1d_renyi3_{name}")["panels"][0]
                      for name in ("time", "frequency", "total")]
        component_template = find("regret_1d_renyi3_time")
    if len(components) != 3:
        raise ValueError("The 1-D comparison needs time, frequency, and total panels")

    def curves(panel):
        lines = []
        for line in panel["lines"]:
            key = line["method"]
            if key and key not in selected:
                continue
            line = dict(line)
            if key in windows:
                rank, length = windows[key]
                line["label"] = f"{LABELS[key]}\n{rank}, L={length}"
            elif not key and "zero" in line["label"].lower():
                line["label"] = "Zero difference"
            lines.append(line)
        if {line["method"] for line in lines if line["method"]} != selected:
            raise ValueError("A selected comparison window is missing from the cache")
        return {**panel, "lines": lines}

    map_panels = []
    for panel in maps["panels"]:
        key = method_from_label(panel["title"])
        if key not in selected:
            continue
        panel = dict(panel)
        if key in windows:
            rank, length = windows[key]
            panel["display_title"] = f"{LABELS[key]} ({rank}; L = {length})"
        map_panels.append(panel)
    if len(map_panels) != 4:
        raise ValueError("The spectrogram comparison needs U-STFT and three fixed windows")
    panels_1d = []
    for name, panel in zip(("time", "frequency", "total"), components):
        panel = curves(panel)
        panel.update(title=name.capitalize(), ylabel=f"Avg. {name} H3 gap (norm.)", convergence=True)
        panels_1d.append(panel)
    panel_2d = curves(joint["panels"][0])
    panel_2d.update(title="Prefix joint 2-D Rényi-3 entropy gap", ylabel="2-D H3 gap (bits)", convergence=True)
    joint_oracle = record.get("joint_entropy_oracle")
    if joint_oracle:
        panel_2d.update(title="Joint 2-D Rényi-3 regret", ylabel="2-D H3 regret (bits)",
                        xlabel="Time (s); hindsight: minimum joint H3 at each prefix")
        for line in panel_2d["lines"]:
            if not line["method"] and "zero" in line["label"].lower():
                line["label"] = "Zero regret"

    def spec(original, panels, **extra):
        metadata = {key: value for key, value in original["metadata"].items()
                    if key not in {"layout", "summary_panel", "message_prefix"}}
        metadata.update(
            keep_panels_together=True, export_individual_methods=False,
            export_difference=False, figures_per_wav=4,
            fixed_windows_included="best, middle, worst by final full-region joint 2-D H3",
            selected_lengths_samples=record["selected_lengths_samples"],
            entropy_comparison="signed entropy difference to the paper tracking-loss-optimal hindsight schedule",
        )
        metadata.update(extra)
        return {"metadata": metadata, "panels": panels}

    return {
        COMPACT_FIGURES[0]: spec(maps, map_panels),
        COMPACT_FIGURES[1]: spec(component_template, panels_1d, layout="entropy_triptych",
                               summary_panel=2, message_prefix="For the total entropy gap, "),
        COMPACT_FIGURES[2]: spec(joint, [panel_2d], **({
            "entropy_comparison": "joint H3 minus a separately optimized feasible schedule at each plotted prefix",
            "hindsight_objective": joint_oracle["objective"],
            "joint_entropy_oracle": joint_oracle,
        } if joint_oracle else {})),
        COMPACT_FIGURES[3]: spec(weights, weights["panels"]),
    }


def spectrogram_views(panels, provenance):
    """Focus the display only; retain full spectra, metrics and shared scales."""
    nyquist = provenance["sample_rate_hz"] / 2
    limit = min(FOCUS_HZ.get(provenance["filename"], nyquist), nyquist)
    visible_power = []
    fractions = {}
    for item in panels:
        values = item["images"][0]["values"]
        frequencies = np.linspace(0, nyquist, values.shape[0])
        power = 10 ** (values / 10)
        mask = frequencies <= limit
        visible_power.append(power[mask].ravel())
        fractions[item["title"]] = float(np.sum(power[mask]) / np.sum(power))
    # Cached images retain power down to their display floor. Raising the
    # reference preserves that floor and needs no transform recomputation.
    reference = max(1.0, float(np.percentile(np.concatenate(visible_power), 99.8)))
    offset = 10 * np.log10(reference)
    focused = []
    complete = []
    for original in panels:
        item = {**original, "images": [dict(original["images"][0])]}
        item["images"][0]["values"] = np.maximum(original["images"][0]["values"] - offset, -70)
        item["images"][0]["clim"] = [-70, 0]
        item["ylim"] = [0, limit]
        item["xlabel"] = "Time (s); full-band H3 rank"
        focused.append(item)
        full = {**item, "ylim": [0, nyquist]}
        full["xlabel"] = "Time (s); full frequency band"
        complete.append(full)
    record = {"filename": provenance["filename"], "focus_hz": [0, limit],
        "full_band_hz": [0, nyquist], "full_band_companion": limit < nyquist,
        "display_power_fraction_in_focus": fractions,
        "reference_shift_db": float(offset),
        "scale": "one pooled 99.8th-percentile focus reference for every method and both views",
        "entropy_metrics": "unchanged; computed on the complete frequency grid"}
    return focused, complete if limit < nyquist else None, record


def save_wav_figures(arrays, provenance, warmup, edition):
    fs = provenance["sample_rate_hz"]
    dataset = slug(Path(provenance["filename"]).stem)
    metrics = diagnostics(arrays, warmup, N_FFT, HOP, fs)
    metadata = {"dataset": dataset, "trials": "single run", "hop": HOP,
        "source": provenance["filename"], "input": provenance,
        "hindsight_objective": OBJECTIVE,
        "entropy_region": "same post-warm-up suffix for every method",
        "fixed_window_rank": "final full-region joint 2-D Rényi-3; stable support order for ties",
        "fixed_windows_included": "all dyadic supports",
        "one_dimensional_summary": "cumulative average regret since the common post-warm-up start",
        "two_dimensional_summary": "joint entropy of each complete time-frequency prefix; no averaging of entropy values"}
    specs = {}
    def save(name, panels):
        specs[name] = {"metadata": metadata, "panels": panels}

    save("regret_2d_renyi3", [regret_panel("Prefix joint 2-D Rényi-3 regret",
        metrics["time_s"], metrics["joint_regrets"], "2-D H3 regret (bits)")])
    spectra = {"posterior_blend": arrays["analysis"]}
    ranks = {index: rank for rank, index in metrics["selected"].items()}
    for index in range(len(arrays["lengths"])):
        spectra[f"fixed_{index}"] = arrays["candidate_specs"][index]
    maps = power_panels(spectra, fs, HOP)
    for item in maps:
        key = next((key for key in spectra if LABELS[key] == item["title"]), None)
        if key and key.startswith("fixed_"):
            index = int(key.split("_")[1])
            qualifier = ranks[index] + "; " if index in ranks else ""
            item["display_title"] = f"{LABELS[key]} ({qualifier}L = {arrays['lengths'][index]})"
        item["xlabel"] = "Time (s); ranked by joint H3"
    focused, complete, view = spectrogram_views(maps, provenance)
    view["full_band_companion"] = False
    view["exported_view"] = "one reviewed frequency view in the four-figure collection"
    metadata["frequency_view"] = view
    save("spectrograms_u_all_fixed", focused)
    for name, values in metrics["components"].items():
        save(f"regret_1d_renyi3_{name}", [regret_panel(f"Cumulative average {name} Rényi-3 regret",
            metrics["time_s"], values["average_regrets"], f"Average {name} H3 regret (norm.)")])
    frames = len(arrays["pmf"])
    save("blend_weights_map", [posterior_panel(arrays["pmf"], arrays["lengths"], frames * HOP / fs)])
    selection = {"candidate_lengths_samples": arrays["lengths"].tolist(),
                 "selected_lengths_samples": {rank: int(arrays["lengths"][index])
                                              for rank, index in metrics["selected"].items()}}
    for name, spec in compact_specs(specs, selection).items():
        save_spec(spec, name + ".png")
    time = (np.arange(frames) + 1) * HOP / fs

    data = edition.root / "data" / dataset
    rows = []
    for position, t in enumerate(metrics["time_s"]):
        row = {"time_s": float(t), "hindsight_length_samples": int(arrays["lengths"][arrays["comparator_path"][warmup + position]]),
            "joint_u_h3_bits": float(metrics["joint_u"][position]),
            "joint_hindsight_h3_bits": float(metrics["joint_hindsight"][position]),
            "joint_regret_bits": float(metrics["joint_regret"][position])}
        for name, values in metrics["components"].items():
            row[f"{name}_u_normalized_h3"] = float(values["universal"][position])
            row[f"{name}_hindsight_normalized_h3"] = float(values["hindsight"][position])
            row[f"{name}_cumulative_regret"] = float(values["regret"][position])
            row[f"{name}_average_regret"] = float(values["average_regrets"]["posterior_blend"][position])
        row["post_warmup_frame_count"] = int(metrics["average_count"][position])
        rows.append(row)
    write_csv(data / "entropy_regret.csv", rows)
    comparison = []
    for position, t in enumerate(metrics["time_s"]):
        for key, joint in metrics["joint_regrets"].items():
            row = {"time_s": float(t), "post_warmup_frame_count": int(metrics["average_count"][position]),
                "method": key, "fixed_length_samples": "" if key == "posterior_blend" else int(arrays["lengths"][int(key.split("_")[1])]),
                "joint_h3_bits": float(metrics["joint_u"][position] if key == "posterior_blend" else metrics["fixed_joint"][int(key.split("_")[1]), position]),
                "joint_hindsight_h3_bits": float(metrics["joint_hindsight"][position]),
                "joint_regret_bits": float(joint[position])}
            for name, values in metrics["components"].items():
                row[f"{name}_normalized_h3"] = float(values["curves"][key][position])
                row[f"{name}_hindsight_normalized_h3"] = float(values["hindsight"][position])
                row[f"{name}_cumulative_regret"] = float(values["cumulative_regrets"][key][position])
                row[f"{name}_average_regret"] = float(values["average_regrets"][key][position])
            comparison.append(row)
    write_csv(data / "all_method_entropy_regret.csv", comparison)
    write_csv(data / "blend_weights.csv", [
        {"time_s": float(t), **{f"L_{int(length)}": float(arrays["pmf"][position, index])
            for index, length in enumerate(arrays["lengths"])}}
        for position, t in enumerate(time)
    ])
    record = {**provenance, "dataset": dataset, "warmup_frames": warmup,
        "candidate_lengths_samples": arrays["lengths"].tolist(),
        "fixed_window_h3_bits": metrics["fixed_h3_bits"].tolist(),
        "selected_lengths_samples": {rank: int(arrays["lengths"][index])
                                     for rank, index in metrics["selected"].items()},
        "final_joint_regret_bits": float(metrics["joint_regret"][-1]),
        "final_1d_cumulative_regret": {name: float(values["regret"][-1])
                                     for name, values in metrics["components"].items()},
        "final_1d_average_regret": {name: {key: float(value[-1]) for key, value in values["average_regrets"].items()}
                                   for name, values in metrics["components"].items()},
        "fixed_windows_included": "best, middle, worst by final full-region joint 2-D H3",
        "figures_per_wav": 4,
        "one_dimensional_summary": metadata["one_dimensional_summary"],
        "two_dimensional_summary": metadata["two_dimensional_summary"],
        "hindsight_objective": OBJECTIVE, "frequency_view": view}
    (data / "input_and_ranking.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_paper_edition(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    # Check all audit evidence and the optional numerical WAV checkpoints used
    # here, rather than loading unrelated large plotting arrays into memory.
    for relative, digest in manifest["sha256"].items():
        if relative.startswith(("data/audit/", "data/paper_cases/")):
            path = source / relative
            if source not in path.resolve().parents or sha256(path) != digest:
                raise ValueError(f"Changed paper evidence: {relative}")
    summary = json.loads((source / "data/audit/summary.json").read_text())
    if summary["configuration"]["seeds"] != list(range(6)):
        raise ValueError("A full six-seed paper sweep is required")
    if not (source / "data/audit/EVIDENCE_AUDIT.md").is_file():
        raise ValueError("The full paper audit report is missing")
    return manifest


def build_collection(source=None, wav_dir=None, output_dir=None):
    wav_dir = Path(wav_dir or ROOT / "benchmarks/wav").resolve()
    output_dir = Path(output_dir or ROOT / "results/paper_wavs").resolve()
    if source is None:
        with figure_edition(output_dir, "Complete paper sweep: six seeds and four finite SNR levels") as audit:
            paper.run_paper_experiments(audit.root / "data/audit", quick=False,
                wav_dir=wav_dir, figure_dir=audit.root / "dissertation")
        source = audit.root
    source = Path(source).resolve()
    manifest = verify_paper_edition(source)
    paths = sorted(wav_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No WAV files in {wav_dir}")
    records = []
    with figure_edition(output_dir,
            f"All {len(paths)} local WAV files; released NumPy transform; full paper evidence from {source}") as edition:
        shutil.copytree(source / "data/audit", edition.root / "data/audit")
        if (source / "run.log").is_file():
            shutil.copy2(source / "run.log", edition.root / "data/audit/full_sweep_run.log")
        for position, path in enumerate(paths, 1):
            samples, provenance = read_excerpt(path)
            print(f"WAV {position}/{len(paths)}: {path.name}; {provenance['excerpt_end_s']:.3f} s", flush=True)
            paper_slug = CACHED_PAPER_WAVS.get(path.name)
            cached = source / "data/paper_cases" / f"{paper_slug}.npz"
            if paper_slug and cached.is_file():
                if provenance["sha256"] != paper._PAPER_AUDIO_SHA256[path.name]:
                    raise ValueError(f"Paper WAV checksum mismatch: {path.name}")
                relative = str(cached.relative_to(source))
                if relative not in manifest["sha256"]:
                    raise ValueError("Unverified paper case cache")
                meta = json.loads(cached.with_suffix(".json").read_text())
                if (meta["fs"], meta["n_fft"], meta["hop_size"], meta["beta"]) != (
                        provenance["sample_rate_hz"], N_FFT, HOP, BETA):
                    raise ValueError("Paper case parameters differ")
                with np.load(cached, allow_pickle=False) as stored:
                    arrays = {key: stored[key] for key in stored.files}
                warmup = meta["warmup"]
                provenance["transform_source"] = f"verified full-paper cache: {relative}"
            else:
                case = paper._case_outputs(samples, N_FFT, HOP, BETA, "geometric")
                arrays = extract_case(case)
                warmup = case["warmup"]
                del case
                provenance["transform_source"] = "released universal_stft/numpy/stft.py via paper._case_outputs"
            records.append(save_wav_figures(arrays, provenance, warmup, edition))
            del arrays, samples
            gc.collect()
        (edition.root / "data/wav_inventory.json").write_text(json.dumps(records, indent=2) + "\n")
        (edition.root / "data/frequency_views.json").write_text(
            json.dumps([record["frequency_view"] for record in records], indent=2) + "\n")
        write_csv(edition.root / "data/wav_inventory.csv", [{key: row[key] for key in (
            "filename", "dataset", "sha256", "bytes", "sample_rate_hz", "original_duration_s",
            "excerpt_start_s", "excerpt_end_s", "n_fft", "hop_samples", "beta",
        )} for row in records])
    readme = edition.root / "README.md"
    with readme.open("a") as stream:
        stream.write("\n## WAV coverage and metric definitions\n\n")
        stream.write(f"{len(records)}/{len(paths)} WAV files included. Each dataset has exactly four figures in each set: "
                     "one four-panel spectrogram, one combined time/frequency/total comparison, "
                     "one joint 2-D entropy comparison, and one blend-weight heatmap. "
                     "Comparisons use U-STFT and the best, middle, and worst fixed windows.\n\n")
        stream.write("All WAVs use their first five seconds (or their full length when shorter); "
                     "hyperbolic_chirp.wav and mixed_test_signal.wav use all 15 seconds. "
                     "No resampling. NFFT = 4096, H = 128, beta = 1.5, dyadic L = 128 through 2048. "
                     "Exact original durations, excerpts, hashes and rankings are in data/wav_inventory.json.\n\n")
        stream.write("**One comparator for every entropy measure:** " + OBJECTIVE + "\n\n")
        stream.write("The 2-D entropy gap is H3 of each method's spectrogram prefix minus H3 of the "
                     "same prefix under this full-horizon hindsight path, in bits. The common region "
                     "starts after the 16-frame warm-up. Power is normalized over the complete "
                     "time-frequency grid; interior rFFT bins account for both full-DFT cells. "
                     "An all-zero prefix is undefined and is recorded as NaN.\n\n")
        stream.write("The combined 1-D time/frequency/total figure shows cumulative average entropy gaps: "
                     "sum of each method's frame entropy minus the corresponding candidate entropy "
                     "along that same path, divided by the number of frames since the common post-warm-up start. "
                     "The denominator starts at one and is not reset when the first 5% is hidden for display. "
                     "They retain the released time/log2(L) and unweighted one-sided rFFT/log2(B) "
                     "normalization. Total equals time plus frequency for U-STFT and every fixed window. "
                     "Raw cumulative sums and the averages for every method are saved in all_method_entropy_regret.csv. "
                     "Joint 2-D entropy retains its complete-prefix PMF definition; it is not averaged a second time. "
                     "These are entropy diagnostics, "
                     "not the HMM log-loss regret certificate; they can be negative because the "
                     "comparator optimizes standardized tracking loss rather than each entropy separately.\n\n")
        stream.write("The same best/middle/worst fixed windows are shown in every comparison for a WAV, ranked "
                     "by final joint 2-D H3 on the common region, with stable support order for ties. Spectrograms share the "
                     "pooled 99.8th-percentile power reference and a −70 to 0 dB scale. "
                     "The reviewed main frequency view is used. Blend weights are shown "
                     "only as a 2-D heatmap retaining the complete five-window PMF.\n\n")
        append_frequency_table(stream, [record["frequency_view"] for record in records])
        stream.write("## Full paper sweep evidence\n\nThe complete six-seed, four-SNR, dyadic/linear "
                     "audit (48 noisy cases plus two clean cases), all runtime repetitions, and "
                     "the three checksum-locked paper WAV checks completed. CSV/JSON results "
                     "and EVIDENCE_AUDIT.md are copied into data/audit. The raw sweep log retains "
                     "one SciPy notice about a skipped non-audio WAV metadata chunk; audio decoding "
                     "and all checks completed. The figure renderer emitted no errors or warnings.\n\n")
        stream.write(f"Source paper edition: `{source}`. Its original plots and evidence are retained.\n\n")
        stream.write(f"Build this complete paired WAV set in one command: "
                     f"`python -m benchmarks.wav_collection --from-paper-edition '{source}'`. "
                     "Omit --from-paper-edition to rerun the full paper sweep first.\n")
    with (source / "README.md").open("a") as stream:
        stream.write(f"\nConsolidated WAV deliverable: `{edition.root}`; original sweep figures and evidence retained.\n")
    print(f"COMPLETE WAV COLLECTION: {edition.root}", flush=True)
    return edition.root


def append_frequency_table(stream, records):
    stream.write("\n## Reviewed frequency views\n\n")
    stream.write("The main humpback views use 0–4 kHz. Fish chorus uses 0–4 kHz, "
                 "speech uses 0–8 kHz, and the curved tracks in cool_soliloquy_A use "
                 "0–16 kHz. The compact collection keeps one reviewed frequency view. "
                 "Shrimp and the 8 kHz test "
                 "recordings retain their complete bands. This changes display limits "
                 "only; all entropy values, rankings and hindsight paths are unchanged.\n\n")
    stream.write("| WAV | Main frequency range (Hz) | Full-band companion |\n|---|---:|---|\n")
    for record in records:
        stream.write(f"| {record['filename']} | 0–{record['focus_hz'][1]:g} | "
                     f"{'yes' if record['full_band_companion'] else 'no'} |\n")


def refine_cached_collection(source, output_dir=None):
    """Consolidate every WAV into four figures using checked plot arrays."""
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    for relative, digest in manifest["sha256"].items():
        path = source / relative
        if source not in path.resolve().parents or sha256(path) != digest:
            raise ValueError(f"Changed WAV collection evidence: {relative}")
    inventory = json.loads((source / "data/wav_inventory.json").read_text())
    plot_files = {relative for name in manifest["evidence"]
                  for relative in (name, str(Path(name).with_suffix(".npz")))}
    records = []
    with figure_edition(output_dir or source.parent,
                       f"Four figures for each of {len(inventory)} WAV files; verified plotted arrays from {source}") as edition:
        for relative in manifest["sha256"]:
            if relative not in plot_files:
                target = edition.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
        for position, original in enumerate(inventory, 1):
            dataset = original["dataset"]
            specs = {Path(relative).stem: load_evidence(source / relative)
                     for relative in manifest["evidence"]
                     if Path(relative).parts[1] == dataset and not Path(relative).stem.endswith("_full_band")
                     and Path(relative).stem != "blend_weights_curves"}
            view = {**original["frequency_view"], "full_band_companion": False,
                    "exported_view": "one reviewed frequency view in the four-figure collection"}
            record = {**original, "frequency_view": view, "figures_per_wav": 4,
                      "fixed_windows_included": "best, middle, worst by final full-region joint 2-D H3"}
            for name, spec in compact_specs(specs, record).items():
                spec["metadata"]["frequency_view"] = view
                edition.save(spec, name + ".png")
            (edition.root / "data" / dataset / "input_and_ranking.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(f"[{position}/{len(inventory)}] Four combined figures: {original['filename']}", flush=True)
        (edition.root / "data/wav_inventory.json").write_text(json.dumps(records, indent=2) + "\n")
        (edition.root / "data/frequency_views.json").write_text(
            json.dumps([record["frequency_view"] for record in records], indent=2) + "\n")
        if len(edition.entries) != 4 * len(records):
            raise AssertionError("The compact collection must have exactly four figures per WAV")
    with (edition.root / "README.md").open("a") as stream:
        stream.write("\n## WAV coverage and metric definitions\n\n")
        stream.write(f"{len(records)} WAVs, exactly four figures per WAV in each export set:\n\n"
                     "1. Spectrograms together: U-STFT and the best, middle, and worst fixed windows.\n"
                     "2. One combined figure: time, frequency, and total cumulative average entropy gaps.\n"
                     "3. Joint 2-D Rényi-3 prefix entropy gap.\n"
                     "4. The complete five-window blend PMF over time as one heatmap.\n\n")
        stream.write("The same three fixed windows are used throughout each WAV, ranked by final "
                     "full-frequency joint 2-D Rényi-3 entropy, with stable support order for ties. "
                     "The full bank remains in the numerical CSV evidence. Spectrograms retain "
                     "their shared −70 to 0 dB reference and reviewed frequency limits.\n\n")
        if any(record.get("joint_entropy_oracle") for record in records):
            stream.write("**Separate 2-D comparator:** Each displayed point subtracts the "
                         "minimum joint H3 of its complete spectrogram prefix, using the "
                         "separate stay/grow/reset schedule recorded in joint_entropy_oracle. "
                         "The exact plotted values, paths, sampling grids, and optimality "
                         "certificates are retained. See joint_entropy_regret.csv for this "
                         "comparison; tracking_entropy_diagnostics.csv retains the old "
                         "tracking-objective diagnostics.\n\n")
        stream.write("**1-D entropy comparisons:** " + OBJECTIVE + " "
                     "These are signed entropy differences to that tracking-loss-optimal schedule, "
                     "not minima of the individual entropy objectives; they may be negative. "
                     "The plotted values and schedule are unchanged. The 1-D curves retain their "
                     "cumulative-average denominator starting at one after the common warm-up. "
                     "Joint 2-D entropy retains its complete-prefix PMF definition without another averaging. "
                     "Tracking-diagnostic CSV columns named regret store these signed entropy gaps.\n\n")
        stream.write("The excerpt convention remains the first five seconds (full duration when shorter), "
                     "and the complete 15-second test/chirp files. Input hashes, sample rates, windows, "
                     "and preprocessing are recorded in data/wav_inventory.json. The complete original "
                     "paper sweep evidence remains in data/audit.\n\n")
        append_frequency_table(stream, [record["frequency_view"] for record in records])
        stream.write(f"\nRebuild this complete collection without recomputing transforms: "
                     f"`python -m benchmarks.wav_collection --from-edition '{edition.root}'`.\n")
    with (source / "README.md").open("a") as stream:
        stream.write(f"\nSuperseded by `{edition.root}`; figures and evidence retained.\n")
    print(f"COMPLETE FOUR-FIGURE WAV COLLECTION: {edition.root}", flush=True)
    return edition.root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-paper-edition", type=Path,
                        help="reuse a completed full six-seed paper audit")
    parser.add_argument("--from-edition", type=Path,
                        help="reuse all WAV plot arrays; no transform recomputation")
    parser.add_argument("--wav-dir", type=Path, help="all local WAV inputs")
    parser.add_argument("--output-dir", type=Path, help="parent for a fresh dated collection")
    arguments = parser.parse_args()
    if arguments.from_edition and arguments.from_paper_edition:
        parser.error("Choose either --from-edition or --from-paper-edition")
    if arguments.from_edition:
        refine_cached_collection(arguments.from_edition, arguments.output_dir)
    else:
        build_collection(arguments.from_paper_edition, arguments.wav_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
