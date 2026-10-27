"""Replace only the 2-D comparator in an existing four-figure WAV collection.

python -m benchmarks.joint_entropy_collection --from-edition PATH

The paper's tracker and its 1-D comparisons retain their original objective.
Every displayed 2-D point receives a separate certified joint-entropy oracle.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from benchmarks.figure_export import ROOT, figure_edition, load_evidence
from benchmarks.joint_entropy import entropy, optimize_prefixes, spectral_moments
from benchmarks import wav_collection as wav


JOINT_OBJECTIVE = (
    "Minimum joint full-frequency Renyi-3 entropy of the entire common-region "
    "spectrogram prefix, optimized separately at every plotted prefix over "
    "dyadic stay/grow/reset schedules. All states are available at the common "
    "post-warm-up start. No transition penalty; no 1-D surrogate."
)


def prefix_stops(frames, points=65):
    """Use a disclosed uniform time grid, including first and final prefixes."""
    if frames < 1 or points < 2:
        raise ValueError("Need positive frames and at least two plotting points")
    return np.unique(np.rint(np.linspace(1, frames, min(frames, points))).astype(int))


def _raw_curves(source, record):
    name = source / "data" / record["dataset"] / "all_method_entropy_regret.csv"
    if not name.is_file():
        name = name.with_name("tracking_entropy_diagnostics.csv")
    with name.open() as stream:
        rows = list(csv.DictReader(stream))
    methods = sorted({row["method"] for row in rows})
    curves = {method: np.array([float(row["joint_h3_bits"]) for row in rows
                               if row["method"] == method]) for method in methods}
    times = np.array([float(row["time_s"]) for row in rows
                      if row["method"] == "posterior_blend"])
    return curves, times


def prepare_moments(source, record, cache, wav_dir):
    """Replay the released implementation and verify every old fixed H3 value."""
    target = cache / record["dataset"]
    target.mkdir(parents=True, exist_ok=True)
    moment_file = target / "moments.npz"
    curves, times = _raw_curves(source, record)
    if moment_file.is_file():
        with np.load(moment_file, allow_pickle=False) as data:
            a, b, transition = data["a"], data["b"], data["transition"]
    else:
        samples, provenance = wav.read_excerpt(wav_dir / record["filename"])
        if (provenance["sha256"] != record["sha256"] or
                provenance["excerpt_samples"] != record["excerpt_samples"]):
            raise ValueError("The input excerpt differs from the plotted recording")
        case = wav.paper._case_outputs(samples, record["n_fft"], record["hop_samples"],
                                       record["beta"], "geometric")
        if case["warmup"] != record["warmup_frames"]:
            raise ValueError("Warm-up differs from the existing collection")
        a, b = spectral_moments(case["candidate_specs"][:, case["warmup"]:], record["n_fft"])
        transition = case["transition_matrix"]
        np.savez_compressed(moment_file, a=a, b=b, transition=transition)
    for state in range(a.shape[1]):
        np.testing.assert_allclose(entropy(np.cumsum(a[:, state]), np.cumsum(b[:, state])),
            curves[f"fixed_{state}"], rtol=0, atol=2e-10,
            err_msg="Replayed candidate spectra differ from saved entropy evidence")
    if len(a) != len(times):
        raise ValueError("Common-region frame counts differ")
    return moment_file


def _solve_prefix(cache_path, stop, tolerance, seconds):
    """Independent numerical job; the parent alone writes checkpoints."""
    with np.load(Path(cache_path) / "moments.npz", allow_pickle=False) as data:
        a, b, transition = data["a"], data["b"], data["transition"]
    result = optimize_prefixes(a, b, transition, [stop], tolerance_bits=tolerance,
        label_limit=100_000 if stop <= 64 else 0, time_limit_s=seconds)[0]
    if result["path"] is not None:
        result["path"] = result["path"].tolist()
    return result


def solve_records(records, cache, points, workers, tolerance, seconds):
    """Schedule independent prefixes globally, then serialize checked results."""
    cache = Path(cache)
    payloads, grids, pending = {}, {}, []
    for record in records:
        dataset = record["dataset"]
        moment_file = cache / dataset / "moments.npz"
        digest = wav.sha256(moment_file)
        with np.load(moment_file, allow_pickle=False) as data:
            stops = prefix_stops(len(data["a"]), points)
        checkpoint = moment_file.with_name("prefix_optima.json")
        payload = {"moment_sha256": digest, "tolerance_bits": tolerance, "results": {}}
        if checkpoint.is_file():
            payload = json.loads(checkpoint.read_text())
            if payload["moment_sha256"] != digest or payload["tolerance_bits"] != tolerance:
                raise ValueError("Oracle checkpoint configuration differs")
        payloads[dataset], grids[dataset] = payload, stops
        todo = [(dataset, int(stop)) for stop in stops if str(stop) not in payload["results"]]
        pending.extend(todo)
        if not todo:
            print(f"COMPLETE JOINT ORACLE: {dataset}", flush=True)
    if not pending:
        return
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(_solve_prefix, str(cache / dataset), stop, tolerance, seconds):
                (dataset, stop) for dataset, stop in pending}
        for job in as_completed(jobs):
            dataset, stop = jobs[job]
            try:
                result = job.result()
            except Exception as error:
                failures.append(f"{dataset}, prefix {stop}: {error}")
                print("UNFINISHED " + failures[-1], flush=True)
                continue
            payload = payloads[dataset]
            payload["results"][str(stop)] = result
            checkpoint = cache / dataset / "prefix_optima.json"
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload) + "\n")
            temporary.replace(checkpoint)
            count = sum(str(value) in payload["results"] for value in grids[dataset])
            if count % 8 == 0 or count == len(grids[dataset]):
                print(f"{dataset}: {count}/{len(grids[dataset])} joint-entropy prefix optima certified", flush=True)
            if count == len(grids[dataset]):
                print(f"COMPLETE JOINT ORACLE: {dataset}", flush=True)
    if failures:
        raise RuntimeError("Some prefixes need further optimization; certified checkpoints retained:\n"
                           + "\n".join(failures))


def replace_joint_spec(spec, record, curves, times, results):
    """Change only the 2-D curve arrays, objective, and matching labels."""
    positions = np.array([entry["stop"] - 1 for entry in results], dtype=int)
    oracle = np.array([entry["entropy_bits"] for entry in results])
    tolerance = max(entry["gap_bits"] for entry in results)
    fixed_min = min(float(np.nanmin(values[positions] - oracle))
                    for method, values in curves.items() if method.startswith("fixed_"))
    if fixed_min < -2e-10:
        raise AssertionError("A fixed path beats the claimed joint-entropy oracle")
    info = {"objective": JOINT_OBJECTIVE, "plotted_prefixes": len(results),
            "total_prefixes": len(times), "sampling": "uniform frame grid including first and last",
            "maximum_certified_gap_bits": tolerance, "minimum_fixed_regret_bits": fixed_min,
            "scope": "discrete window schedules; blended spectra need not belong to this class"}
    record = {**record, "joint_entropy_oracle": info,
              "final_joint_regret_bits": float(curves["posterior_blend"][-1] - oracle[-1])}
    panel = {**spec["panels"][0], "lines": []}
    for line in spec["panels"][0]["lines"]:
        if line["method"]:
            panel["lines"].append({**line, "x": times[positions],
                "y": curves[line["method"]][positions] - oracle})
        else:
            panel["lines"].append({**line, "label": "Zero regret"})
    updated = {"metadata": {**spec["metadata"], "joint_entropy_oracle": info,
                "hindsight_objective": JOINT_OBJECTIVE,
                "entropy_comparison": "joint H3 minus its separately optimized prefix schedule"},
               "panels": [panel]}
    return updated, record


def rebuild(source, output_dir=None, wav_dir=None, cache=None, points=65, workers=2,
            tolerance=1e-6, seconds=120):
    source = Path(source).resolve()
    cache = Path(cache or "/private/tmp/ustft-joint-cache/collection").resolve()
    wav_dir = Path(wav_dir or ROOT / "benchmarks/wav").resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    for relative, digest in manifest["sha256"].items():
        path = source / relative
        if source not in path.resolve().parents or wav.sha256(path) != digest:
            raise ValueError(f"Changed input evidence: {relative}")
    inventory = json.loads((source / "data/wav_inventory.json").read_text())
    for position, record in enumerate(inventory, 1):
        prepare_moments(source, record, cache, wav_dir)
        print(f"Verified moments {position}/{len(inventory)}: {record['filename']}", flush=True)
    solve_records(inventory, cache, points, workers, tolerance, seconds)
    plot_files = {relative for relative in manifest["evidence"]}
    plot_files |= {str(Path(relative).with_suffix(".npz")) for relative in plot_files}
    records = []
    with figure_edition(output_dir or source.parent,
            "Four figures per WAV; independent joint-H3 prefix hindsight comparators") as edition:
        for relative in manifest["sha256"]:
            if relative not in plot_files:
                target = edition.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
        for record in inventory:
            dataset = record["dataset"]
            specs = {Path(relative).stem: load_evidence(source / relative)
                     for relative in manifest["evidence"] if Path(relative).parts[1] == dataset}
            curves, times = _raw_curves(source, record)
            payload = json.loads((cache / dataset / "prefix_optima.json").read_text())
            results = [payload["results"][str(stop)] for stop in prefix_stops(len(times), points)]
            specs["03_regret_2d_renyi3"], record = replace_joint_spec(
                specs["03_regret_2d_renyi3"], record, curves, times, results)
            for name, spec in wav.compact_specs(specs, record).items():
                edition.save(spec, name + ".png")
            folder = edition.root / "data" / dataset
            shutil.copytree(cache / dataset, folder / "joint_entropy_oracle")
            rows = []
            for result in results:
                index = result["stop"] - 1
                for method, values in curves.items():
                    rows.append({"time_s": float(times[index]), "prefix_frames": result["stop"],
                        "method": method, "method_joint_h3_bits": float(values[index]),
                        "optimal_joint_h3_bits": result["entropy_bits"],
                        "regret_bits": float(values[index] - result["entropy_bits"]),
                        "oracle_optimality_gap_bits": result["gap_bits"]})
            wav.write_csv(folder / "joint_entropy_regret.csv", rows)
            # Keep the previous comparator's diagnostics, with explicit names.
            for old, new in (("all_method_entropy_regret.csv", "tracking_entropy_diagnostics.csv"),
                             ("entropy_regret.csv", "tracking_entropy_summary.csv")):
                old_path = folder / old
                if old_path.is_file():
                    old_path.rename(folder / new)
            (folder / "input_and_ranking.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
        (edition.root / "data/wav_inventory.json").write_text(json.dumps(records, indent=2) + "\n")
        if len(edition.entries) != 4 * len(records):
            raise AssertionError("Expected exactly four figures per WAV")
    with (edition.root / "README.md").open("a") as stream:
        stream.write("\n## Separate joint 2-D entropy hindsight comparator\n\n" + JOINT_OBJECTIVE + "\n\n")
        stream.write(f"Each 2-D curve has {points} uniformly spaced evaluated prefixes (or all prefixes "
                     "for shorter fixtures), including the first and last. Lines join these evaluated "
                     "points; unevaluated frames are not claimed to have been optimized. The entire "
                     "prefix enters each objective. The entropy is not summed or averaged again.\n\n")
        stream.write(f"Every oracle is certified within {tolerance:g} bits. Exact moment Pareto labels "
                     "are used where tractable; supporting-line bounds and integer outer approximation "
                     "certify larger problems. All paths and certificates are in each dataset's "
                     "joint_entropy_oracle folder; joint_entropy_regret.csv contains the plotted data "
                     "and all five fixed windows. No framewise greedy surrogate is used.\n\n")
        stream.write("All fixed windows are feasible and have nonnegative regret up to floating-point "
                     "roundoff. U-STFT blends coefficients, so its output is outside this discrete-path "
                     "comparator class; its entropy difference is not forcibly made nonnegative.\n\n")
        stream.write("The U-STFT algorithm, 1-D comparisons, spectrograms, selected windows, blend "
                     "PMFs, frequency views, and excerpt lengths are unchanged. The 1-D panels still "
                     "use the paper's standardized time-plus-frequency hindsight schedule: " + wav.OBJECTIVE + "\n\n")
        stream.write("The original tracking-comparator CSVs are retained as tracking_entropy_diagnostics.csv "
                     "and tracking_entropy_summary.csv. Their historical joint_regret columns describe "
                     "the old tracking-objective comparison; the displayed 2-D regrets are exclusively "
                     "in joint_entropy_regret.csv. Full paper sweep evidence is preserved in data/audit.\n\n")
        stream.write(f"Source: `{source}`. Rebuild this edition's plots without optimization using "
                     f"`python -m benchmarks.wav_collection --from-edition '{edition.root}'`.\n")
    with (source / "README.md").open("a") as stream:
        stream.write(f"\nSuperseded by `{edition.root}` with separate joint-entropy prefix optima; prior evidence retained.\n")
    print(f"COMPLETE JOINT-ENTROPY WAV COLLECTION: {edition.root}", flush=True)
    return edition.root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-edition", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--points", type=int, default=65)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tolerance-bits", type=float, default=1e-6)
    parser.add_argument("--seconds-per-prefix", type=float, default=120)
    args = parser.parse_args()
    rebuild(args.from_edition, args.output_dir, args.wav_dir, args.cache,
            args.points, args.workers, args.tolerance_bits, args.seconds_per_prefix)


if __name__ == "__main__":
    main()
