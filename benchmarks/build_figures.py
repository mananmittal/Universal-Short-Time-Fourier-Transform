"""Rebuild a complete paired chapter edition from its stored plotting arrays.

python -m benchmarks.build_figures --from-edition results/figures/<edition>

This command performs no STFT evaluation or download. Initial editions are
produced by the existing example/benchmark commands with the shared exporter.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from benchmarks.figure_export import figure_edition, load_evidence


def rebuild(source, output_dir=None):
    source = Path(source).expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    for relative, digest in manifest["sha256"].items():
        path = source / relative
        if source not in path.resolve().parents:
            raise ValueError("Evidence path leaves the source edition")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Evidence checksum changed: {path}")
    with figure_edition(output_dir or source.parent, f"Cached plotted arrays from {source}") as edition:
        plotted = {relative for name in manifest["evidence"]
                   for relative in (name, str(Path(name).with_suffix(".npz")))}
        # Keep the audit, WAV inventory, raw metric CSVs and input provenance
        # beside a replayed collection. The plot cache itself is regenerated.
        for relative in manifest["sha256"]:
            if relative not in plotted:
                target = edition.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
        for relative in manifest["evidence"]:
            path = source / relative
            if relative not in manifest["sha256"] or str(Path(relative).with_suffix(".npz")) not in manifest["sha256"]:
                raise ValueError(f"Unverified evidence: {relative}")
            edition.save(load_evidence(path), path.stem + ".png")
    extra = (source / "README.md").read_text().partition("## WAV coverage and metric definitions")[2]
    if extra:
        # Preserve the collection's metric definitions; remove only previous
        # supersession annotations from the copied README tail.
        extra = extra.split("\nSuperseded by ", 1)[0]
        with (edition.root / "README.md").open("a") as stream:
            stream.write("\n## WAV coverage and metric definitions" + extra)
    # Only the README is annotated; earlier figures and numerical evidence are
    # immutable. Failed builds never supersede a completed edition.
    readme = source / "README.md"
    with readme.open("a") as stream:
        stream.write(f"\nSuperseded by `{edition.root}`; figures and evidence retained.\n")
    print(f"Wrote paired chapter figures to {edition.root}")
    return edition.root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-edition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Parent for a new dated edition")
    arguments = parser.parse_args()
    rebuild(arguments.from_edition, arguments.output_dir)


if __name__ == "__main__":
    main()
