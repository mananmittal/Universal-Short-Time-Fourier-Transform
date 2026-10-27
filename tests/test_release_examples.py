import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import run_paper


ROOT = Path(__file__).resolve().parents[1]
PLOT_SOURCES = (
    ROOT / "examples" / "example.py",
    ROOT / "examples" / "example.ipynb",
    ROOT / "examples" / "example_torch.py",
    ROOT / "examples" / "example_torch.ipynb",
    ROOT / "benchmarks" / "evaluate.py",
)
EXAMPLE_PAIRS = (
    (ROOT / "examples" / "example.py", ROOT / "examples" / "example.ipynb"),
    (
        ROOT / "examples" / "example_torch.py",
        ROOT / "examples" / "example_torch.ipynb",
    ),
)


def _plot_source(path):
    if path.suffix != ".ipynb":
        return path.read_text()
    notebook = json.loads(path.read_text())
    return "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_running_2d_renyi_is_not_cumulatively_summed_again():
    for path in PLOT_SOURCES:
        source = _plot_source(path)
        assert "np.cumsum(cand_2d_renyi_over_time" not in source
        assert "np.cumsum(u_renyi_2d_run" not in source
        assert "Prefix 2-D Rényi Entropy" in source
        assert "running_2d_renyi_torch" not in source
        assert "inverse_universal_spectrogram_torch" not in source


def test_notebooks_match_scripts_on_release_critical_lines():
    markers = (
        "duration = ",
        "beta = ",
        "running_2d_renyi(",
        "inverse_universal_spectrogram(",
        "Prefix 2-D Rényi Entropy",
    )
    for script_path, notebook_path in EXAMPLE_PAIRS:
        script_lines = [
            line.strip() for line in _plot_source(script_path).splitlines()
            if any(marker in line for marker in markers)
        ]
        notebook_lines = [
            line.strip() for line in _plot_source(notebook_path).splitlines()
            if any(marker in line for marker in markers)
        ]
        assert notebook_lines == script_lines


def test_paper_greedy_path_is_the_per_frame_metric_argmin():
    rng = np.random.default_rng(20260808)
    x = rng.standard_normal(512)

    for spacing in ("geometric", "linear"):
        case = run_paper._case_outputs(
            x, n_fft=128, hop_size=16, beta=1.5, spacing=spacing
        )
        expected = np.nanargmin(case["candidate_costs"], axis=1)
        warmup = case["warmup"]
        common_costs = case["candidate_costs"][warmup:]
        eta = 1.5 / (np.std(common_costs, axis=1) + 1e-6)
        expected_fixed_totals = np.sum(
            eta[:, None]
            * (common_costs - np.min(common_costs, axis=1, keepdims=True)),
            axis=0,
        )

        assert np.array_equal(case["framewise_indices_full"], expected)
        assert np.array_equal(case["framewise_indices"], expected[warmup:])
        assert np.array_equal(case["greedy"][2], case["lengths"][expected])
        assert np.allclose(
            case["fixed_tracking_totals"], expected_fixed_totals, atol=1e-12
        )
        assert case["fixed_index"] == int(
            np.argmin(expected_fixed_totals)
        )
        assert np.allclose(
            case["best_fixed"][0][warmup:],
            case["candidate_specs"][case["fixed_index"], warmup:],
            rtol=2e-13,
            atol=2e-13,
        )


def test_paper_joint_record_separates_fixed_selection_rules():
    x = np.random.default_rng(7).standard_normal(512)
    case = run_paper._case_outputs(
        x, n_fft=128, hop_size=16, beta=1.5, spacing="geometric"
    )
    record = run_paper._joint_renyi3_case_record(
        "Test", "Test", "test", case, fs=1024, n_fft=128, hop_size=16,
        case_type="controlled_clean",
    )

    assert set(record["metrics"]) == {
        "posterior_blend",
        "framewise_selection",
        "best_fixed_support",
        "constrained_schedule",
    }
    assert record["best_fixed_index"] == case["fixed_index"]
    assert record["best_fixed_length"] == case["lengths"][case["fixed_index"]]
    assert record["joint_optimal_fixed_length"] in case["lengths"]
    assert "scaled Gibbs emission" in record["fixed_selection_rule"]


def test_canonical_paper_figure_manifest(tmp_path):
    names = run_paper._CANONICAL_PAPER_FIGURES
    assert len(names) == 20
    assert len(set(names)) == len(names)

    for name in names[:-1]:
        (tmp_path / name).touch()
    with pytest.raises(RuntimeError, match=names[-1]):
        run_paper._require_canonical_paper_figures(tmp_path)

    (tmp_path / names[-1]).touch()
    paths = run_paper._require_canonical_paper_figures(tmp_path)
    assert [path.name for path in paths] == list(names)
