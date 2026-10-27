"""Chapter figure contracts, including colour-model reference cases."""

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmarks import plot_style as style
from benchmarks.figure_export import (
    _check_text, _message, _zoom, capture_figure, figure_edition,
    load_evidence, render_figure, save_spec,
)
from benchmarks.plot_suites import curve_panel, power_panels


@pytest.mark.parametrize("first,second,expected", [
    ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
    ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
    ((50, 0, 0), (50, -1, 2), 2.3669),
    ((50, 2.49, -0.001), (50, -2.49, 0.0009), 7.1792),
])
def test_sharma_ciede2000_reference_pairs(first, second, expected):
    assert style.ciede2000(first, second) == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("methods", style.COOCCURRENCE)
def test_colour_thresholds_for_actual_comparisons(methods):
    style.check_palette(methods)


def test_named_palette_search_retains_best_fixed_and_maximises_separation():
    chosen = style.search_named_palette(7, pinned=("#F99379",))
    expected = {style.PALETTE[key] for key in
                ("best_fixed_support", "posterior_blend", "fixed_0", "fixed_1", "fixed_2", "fixed_3", "fixed_4")}
    assert set(chosen) == expected
    normal, protan, deutan = style.palette_distances(chosen)
    pairs = np.triu_indices(len(chosen), 1)
    assert np.min(normal[pairs]) >= 28
    assert np.min(protan[pairs]) >= 15
    assert np.min(deutan[pairs]) >= 15


def test_black_roles_equal_width_and_unique_dash_patterns():
    assert len(set(style.PALETTE.values())) == len(style.PALETTE)
    assert style.PALETTE["best_fixed_support"] == "#F99379"
    assert style.LINESTYLES["best_fixed_support"] == (0, (7, 2, 2, 2))
    for key in style.METHOD_ORDER:
        assert (style.PALETTE[key] == "#000000") == (style.ROLES[key] == "ORACLE")
        if style.LINESTYLES[key] == "-":
            assert style.ROLES[key] in {"ORACLE", "BASELINE"}
        settings = style.line_style(key, offset=1, count=4, points=100)
        assert settings["linewidth"] == style.LINE_WIDTH
        assert settings["markerfacecolor"] == style.PALETTE[key]
        assert settings["markeredgecolor"] == "white"
        assert style.line_style(key, slides=True)["marker"] == "None"
    for group in style.COOCCURRENCE:
        patterns = [style.LINESTYLES[key] for key in group]
        assert len({repr(value) for value in patterns}) == len(group)
    starts = [style.line_style(key, offset=i, count=4, points=100)["markevery"][0]
              for i, key in enumerate(style.COOCCURRENCE[0])]
    assert len(set(starts)) == 4
    with pytest.raises(TypeError):
        style.PALETTE["posterior_blend"] = "#000000"


def test_palette_rejects_indistinguishable_pairs_and_framewise_is_not_oracle():
    with pytest.raises(ValueError, match="threshold"):
        style.check_palette(("framewise_selection", "fixed_4"))
    assert style.method_from_label("Framewise minimum") == "framewise_selection"
    assert style.ROLES["framewise_selection"] == "METHOD"
    assert style.method_from_label("Path-code bound") is None


def test_shared_power_reference_includes_all_methods():
    panels = power_panels({"posterior_blend": np.ones((4, 5)),
                          "fixed_0": np.full((4, 5), 0.1)}, 1000, 16)
    assert panels[0]["images"][0]["clim"] == panels[1]["images"][0]["clim"]
    assert np.allclose(panels[0]["images"][0]["values"] - panels[1]["images"][0]["values"], 20)


def test_messages_are_checked_against_plotted_values_and_ties():
    x = np.arange(1, 6, dtype=float)
    curves = {"posterior_blend": np.full(5, 3.0), "framewise_selection": np.full(5, 1.0),
              "best_fixed_support": np.full(5, 2.0)}
    p = curve_panel("Comparison", x, curves, "Entropy")
    assert _message([p]) == "Framewise selection ends lowest at 1, then Best fixed window at 2."
    for line in p["lines"]:
        line["y"] = np.ones(5)
    assert _message([p]) == "All 3 methods end at 1."
    regret = {**p["lines"][0], "label": "Average regret", "method": None, "y": np.array([3, 1, 1, 1, 1])}
    bound = {**p["lines"][1], "label": "Path-code bound", "method": None, "y": np.full(5, 2.0)}
    p["lines"] = [regret, bound]
    assert "within" not in _message([p])
    regret["y"][0] = 1
    assert "within" in _message([p])
    p["images"] = [{"values": np.array([[0.2, 0.8]]), "label": "Probability"}]
    assert _message([p]) == "The largest posterior probability is 0.8 in this run."


def test_rendered_curve_styles_legend_and_slide_labels():
    x = np.linspace(0.01, 5, 80)
    p = curve_panel("Prefix comparison", x, {
        key: 2 + np.sin(x) + index * 0.1 for index, key in enumerate(style.COOCCURRENCE[0])}, "Entropy (bits)")
    meta = {"dataset": "test", "trials": "single run", "hop": 16}
    for slides in (False, True):
        fig = render_figure([p], meta, _message([p]), slides)
        try:
            _check_text(fig)
            axis = fig.axes[0]
            assert not axis.spines["top"].get_visible()
            assert not axis.spines["right"].get_visible()
            assert [line.get_label() for line in axis.lines] == [style.LABELS[key] for key in style.COOCCURRENCE[0]]
            assert all(line.get_linewidth() == style.LINE_WIDTH for line in axis.lines)
            assert all(text.get_fontweight() == "normal" for text in axis.texts)
            if slides:
                assert not fig.legends and axis.get_legend() is None
                assert len(axis.get_xticks()) == len(axis.get_yticks()) == 2
                assert fig._suptitle.get_fontsize() == 28
                assert len(fig._suptitle.get_text().splitlines()) <= 2
                assert all(text.get_fontsize() == 18 for text in axis.texts)
                assert all(text.get_position()[0] > 1 for text in axis.texts)
                assert not any(line.get_visible() for line in axis.get_xgridlines() + axis.get_ygridlines())
            else:
                assert fig._suptitle is None
                legend = fig.legends[0]
                assert legend.get_window_extent().width <= fig.bbox.width
                assert legend.get_window_extent().y1 < axis.get_window_extent().y0
        finally:
            plt.close(fig)


def test_slide_message_wrap_uses_rendered_text_width():
    p = curve_panel("Regret", np.arange(1, 5), {"posterior_blend": np.full(4, .134)}, "Regret")
    message = "The regret stays within its path-code bound (0.134 at the end, versus 0.135)."
    fig = render_figure([p], {"dataset": "test", "trials": "single run"}, message, slides=True)
    try:
        _check_text(fig)
        assert len(fig._suptitle.get_text().splitlines()) == 2
        assert fig._suptitle.get_window_extent().x1 <= fig.bbox.width * .95
    finally:
        plt.close(fig)


def test_startup_zoom_and_distant_reference_do_not_stretch_axis():
    x = np.arange(1, 101, dtype=float)
    y = np.r_[1000, np.full(99, 1.0)]
    p = curve_panel("Cumulative average", x, {"posterior_blend": y}, "Average loss")
    zoom = _zoom(p)
    assert zoom["xlim"][0] == 6
    assert zoom["ylim"][1] < 2
    p["lines"].append({"x": np.array([0, 1]), "y": np.array([2000, 2000]),
                       "method": None, "label": "Bound", "reference_axis": "horizontal", "linestyle": ":"})
    fig = render_figure([p], {"dataset": "test", "trials": "single run"}, "A plotted reference lies above the axis.")
    try:
        assert fig.axes[0].get_ylim()[1] < 2
        assert any("above axis" in text.get_text() for text in fig.legends[0].get_texts())
    finally:
        plt.close(fig)


def test_fresh_editions_preserve_arrays_and_have_all_format_twins(tmp_path):
    x = np.arange(1, 5, dtype=float)
    p = curve_panel("Metric", x, {"posterior_blend": np.array([3, 2, 1, 1])}, "Loss")
    spec = {"metadata": {"dataset": "test", "trials": "single run", "hop": 16}, "panels": [p]}
    with figure_edition(tmp_path, "Deterministic plotting fixture") as first:
        save_spec(spec, "metric.png")
    with figure_edition(tmp_path, "Deterministic plotting fixture") as second:
        save_spec(spec, "metric.png")
    assert first.root != second.root
    loaded = load_evidence(first.root / "data/test/metric.json")
    assert np.array_equal(loaded["panels"][0]["lines"][0]["y"], p["lines"][0]["y"])
    for edition in (first, second):
        assert "0 errors; 0 warnings" in (edition.root / "build.log").read_text()
        for directory in ("dissertation", "slides"):
            assert (edition.root / directory / "test/metric.png").is_file()
            assert (edition.root / directory / "test/metric.pdf").is_file()
    manifest = json.loads((first.root / "manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((first.root / relative).read_bytes()).hexdigest() == expected


def test_capture_keeps_arrays_but_ignores_visual_overrides_and_annotations():
    fig, axis = plt.subplots()
    values = np.array([1.0, np.nan, 3.0])
    axis.plot([0, 1, 2], values, color="black", lw=9, label="U-STFT")
    axis.text(0.5, 0.5, "Unverified claim", transform=axis.transAxes)
    try:
        captured = capture_figure(fig, {"dataset": "test", "trials": "single run"})
        line = captured["panels"][0]["lines"][0]
        assert np.array_equal(line["y"], values, equal_nan=True)
        assert line["method"] == "posterior_blend"
        assert "color" not in line and "linewidth" not in line
        assert "Unverified claim" not in repr(captured)
    finally:
        plt.close(fig)


def test_capture_retains_probability_units_and_identifies_posterior_overlay():
    fig, axes = plt.subplots(2, 1)
    axes[0].plot([0, 1], [0, 1], color="#D55E00", label="Hindsight schedule")
    axes[1].set_title("Posterior")
    image = axes[1].imshow([[.2, .8]], vmin=0, vmax=1)
    fig.colorbar(image, ax=axes[1]).set_label("Probability")
    axes[1].plot([0, 1], [0, .6], color="#D55E00")
    try:
        captured = capture_figure(fig, {"dataset": "test", "trials": "single run"})
        posterior = captured["panels"][1]
        assert posterior["lines"][0]["method"] == "posterior_blend"
        assert posterior["images"][0]["label"] == "Probability"
        assert _message([posterior]) == "The largest posterior probability is 0.8 in this run."
    finally:
        plt.close(fig)


def test_map_export_preserves_dataset_and_method_subfolders(tmp_path):
    metadata = {"dataset": "test-dataset", "trials": "single run", "hop": 16}
    panels = power_panels({"posterior_blend": np.ones((5, 9)),
                          "fixed_0": np.full((5, 9), 0.5)}, 1000, 16)
    with figure_edition(tmp_path, "Plot-only fixture") as edition:
        save_spec({"metadata": metadata, "panels": panels}, "maps.png")
    for directory in ("dissertation", "slides"):
        root = edition.root / directory / "test-dataset"
        assert (root / "maps.png").is_file()
        assert (root / "methods" / "posterior_blend" / "maps.png").is_file()
        assert (root / "methods" / "fixed_0" / "maps.png").is_file()


def test_legacy_spectrogram_helpers_still_append_pdf_pages(tmp_path):
    from matplotlib.backends.backend_pdf import PdfPages
    from benchmarks.plotting import plot_single_spectrogram, show_mel_individual
    with figure_edition(tmp_path / "editions", "Legacy PDF plotting fixture"):
        with PdfPages(tmp_path / "report.pdf") as pdf:
            data = np.full((4, 5), 0.5)
            plot_single_spectrogram(data, "Power fixture", pdf, 1000, 16)
            show_mel_individual(data, "Mel fixture", pdf, 1000, 16, np.arange(6) * 100)
            assert pdf.get_pagecount() == 2
    assert (tmp_path / "report.pdf").read_bytes().startswith(b"%PDF")
