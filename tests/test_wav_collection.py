"""Numerical and presentation contracts for the all-WAV entropy collection."""

import wave
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmarks import run_paper as paper
from benchmarks import wav_collection
from benchmarks.figure_export import _check_text, _message, _zoom, figure_edition, render_figure, save_spec
from benchmarks.plot_suites import curve_panel, power_panels, posterior_panel
from benchmarks.wav_collection import (
    diagnostics, extract_case, joint_prefix_bits, read_excerpt, regret_panel,
    refine_cached_collection, spectrogram_views,
)


@pytest.mark.parametrize("dtype,width", [(np.int16, 2), (np.int32, 4), (np.uint8, 1)])
def test_excerpt_matches_paper_normalization_and_reads_five_seconds(tmp_path, dtype, width):
    fs = 32
    samples = np.arange(6 * fs * 2).reshape(-1, 2).astype(dtype)
    path = tmp_path / "recording.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(width)
        stream.setframerate(fs)
        stream.writeframes(samples.tobytes())
    actual, provenance = read_excerpt(path)
    paper_fs, expected = paper._read_wav_mono(path)
    assert paper_fs == fs
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert len(actual) == 5 * fs
    assert provenance["original_duration_s"] == 6
    assert provenance["excerpt_end_s"] == 5
    assert provenance["channels"] == 2


def test_test_wav_keeps_full_length(tmp_path):
    path = tmp_path / "mixed_test_signal.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(32)
        stream.writeframes(np.arange(15 * 32, dtype=np.int16).tobytes())
    actual, provenance = read_excerpt(path)
    assert len(actual) == 15 * 32
    assert provenance["excerpt_end_s"] == 15


def test_joint_prefix_uses_full_frequency_cells_and_handles_initial_silence():
    # The complete DFT power is [DC, interior, Nyquist, interior].
    spectra = np.array([[0, 0, 0], [1, 2, 3], [2, 1, 4]], dtype=complex)
    result = joint_prefix_bits(spectra, 4)
    assert np.isnan(result[0])
    for stop in (2, 3):
        power = np.abs(spectra[1:stop, [0, 1, 2, 1]]) ** 2
        pmf = power / power.sum()
        expected = -.5 * np.log2(np.sum(pmf ** 3))
        assert result[stop - 1] == pytest.approx(expected, abs=1e-13)
    np.testing.assert_allclose(joint_prefix_bits(7 * spectra, 4)[1:], result[1:], atol=1e-13)
    assert np.isnan(joint_prefix_bits(np.zeros((3, 3)), 4)).all()


def test_all_measures_follow_the_paper_schedule_and_actual_blended_entropy():
    fs, hop, n_fft = 128, 4, 64
    time = np.arange(256) / fs
    x = np.sin(2 * np.pi * 9 * time)
    x[::37] += 2
    case = paper._case_outputs(x, n_fft, hop, 1.5, "geometric")
    arrays = extract_case(case)
    start = case["warmup"]
    metrics = diagnostics(arrays, start, n_fft, hop, fs)
    rows = np.arange(len(arrays["comparator_path"]))[start:]
    path = case["hmm_audit"]["comparator_path"][start:]
    expected_2d = (joint_prefix_bits(case["trellis"][0][start:], n_fft)
                   - joint_prefix_bits(case["constrained"][0][start:], n_fft))
    np.testing.assert_allclose(metrics["joint_regret"], expected_2d, atol=1e-13)
    expected_time = np.cumsum(case["blended_h_time"][start:]
                             - case["candidate_h_time"][rows, path])
    np.testing.assert_allclose(metrics["components"]["time"]["regret"], expected_time)
    np.testing.assert_allclose(metrics["components"]["total"]["regret"],
        metrics["components"]["time"]["regret"] + metrics["components"]["frequency"]["regret"], atol=1e-12)
    denominator = np.arange(1, len(rows) + 1)
    np.testing.assert_array_equal(metrics["average_count"], denominator)
    expected_methods = {"posterior_blend", *(f"fixed_{i}" for i in range(len(arrays["lengths"])))}
    assert set(metrics["joint_regrets"]) == expected_methods
    for index, spectrum in enumerate(arrays["candidate_specs"]):
        expected = joint_prefix_bits(spectrum[start:], n_fft)
        np.testing.assert_allclose(metrics["fixed_joint"][index], expected)
        np.testing.assert_allclose(metrics["joint_regrets"][f"fixed_{index}"],
                                   expected - metrics["joint_hindsight"])
    for name, candidate_key in (("time", "candidate_h_time"),
                                ("frequency", "candidate_h_spec"),
                                ("total", "candidate_costs")):
        component = metrics["components"][name]
        assert set(component["average_regrets"]) == expected_methods
        for key in expected_methods:
            values = (component["universal"] if key == "posterior_blend"
                      else arrays[candidate_key][rows, int(key.split("_")[1])])
            expected = np.cumsum(values - component["hindsight"])
            np.testing.assert_allclose(component["cumulative_regrets"][key], expected)
            np.testing.assert_allclose(component["average_regrets"][key], expected / denominator)
            np.testing.assert_allclose(metrics["components"]["total"]["average_regrets"][key],
                metrics["components"]["time"]["average_regrets"][key]
                + metrics["components"]["frequency"]["average_regrets"][key], atol=1e-12)
    ranking = np.argsort(metrics["fixed_h3_bits"], kind="stable")
    assert metrics["selected"] == {"best": ranking[0], "middle": ranking[len(ranking) // 2], "worst": ranking[-1]}
    bad = {**arrays, "comparator_path": arrays["comparator_path"].copy()}
    bad["comparator_path"][0] = 1
    with pytest.raises(ValueError, match="start/state"):
        diagnostics(bad, start, n_fft, hop, fs)
    bad = {**arrays, "pmf": arrays["pmf"].copy()}
    bad["pmf"][0] *= 2
    with pytest.raises(ValueError, match="probability"):
        diagnostics(bad, start, n_fft, hop, fs)


def test_export_has_four_combined_figures_and_ranked_windows(tmp_path, monkeypatch):
    fs, hop, n_fft = 128, 2, 64
    time = np.arange(256) / fs
    x = np.sin(2 * np.pi * 9 * time)
    x[::37] += 2
    case = paper._case_outputs(x, n_fft, hop, 1.5, "geometric")
    arrays = extract_case(case)
    monkeypatch.setattr(wav_collection, "N_FFT", n_fft)
    monkeypatch.setattr(wav_collection, "HOP", hop)
    saved = {}
    monkeypatch.setattr(wav_collection, "save_spec", lambda spec, name: saved.setdefault(name, spec))
    wav_collection.save_wav_figures(arrays,
        {"filename": "fixture.wav", "sample_rate_hz": fs}, case["warmup"],
        SimpleNamespace(root=tmp_path))
    metrics = diagnostics(arrays, case["warmup"], n_fft, hop, fs)
    keys = {"posterior_blend", *(f"fixed_{index}" for index in metrics["selected"].values())}
    assert set(saved) == {name + ".png" for name in wav_collection.COMPACT_FIGURES}
    assert all(spec["metadata"]["keep_panels_together"]
               and not spec["metadata"]["export_individual_methods"]
               and not spec["metadata"]["export_difference"] for spec in saved.values())
    maps = saved["01_spectrograms.png"]["panels"]
    assert len(maps) == 4
    assert {panel["title"] for panel in maps} == {wav_collection.LABELS[key] for key in keys}
    assert not any("blend_weights_curves" in name for name in saved)
    panels = saved["02_regret_time_frequency_total.png"]["panels"]
    assert len(panels) == 3
    for name, item in zip(("time", "frequency", "total"), panels):
        assert item["title"].lower() == name
        assert _zoom(item)["xlim"][0] > item["xlim"][0]
        assert {line["method"] for line in item["lines"] if line["method"]} == keys
        for line in item["lines"]:
            if line["method"]:
                np.testing.assert_allclose(line["y"], metrics["components"][name]["average_regrets"][line["method"]])
    joint = saved["03_regret_2d_renyi3.png"]["panels"][0]
    for line in joint["lines"]:
        if line["method"]:
            np.testing.assert_allclose(line["y"], metrics["joint_regrets"][line["method"]])
    for slides in (False, True):
        for spec in saved.values():
            panels, meta = spec["panels"], spec["metadata"]
            fig = render_figure(panels, meta, _message(panels), slides)
            try:
                _check_text(fig)
            finally:
                plt.close(fig)


def test_rank_labels_and_regret_objective_fit_both_formats():
    maps = power_panels({"posterior_blend": np.ones((10, 33)),
        "fixed_0": np.ones((10, 33)) * .1, "fixed_2": np.ones((10, 33)) * .5,
        "fixed_4": np.ones((10, 33)) * .9}, 128, 4)
    maps[1]["display_title"] = "Fixed H (best; L = 128)"
    maps[2]["display_title"] = "Fixed 4H (middle; L = 512)"
    maps[3]["display_title"] = "Fixed 16H (worst; L = 2048)"
    curve = regret_panel("Cumulative frequency Rényi-3 regret", np.linspace(.2, 5, 100),
                         np.linspace(-.5, 2, 100), "Frequency H3 regret (norm. sum)")
    meta = {"dataset": "test", "trials": "single run", "hop": 4}
    for slides in (False, True):
        for panels in (maps, [curve]):
            fig = render_figure(panels, meta, _message(panels), slides)
            try:
                _check_text(fig)
                if panels is maps:
                    assert any("best" in axis.get_title(loc="left" if slides else "center") for axis in fig.axes)
                    assert all(image.get_clim() == (-70, 0) for axis in fig.axes for image in axis.images)
                else:
                    assert "hindsight: min scaled time + frequency H3" in fig.axes[0].get_xlabel()
            finally:
                plt.close(fig)


def test_weight_messages_identify_largest_weights_and_axes_keep_probability_bounds():
    weights = np.array([[1, 0, 0], [.2, .3, .5]])
    heatmap = posterior_panel(weights, np.array([128, 256, 512]), 1)
    assert _message([heatmap]) == "At the end, the blend gives L=512 weight 0.5, then L=256 weight 0.3."
    p = curve_panel("Blend weights", np.array([.1, 1]), {
        "fixed_0": weights[:, 0], "fixed_1": weights[:, 1], "fixed_2": weights[:, 2],
    }, "Blend weight")
    assert _message([p]) == "Fixed 4H ends highest at 0.5, then Fixed 2H at 0.3."
    assert _zoom(p)["ylim"] == [0, 1]
    for slides in (False, True):
        fig = render_figure([p], {"dataset": "test", "trials": "single run"}, _message([p]), slides)
        try:
            _check_text(fig)
            assert fig.axes[0].get_ylim() == (0, 1)
        finally:
            plt.close(fig)


def test_two_line_posterior_message_clears_the_map_heading():
    weights = np.array([[1, 0, 0, 0, 0], [.00202, .000003, .0001, .00001, .997867]])
    heatmap = posterior_panel(weights, np.array([128, 256, 512, 1024, 2048]), 5)
    fig = render_figure([heatmap], {"dataset": "test", "trials": "single run", "hop": 128},
                        _message([heatmap]), slides=True)
    try:
        _check_text(fig)
        assert len(fig._suptitle.get_text().splitlines()) == 2
        renderer = fig.canvas.get_renderer()
        assert fig._suptitle.get_window_extent(renderer).y0 > fig.axes[0]._left_title.get_window_extent(renderer).y1
    finally:
        plt.close(fig)


def test_slide_domains_preserve_measured_time_and_nyquist_limits():
    maps = power_panels({"posterior_blend": np.ones((1723, 33))}, 44100, 128)
    p = curve_panel("Blend weights", np.array([.002, 5.001]), {
        "fixed_0": np.array([1, .4]), "fixed_1": np.array([0, .6]),
    }, "Blend weight")
    for item in (maps[0], p):
        fig = render_figure([item], {"dataset": "test", "trials": "single run", "hop": 128},
                            _message([item]), slides=True)
        try:
            _check_text(fig)
            assert fig.axes[0].get_xlim() == tuple(item["xlim"])
            if item is maps[0]:
                assert fig.axes[0].get_ylim() == (0, 22050)
        finally:
            plt.close(fig)


def test_focus_crops_every_method_identically_and_keeps_a_shared_full_band_copy():
    maps = power_panels({"posterior_blend": np.ones((4, 2049)),
                        "fixed_0": np.ones((4, 2049)) * .1}, 48000, 128)
    provenance = {"filename": "humpback_whale_song.wav",
                  "sample_rate_hz": 48000}
    focused, complete, record = spectrogram_views(maps, provenance)
    assert record["focus_hz"] == [0, 4000]
    assert record["full_band_companion"]
    for crop, full, original in zip(focused, complete, maps):
        assert crop["ylim"] == [0, 4000]
        assert full["ylim"] == [0, 24000]
        assert crop["images"][0]["clim"] == full["images"][0]["clim"] == [-70, 0]
        np.testing.assert_array_equal(crop["images"][0]["values"], full["images"][0]["values"])
        assert crop["images"][0]["extent"] == original["images"][0]["extent"]
        assert crop["images"][0]["values"].shape == original["images"][0]["values"].shape
    provenance["filename"] = "snapping_shrimp.wav"
    focused, complete, record = spectrogram_views(maps, provenance)
    assert complete is None
    assert record["focus_hz"] == [0, 24000]


def test_focused_four_panel_labels_do_not_overlap():
    maps = power_panels({"posterior_blend": np.ones((4, 2049)),
                        "fixed_0": np.ones((4, 2049)) * .1,
                        "fixed_2": np.ones((4, 2049)) * .2,
                        "fixed_4": np.ones((4, 2049)) * .3}, 48000, 128)
    focused, _, _ = spectrogram_views(maps, {
        "filename": "humpback_whale_song.wav",
        "sample_rate_hz": 48000})
    fig = render_figure(focused, {"dataset": "humpback", "trials": "single run", "hop": 128},
                        _message(focused), slides=False)
    try:
        _check_text(fig)
        # The original long description collided with the neighbouring label.
        fig.axes[0].set_xlabel("Time (s); ranked by full-region joint H3\nsingle run")
        fig.axes[2].set_xlabel("Time (s); ranked by full-region joint H3\nsingle run")
        with pytest.raises(ValueError, match="Axis labels overlap"):
            _check_text(fig)
    finally:
        plt.close(fig)


def test_cached_refinement_keeps_four_figures_and_preserves_numeric_evidence(tmp_path, monkeypatch):
    dataset = "humpback8kmono"
    fs, hop, n_fft = 128, 2, 64
    x = np.sin(2 * np.pi * 9 * np.arange(256) / fs)
    x[::37] += 2
    case = paper._case_outputs(x, n_fft, hop, 1.5, "geometric")
    monkeypatch.setattr(wav_collection, "N_FFT", n_fft)
    monkeypatch.setattr(wav_collection, "HOP", hop)
    with figure_edition(tmp_path / "source", "Plotting fixture; no experiment evidence") as source:
        record = wav_collection.save_wav_figures(extract_case(case),
            {"filename": "humpback8kmono.wav", "sample_rate_hz": fs}, case["warmup"], source)
        (source.root / "data/wav_inventory.json").write_text(json.dumps([record]))
        assert len(source.entries) == 4
    result = refine_cached_collection(source.root, tmp_path / "refined")
    assert not list(result.rglob("*blend_weights_curves*"))
    for filename in ("entropy_regret.csv", "all_method_entropy_regret.csv", "blend_weights.csv"):
        relative = Path("data") / dataset / filename
        assert (result / relative).read_bytes() == (source.root / relative).read_bytes()
    for style in ("dissertation", "slides"):
        assert len(list((result / style).rglob("*.png"))) == 4
        assert len(list((result / style).rglob("*.pdf"))) == 4
        assert not (result / style / dataset / "methods").exists()
        assert (result / style / dataset / "04_blend_weights_pmf.png").is_file()
    assert "0 errors; 0 warnings" in (result / "build.log").read_text()
