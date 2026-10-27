"""Compare the joint-entropy oracle with exhaustive feasible schedules."""

import itertools

import numpy as np
import pytest

from benchmarks.joint_entropy import entropy, optimize_prefixes, spectral_moments
from benchmarks.joint_entropy_collection import prefix_stops, replace_joint_spec, JOINT_OBJECTIVE


def support(states):
    result = np.eye(states, dtype=bool)
    result[:, 0] = True
    for s in range(states - 1):
        result[s, s + 1] = True
    return result


def exhaustive(a, b, transitions):
    best = np.inf
    for path in itertools.product(range(a.shape[1]), repeat=len(a)):
        if any(not transitions[i, j] for i, j in zip(path, path[1:])):
            continue
        value = entropy(a[np.arange(len(a)), path].sum(),
                        b[np.arange(len(b)), path].sum())
        if np.isfinite(value):
            best = min(best, float(value))
    return best


@pytest.mark.parametrize("n_fft", [5, 6])
def test_moments_match_the_full_dft_grid_and_ignore_global_gain(n_fft):
    rng = np.random.default_rng(52)
    real = rng.normal(size=(3, 6, n_fft))
    spectra = np.fft.rfft(real, axis=-1)
    a, b = spectral_moments(spectra, n_fft)
    power = np.abs(np.fft.fft(real, axis=-1)) ** 2
    for state in range(3):
        for stop in range(1, 7):
            pmf = power[state, :stop] / power[state, :stop].sum()
            expected = -.5 * np.log2(np.sum(pmf ** 3))
            assert entropy(a[:stop, state].sum(), b[:stop, state].sum()) == pytest.approx(expected)
    scaled = spectral_moments(spectra * 19, n_fft)
    np.testing.assert_allclose(entropy(*scaled), entropy(a, b), atol=1e-13)


@pytest.mark.parametrize("label_limit", [100_000, 0])
@pytest.mark.parametrize("seed", [1, 5, 11])
def test_every_prefix_matches_exhaustive_schedules(seed, label_limit):
    rng = np.random.default_rng(seed)
    spectra = rng.normal(size=(3, 6, 5)) + 1j * rng.normal(size=(3, 6, 5))
    a, b = spectral_moments(spectra, 8)
    transitions = support(3)
    results = optimize_prefixes(a, b, transitions, label_limit=label_limit,
                                tolerance_bits=1e-7, time_limit_s=15)
    for stop, result in enumerate(results, 1):
        expected = exhaustive(a[:stop], b[:stop], transitions)
        assert expected - 2e-10 <= result["entropy_bits"] <= expected + 1e-7
        assert 0 <= result["gap_bits"] <= 1e-7
        assert result["entropy_bits"] - result["gap_bits"] <= expected + 2e-9
        path = result["path"]
        assert len(path) == stop
        assert np.all(transitions[path[:-1], path[1:]])
        actual = entropy(a[np.arange(stop), path].sum(), b[np.arange(stop), path].sum())
        assert actual == pytest.approx(result["entropy_bits"], abs=1e-13)
        fixed = entropy(a[:stop].sum(axis=0), b[:stop].sum(axis=0))
        assert np.min(fixed - result["entropy_bits"]) >= -2e-12


def test_entropy_optimum_can_require_a_nondominant_partial_path():
    # The smaller energy history has worse current entropy, but wins after
    # both paths receive the same later energy. One entropy label loses it.
    a = np.array([[1., 2.], [1., 1.]])
    b = np.array([[.1, 1.], [1., 1.]])
    assert entropy(a[0, 1], b[0, 1]) < entropy(a[0, 0], b[0, 0])
    result = optimize_prefixes(a, b, np.ones((2, 2)), stops=[2])[0]
    assert result["path"][0] == 0
    assert result["entropy_bits"] == pytest.approx(entropy(2., 1.1))


@pytest.mark.parametrize("label_limit", [100_000, 0])
def test_leading_silence_and_sparse_prefix_requests(label_limit):
    a = np.array([[0., 0.], [0., 0.], [1., 2.], [2., 1.]])
    b = np.array([[0., 0.], [0., 0.], [.2, 1.], [1., .2]])
    results = optimize_prefixes(a, b, support(2), stops=[1, 2, 4], label_limit=label_limit)
    assert [r["stop"] for r in results] == [1, 2, 4]
    assert np.isnan(results[0]["entropy_bits"])
    assert np.isnan(results[1]["entropy_bits"])
    assert results[2]["entropy_bits"] == pytest.approx(exhaustive(a, b, support(2)))


def test_invalid_moments_and_transitions_are_rejected():
    with pytest.raises(ValueError, match="moments"):
        optimize_prefixes(np.ones((3, 2)), -np.ones((3, 2)), support(2))
    with pytest.raises(ValueError, match="prefixes"):
        optimize_prefixes(np.ones((3, 2)), np.ones((3, 2)), support(2), stops=[0])


def test_plot_sampling_includes_the_complete_record_and_small_fixtures():
    stops = prefix_stops(1859)
    assert len(stops) == 65
    assert stops[0] == 1 and stops[-1] == 1859
    assert np.all(np.diff(stops) > 0)
    np.testing.assert_array_equal(prefix_stops(4), [1, 2, 3, 4])


def test_joint_replacement_changes_only_the_selected_metric_arrays():
    x = np.arange(1, 11, dtype=float)
    curves = {"posterior_blend": x * .1, "fixed_0": x * .2}
    spec = {"metadata": {"dataset": "fixture"}, "panels": [{"title": "old", "lines": [
        {"method": key, "x": x.copy(), "y": values.copy()}
        for key, values in curves.items()]}]}
    results = [{"stop": stop, "entropy_bits": stop * .05, "gap_bits": 1e-9}
               for stop in (1, 4, 10)]
    updated, record = replace_joint_spec(spec, {"dataset": "fixture"}, curves, x, results)
    assert updated["metadata"]["hindsight_objective"] == JOINT_OBJECTIVE
    assert record["joint_entropy_oracle"]["plotted_prefixes"] == 3
    for line in updated["panels"][0]["lines"]:
        np.testing.assert_array_equal(line["x"], [1, 4, 10])
        np.testing.assert_allclose(line["y"], curves[line["method"]][[0, 3, 9]] - np.array([.05, .2, .5]))
    assert len(spec["panels"][0]["lines"][0]["x"]) == 10


def test_compact_replay_keeps_the_1d_and_2d_objectives_separate():
    from benchmarks.plot_suites import curve_panel, power_panels, posterior_panel
    from benchmarks.wav_collection import compact_specs
    x = np.arange(1., 11.)
    methods = {"posterior_blend": x, **{f"fixed_{i}": x + i for i in range(5)}}
    tracking = {"dataset": "fixture", "hindsight_objective": "original 1-D objective"}
    joint = {**tracking, "hindsight_objective": JOINT_OBJECTIVE}
    specs = {
        "01_spectrograms": {"metadata": tracking, "panels": power_panels(
            {key: np.ones((10, 17)) for key in methods}, 32, 2)},
        "02_regret_time_frequency_total": {"metadata": tracking, "panels": [
            curve_panel(name, x, methods, name) for name in ("time", "frequency", "total")]},
        "03_regret_2d_renyi3": {"metadata": joint, "panels": [curve_panel("joint", x, methods, "H3")]},
        "04_blend_weights_pmf": {"metadata": tracking, "panels": [
            posterior_panel(np.ones((10, 5)) / 5, np.array([2, 4, 8, 16, 32]), 1)]},
    }
    record = {"candidate_lengths_samples": [2, 4, 8, 16, 32],
              "selected_lengths_samples": {"best": 32, "middle": 8, "worst": 2},
              "joint_entropy_oracle": {"objective": JOINT_OBJECTIVE}}
    result = compact_specs(specs, record)
    assert result["02_regret_time_frequency_total"]["metadata"]["hindsight_objective"] == "original 1-D objective"
    assert result["03_regret_2d_renyi3"]["metadata"]["hindsight_objective"] == JOINT_OBJECTIVE
    assert "at each prefix" in result["03_regret_2d_renyi3"]["panels"][0]["xlabel"]


def test_parallel_prefix_checkpoints_match_exhaustive_and_resume(tmp_path):
    import json
    from benchmarks.joint_entropy_collection import solve_records
    rng = np.random.default_rng(9)
    a, b = spectral_moments(rng.normal(size=(2, 5, 5)), 8)
    folder = tmp_path / "fixture"
    folder.mkdir()
    np.savez_compressed(folder / "moments.npz", a=a, b=b, transition=support(2))
    solve_records([{"dataset": "fixture"}], tmp_path, 65, 2, 1e-6, 15)
    checkpoint = folder / "prefix_optima.json"
    before = checkpoint.read_bytes()
    results = json.loads(before)["results"]
    assert set(results) == {str(i) for i in range(1, 6)}
    for stop in range(1, 6):
        assert results[str(stop)]["entropy_bits"] == pytest.approx(
            exhaustive(a[:stop], b[:stop], support(2)), abs=1e-6)
    solve_records([{"dataset": "fixture"}], tmp_path, 65, 2, 1e-6, 15)
    assert checkpoint.read_bytes() == before
