"""Hindsight comparators for the joint spectrogram PMF, independent of tracking.

For a path, let A be accumulated full-DFT power and B accumulated cubed power.
Its joint H3 is (3 log2 A - log2 B) / 2. These two moments are additive;
entropy itself is not. Exact Pareto labels solve short prefixes. Larger
prefixes use mixed-integer outer approximation with an explicit entropy-gap
certificate. No greedy path or one-label-per-state entropy DP is substituted.
"""

from time import monotonic
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csc_matrix, vstack


def spectral_moments(spectra, n_fft):
    """Return (frame, state) moments on the paper's full-frequency grid."""
    spectra = np.asarray(spectra)
    if spectra.ndim != 3 or spectra.shape[-1] != n_fft // 2 + 1:
        raise ValueError("Expected (state, frame, rFFT bin) candidate spectra")
    power = np.abs(spectra) ** 2
    if not np.all(np.isfinite(power)):
        raise ValueError("Spectral power must be finite")
    scale = float(power.max(initial=0))
    if scale:
        power /= scale
    multiplicity = np.full(power.shape[-1], 2.)
    multiplicity[0] = 1
    if n_fft % 2 == 0:
        multiplicity[-1] = 1
    return ((power * multiplicity).sum(axis=-1).T,
            (power ** 3 * multiplicity).sum(axis=-1).T)


def entropy(a, b):
    a, b = np.asarray(a), np.asarray(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a > 0) & (b > 0),
                        .5 * (3 * np.log2(a) - np.log2(b)), np.nan)


def _pareto_prefixes(a, b, support, stops, label_limit):
    """Keep every nondominated (low A, high B) label at each ending state."""
    states = a.shape[1]
    fronts = [np.zeros((1, 2)) for _ in range(states)]
    history, results = [], {}
    for t in range(max(stops)):
        layer, parents = [], []
        for state in range(states):
            predecessors = np.flatnonzero(support[:, state])
            values = np.concatenate([fronts[p] for p in predecessors])
            back = np.concatenate([
                np.column_stack((np.full(len(fronts[p]), p), np.arange(len(fronts[p]))))
                for p in predecessors])
            values = values + [a[t, state], b[t, state]]
            order = np.lexsort((-values[:, 1], values[:, 0]))
            values, back = values[order], back[order]
            keep = values[:, 1] > np.r_[-np.inf, np.maximum.accumulate(values[:-1, 1])]
            layer.append(values[keep])
            parents.append(back[keep].astype(np.int32))
        if sum(map(len, layer)) > label_limit:
            break
        fronts = layer
        history.append(parents)
        if t + 1 not in stops:
            continue
        choices = []
        for state, values in enumerate(fronts):
            h = entropy(values[:, 0], values[:, 1])
            if np.any(np.isfinite(h)):
                index = int(np.nanargmin(h))
                choices.append((float(h[index]), state, index))
        if not choices:
            results[t + 1] = {"entropy_bits": np.nan, "gap_bits": 0.,
                              "path": None, "solver": "exact Pareto labels"}
            continue
        h, state, index = min(choices)
        path = np.empty(t + 1, dtype=np.int32)
        for frame in range(t, -1, -1):
            path[frame] = state
            state, index = history[frame][state][index]
        results[t + 1] = {"entropy_bits": h, "gap_bits": 0., "path": path,
                          "solver": "exact Pareto labels"}
    return results


def _linear_paths(a, b, support, slopes, trace=False):
    """Maximize B - slope*A; these are feasible seeds and supporting bounds."""
    count, states = len(slopes), a.shape[1]
    pa = np.zeros((count, states))
    pb = np.zeros_like(pa)
    backs = []
    rows = np.arange(count)
    for t in range(len(a)):
        scores = pb - slopes[:, None] * pa
        previous = np.empty((count, states), dtype=np.int32)
        for s in range(states):
            pred = np.flatnonzero(support[:, s])
            previous[:, s] = pred[np.argmax(scores[:, pred], axis=1)]
        pa = np.take_along_axis(pa, previous, axis=1) + a[t]
        pb = np.take_along_axis(pb, previous, axis=1) + b[t]
        if trace:
            backs.append(previous)
    if not trace:
        return pa, pb
    end = np.argmax(pb - slopes[:, None] * pa, axis=1)
    paths = np.empty((count, len(a)), dtype=np.int32)
    for t in range(len(a) - 1, -1, -1):
        paths[:, t] = end
        end = backs[t][rows, end]
    return pa, pb, paths


def _energy_interval(slopes, intercepts, maximum, incumbent_a):
    """Intersect necessary energy ranges for beating the feasible seed.

Each supporting line gives B <= slope*A + intercept. A better path must
therefore satisfy A**3 <= slope*A + intercept in seed-normalized units.
"""
    peak = np.clip(np.sqrt(slopes / 3), 1., maximum)
    slack = 2e-10 * np.maximum.reduce((np.ones_like(peak),
                                     np.abs(intercepts), slopes * peak, peak ** 3))
    left, right = np.ones_like(peak), peak.copy()
    for _ in range(60):
        middle = (left + right) / 2
        possible = slopes * middle + intercepts - middle ** 3 >= -slack
        right = np.where(possible, middle, right)
        left = np.where(possible, left, middle)
    low = max(1., float(left.max()) - 1e-8 * max(1., incumbent_a))
    left, right = peak.copy(), np.full_like(peak, maximum)
    for _ in range(60):
        middle = (left + right) / 2
        possible = slopes * middle + intercepts - middle ** 3 >= -slack
        left = np.where(possible, middle, left)
        right = np.where(possible, right, middle)
    high = min(maximum, float(right.min()) + 1e-8 * max(1., incumbent_a))
    if not low - 1e-7 <= incumbent_a <= high + 1e-7:
        raise ArithmeticError("Supporting bounds excluded their feasible seed")
    return min(low, incumbent_a), max(high, incumbent_a)


def _milp_prefix(a, b, support, tolerance_bits, time_limit_s):
    """Certify max B/A**3 using linear outer bounds on B-r*A**3."""
    start = monotonic()
    frames, states = a.shape
    low = np.zeros(states)
    high = np.zeros(states)
    for t in range(frames):
        low = np.array([low[support[:, s]].min() for s in range(states)]) + a[t]
        high = np.array([high[support[:, s]].max() for s in range(states)]) + a[t]
    positives = a[a > 0]
    if not len(positives):
        return {"entropy_bits": np.nan, "gap_bits": 0., "path": None,
                "solver": "zero-energy prefix"}
    scale_a = max(float(low.min()), float(positives.min()))
    maximum = float(high.max()) / scale_a
    ratios = np.divide(b, a, out=np.zeros_like(b), where=a > 0)
    positive = ratios[ratios > 0]
    slopes = np.r_[0., np.geomspace(max(positive.min() * .01, positive.max() * 1e-8),
                                   positive.max() * 3.1, 256)]
    pa, pb = _linear_paths(a, b, support, slopes)
    h = entropy(pa, pb)
    # A seed must end in a state maximizing its linear objective, so its
    # backtrace is exactly reproducible. Include every fixed path separately.
    end = np.argmax(pb - slopes[:, None] * pa, axis=1)
    line_h = h[np.arange(len(slopes)), end]
    best_line = int(np.nanargmin(line_h))
    _, _, traces = _linear_paths(a, b, support, slopes[best_line:best_line + 1], trace=True)
    path = traces[0]
    fixed_h = entropy(a.sum(axis=0), b.sum(axis=0))
    eligible = np.diag(support) & np.isfinite(fixed_h)
    if np.any(eligible):
        fixed = int(np.argmin(np.where(eligible, fixed_h, np.inf)))
        if fixed_h[fixed] < line_h[best_line]:
            path = np.full(frames, fixed, dtype=np.int32)
    row = np.arange(frames)
    raw_a, raw_b = a[row, path].sum(), b[row, path].sum()
    seed_ratio = raw_b / raw_a ** 3
    scale_b = seed_ratio * scale_a ** 3
    aa, bb = a.ravel() / scale_a, b.ravel() / scale_b
    incumbent_a = raw_a / scale_a
    intercepts = (pb - slopes[:, None] * pa).max(axis=1) / scale_b
    low_a, high_a = _energy_interval(slopes * scale_a / scale_b, intercepts,
                                    maximum, incumbent_a)

    # Often the supporting lines already isolate a globally optimal path.
    # Certify it directly before constructing an unnecessary integer model.
    scaled_slopes = slopes * scale_a / scale_b
    stationary = np.full_like(slopes, low_a)
    np.divide(-3 * intercepts, 2 * scaled_slopes, out=stationary,
              where=scaled_slopes > 0)
    stationary = np.clip(stationary, low_a, high_a)
    bound = np.maximum.reduce((
        (scaled_slopes * low_a + intercepts) / low_a ** 3,
        (scaled_slopes * high_a + intercepts) / high_a ** 3,
        (scaled_slopes * stationary + intercepts) / stationary ** 3))
    supporting_gap = .5 * np.log2(max(1., float(bound.min())) * (1 + 1e-8))
    if supporting_gap <= tolerance_bits:
        return {"entropy_bits": float(entropy(raw_a, raw_b)),
                "gap_bits": float(supporting_gap), "path": path,
                "solver": "global supporting-line certificate", "iterations": 0,
                "elapsed_s": monotonic() - start}

    variables = frames * states
    rr, cc, vv, lb, ub = [], [], [], [], []
    for t in range(frames):
        r = len(lb)
        rr.extend([r] * states)
        cc.extend(range(t * states, (t + 1) * states))
        vv.extend([1.] * states)
        lb.append(1.); ub.append(1.)
        if t:
            # Destination-set flow inequalities prevent the relaxation from
            # reusing one predecessor's fractional mass in adjacent states.
            # Singletons alone suffice for integral paths but give a weaker
            # relaxation. Every feasible discrete schedule satisfies these.
            for first in range(states):
                for last in range(first + 1, states + 1):
                    destinations = np.arange(first, last)
                    pred = np.flatnonzero(np.any(support[:, destinations], axis=1))
                    if len(pred) == states:
                        continue
                    r = len(lb)
                    rr.extend([r] * (len(destinations) + len(pred)))
                    cc.extend(t * states + destinations)
                    cc.extend((t - 1) * states + pred)
                    vv.extend([1.] * len(destinations))
                    vv.extend([-1.] * len(pred))
                    lb.append(-np.inf); ub.append(0.)
    base = coo_matrix((vv, (rr, cc)), shape=(len(lb), variables + 1)).tocsc()
    base = vstack((base, csc_matrix(np.r_[aa, 0.][None, :])), format="csc")
    lb.append(low_a); ub.append(high_a)
    bounds = Bounds(np.r_[np.zeros(variables), -np.inf],
                    np.r_[np.ones(variables), np.inf])
    objective = np.r_[np.zeros(variables), -1.]
    integrality = np.r_[np.ones(variables, dtype=np.uint8), 0]
    knots = [*np.linspace(low_a, high_a, 24), incumbent_a]
    ratio, upper_ratio = 1., np.inf
    for iteration in range(60):
        remaining = time_limit_s - (monotonic() - start)
        if remaining <= 0:
            break
        cuts = np.array([np.r_[3 * ratio * point ** 2 * aa - bb, 1.]
                         for point in knots])
        constraint = LinearConstraint(vstack((base, csc_matrix(cuts)), format="csc"),
            np.r_[lb, np.full(len(knots), -np.inf)],
            np.r_[ub, [2 * ratio * point ** 3 for point in knots]])
        # SciPy forwards this supported HiGHS option but warns because it is
        # not in SciPy's small option whitelist. Bound solver CPU use explicitly.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads.*",
                                    category=RuntimeWarning)
            answer = milp(objective, integrality=integrality, bounds=bounds,
                constraints=constraint, options={
                    "time_limit": min(remaining, 10. * (iteration + 1)),
                    "mip_rel_gap": 1e-4,
                    "mip_abs_gap": .2 * ratio * low_a ** 3 * np.expm1(2 * np.log(2) * tolerance_bits),
                    "mip_feasibility_tolerance": 1e-9,
                    "primal_feasibility_tolerance": 1e-9,
                    "dual_feasibility_tolerance": 1e-9,
                    "threads": 1})
        if answer.x is None or answer.mip_dual_bound is None:
            if answer.status == 1:
                continue
            raise RuntimeError(f"Joint entropy optimization failed: {answer.message}")
        candidate = np.argmax(answer.x[:-1].reshape(frames, states), axis=1)
        if np.any(~support[candidate[:-1], candidate[1:]]):
            raise ArithmeticError("Solver returned an inadmissible path")
        av = aa.reshape(frames, states)[row, candidate].sum()
        bv = bb.reshape(frames, states)[row, candidate].sum()
        # Every tangent underestimates A**3. The MILP dual bound therefore
        # bounds max(B-r*A**3) above; A >= low_a converts it to a ratio bound.
        dual = max(0., -float(answer.mip_dual_bound))
        dual += 1e-8 * max(1., ratio, dual)
        upper_ratio = min(upper_ratio, ratio + dual / low_a ** 3)
        candidate_ratio = bv / av ** 3
        if candidate_ratio > ratio:
            ratio, path = candidate_ratio, candidate
        upper_ratio = max(upper_ratio, ratio)
        gap = .5 * np.log2(upper_ratio / ratio)
        if gap <= tolerance_bits:
            return {"entropy_bits": float(entropy(a[row, path].sum(), b[row, path].sum())),
                    "gap_bits": float(gap), "path": path,
                    "solver": "certified integer outer approximation",
                    "iterations": iteration + 1, "elapsed_s": monotonic() - start}
        # Improve the fractional objective before spending time proving the
        # old auxiliary problem to unnecessarily high relative precision.
        low_a, high_a = _energy_interval(scaled_slopes / ratio, intercepts / ratio,
                                        maximum, float(aa.reshape(frames, states)[row, path].sum()))
        lb[-1], ub[-1] = low_a, high_a
        knots = [point for point in knots if low_a <= point <= high_a]
        knots.extend([low_a, high_a, float(av)])
    raise RuntimeError(f"Joint entropy optimum was not certified within {time_limit_s:g} s; "
                       f"remaining entropy bound {0.5 * np.log2(upper_ratio / ratio):g} bits")


def optimize_prefixes(a, b, transition, stops=None, *, tolerance_bits=1e-6,
                      label_limit=100_000, time_limit_s=120., progress=None):
    """Optimize each requested common-region prefix, with all start states allowed.

    All candidate windows are already available at the common post-warm-up
    start. A separate path is returned for each stop (exclusive frame count).
    Transition probabilities determine feasibility, not a path penalty.
    An uncertified solve raises; it is never silently exported as hindsight.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    support = np.asarray(transition) > 0
    if (a.ndim != 2 or a.shape != b.shape or not len(a) or
            support.shape != (a.shape[1], a.shape[1]) or
            not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or
            np.any(a < 0) or np.any(b < 0) or np.any((a == 0) != (b == 0)) or
            not np.all(support.any(axis=0))):
        raise ValueError("Invalid moments or transition support")
    stops = sorted(set(range(1, len(a) + 1) if stops is None else map(int, stops)))
    if not stops or stops[0] < 1 or stops[-1] > len(a) or tolerance_bits <= 0:
        raise ValueError("Invalid prefixes or tolerance")
    results = _pareto_prefixes(a, b, support, set(stops), label_limit)
    for stop in stops:
        if stop not in results:
            try:
                results[stop] = _milp_prefix(a[:stop], b[:stop], support,
                                             tolerance_bits, time_limit_s)
            except RuntimeError as error:
                raise RuntimeError(f"Prefix {stop}: {error}") from error
        result = results[stop]
        result["stop"] = stop
        if progress:
            progress(result)
    return [results[stop] for stop in stops]
