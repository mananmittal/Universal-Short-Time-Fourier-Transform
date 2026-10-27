"""Shared chapter styles and perceptual colour checks (sRGB, D65).

Colour difference: Sharma, Wu & Dalal (2005),
https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/.
CVD simulation: Machado, Oliveira & Fernandes (2009), severity 1.0,
https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html.
The matrices operate on linear RGB, before clipping to the display gamut.
"""

from itertools import combinations
from types import MappingProxyType
import re

import numpy as np
from matplotlib.colors import to_rgb


REFERENCE_GREY = "#4D4D4D"
LINE_WIDTH = 1.8
SLIDE_SIZE = (13.33, 7.5)
SLIDE_PIXELS = (2666, 1500)
MACHADO = {
    "protanopia": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deuteranopia": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
}


def _linear_rgb(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92,
                    ((rgb + 0.055) / 1.055) ** 2.4)


def colour_lab(colour, vision=None):
    """CIELAB from display sRGB, optionally after full Machado dichromacy."""
    rgb = _linear_rgb(to_rgb(colour))
    if vision:
        rgb = np.clip(MACHADO[vision] @ rgb, 0.0, 1.0)
    xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]) @ rgb / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz),
                 xyz / (3 * delta ** 2) + 4.0 / 29.0)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]),
                     200 * (f[1] - f[2])])


def ciede2000(lab1, lab2):
    """CIEDE2000 with unit parametric factors, including zero-chroma cases."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.degrees(np.arctan2(b1, ap1)) % 360
    hp2 = np.degrees(np.arctan2(b2, ap2)) % 360
    dl, dc = l2 - l1, cp2 - cp1
    dh = hp2 - hp1
    if cp1 * cp2 == 0:
        dh = 0.0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    d_h = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh / 2))
    lp, cp = (l1 + l2) / 2, (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        hp = hp1 + hp2
    elif abs(hp1 - hp2) <= 180:
        hp = (hp1 + hp2) / 2
    elif hp1 + hp2 < 360:
        hp = (hp1 + hp2 + 360) / 2
    else:
        hp = (hp1 + hp2 - 360) / 2
    t = (1 - 0.17 * np.cos(np.radians(hp - 30))
         + 0.24 * np.cos(np.radians(2 * hp))
         + 0.32 * np.cos(np.radians(3 * hp + 6))
         - 0.20 * np.cos(np.radians(4 * hp - 63)))
    sl = 1 + 0.015 * (lp - 50) ** 2 / np.sqrt(20 + (lp - 50) ** 2)
    sc, sh = 1 + 0.045 * cp, 1 + 0.015 * cp * t
    rt = (-2 * np.sqrt(cp ** 7 / (cp ** 7 + 25 ** 7))
          * np.sin(np.radians(60 * np.exp(-((hp - 275) / 25) ** 2))))
    return float(np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2
                         + (d_h / sh) ** 2 + rt * (dc / sc) * (d_h / sh)))


# Saturated named-palette candidates, rather than interpolated pastel colours.
# Kelly: ISCC-NBS maximum-contrast colours; Tol: bright/high-contrast schemes.
NAMED_PALETTES = MappingProxyType({
    "tab10": ("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
              "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"),
    "Set1": ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
             "#FFFF33", "#A65628", "#F781BF", "#999999"),
    "Dark2": ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
              "#E6AB02", "#A6761D", "#666666"),
    "Kelly": ("#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032",
              "#C2B280", "#848482", "#008856", "#E68FAC", "#0067A5",
              "#F99379", "#604E97", "#F6A600", "#B3446C", "#DCD300",
              "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26"),
    "Okabe–Ito": ("#E69F00", "#56B4E9", "#009E73", "#F0E442",
                  "#0072B2", "#D55E00", "#CC79A7"),
    "Tol": ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
            "#AA3377", "#BBBBBB", "#004488", "#DDAA33", "#BB5566"),
})


def palette_distances(colours):
    """Return the three pairwise distance matrices used by tests and search."""
    labs = [colour_lab(c) for c in colours]
    distances = np.array([[ciede2000(a, b) for b in labs] for a in labs])
    cvd = []
    for vision in MACHADO:
        simulated = np.array([colour_lab(c, vision) for c in colours])
        cvd.append(np.linalg.norm(simulated[:, None] - simulated[None, :], axis=2))
    return distances, *cvd


def search_named_palette(count, pinned=()):
    """Exact max-min search over feasible cliques in the named colour pool.

    L* and distance-to-black restrictions prune vertices; Machado distances
    prune edges. A threshold clique search maximises the normal-vision minimum.
    Pinned chapter colours are included without altering them.
    """
    black = colour_lab("#000000")
    pool = list(dict.fromkeys([*pinned, *(
        colour for palette in NAMED_PALETTES.values() for colour in palette
    )]))
    pool = [c for c in pool if colour_lab(c)[0] <= 75
            and ciede2000(colour_lab(c), black) >= 25]
    normal, protan, deutan = palette_distances(pool)
    pinned_indices = [pool.index(c) for c in pinned]

    def clique(threshold):
        edges = (normal >= threshold) & (protan >= 15) & (deutan >= 15)
        if any(not edges[i, j] for i, j in combinations(pinned_indices, 2)):
            return None
        choices = [i for i in range(len(pool)) if i not in pinned_indices
                   and all(edges[i, j] for j in pinned_indices)]

        def visit(selected, candidates):
            if len(selected) == count:
                return selected
            while len(selected) + len(candidates) >= count:
                vertex, *candidates = candidates
                found = visit(selected + [vertex],
                              [i for i in candidates if edges[vertex, i]])
                if found:
                    return found
            return None
        return visit(pinned_indices.copy(), choices)

    thresholds = sorted(set(normal[np.triu_indices(len(pool), 1)]))
    low, high, best = 0, len(thresholds) - 1, None
    while low <= high:
        middle = (low + high) // 2
        found = clique(thresholds[middle])
        if found is None:
            high = middle - 1
        else:
            best = found
            low = middle + 1
    if best is None or thresholds[max(high, 0)] < 28:
        raise ValueError(f"No {count}-colour named palette meets the chapter rules")
    return tuple(pool[i] for i in best)


# This seven-colour solution is the max-min named-palette search with the
# inherited best-fixed colour pinned. Candidate banks and the four-method
# comparisons are separate diagnostics, so their actual co-occurrence sets
# (not unrelated methods from other experiments) are tested below.
METHOD_ORDER = ("constrained_schedule", "best_fixed_support", "posterior_blend",
                "framewise_selection", "fixed_0", "fixed_1", "fixed_2",
                "fixed_3", "fixed_4", "dyadic_bank", "linear_bank")
PALETTE = MappingProxyType({
    "constrained_schedule": "#000000",
    "best_fixed_support": "#F99379",
    "posterior_blend": "#004488",
    "framewise_selection": "#2CA02C",
    "fixed_0": "#BCBD22",
    "fixed_1": "#17BECF",
    "fixed_2": "#E7298A",
    "fixed_3": "#654522",
    "fixed_4": "#228833",
    "dyadic_bank": "#0072B2",
    "linear_bank": "#D55E00",
})
LABELS = MappingProxyType({
    "constrained_schedule": "Hindsight schedule",
    "best_fixed_support": "Best fixed window",
    "posterior_blend": "U-STFT",
    "framewise_selection": "Framewise selection",
    "fixed_0": "Fixed H",
    "fixed_1": "Fixed 2H",
    "fixed_2": "Fixed 4H",
    "fixed_3": "Fixed 8H",
    "fixed_4": "Fixed 16H",
    "dyadic_bank": "Dyadic bank",
    "linear_bank": "Linear bank",
})
ROLES = MappingProxyType({key: (
    "ORACLE" if key == "constrained_schedule" else
    "BASELINE" if key == "fixed_0" else "METHOD"
) for key in METHOD_ORDER})
LINESTYLES = MappingProxyType({
    "constrained_schedule": "-",
    "best_fixed_support": (0, (7, 2, 2, 2)),
    "posterior_blend": (0, (7, 2)),
    "framewise_selection": (0, (4, 1.6)),
    "fixed_0": "-",
    "fixed_1": (0, (1.4, 1.4)),
    "fixed_2": "-.",
    "fixed_3": (0, (5, 1.3, 5, 3.5)),
    "fixed_4": (0, (12, 3)),
    "dyadic_bank": (0, (6, 2, 1.5, 2)),
    "linear_bank": (0, (3, 2, 1, 2, 1, 2)),
})
MARKERS = MappingProxyType(dict(zip(METHOD_ORDER,
    ("o", "s", "D", "^", "v", "P", "X", "h", "<", ">", "p"))))
# A light probability map keeps the oracle and method overlays visible.
PROBABILITY_COLOURS = ("#FFFFFF", "#B8B8B8")
PROBABILITY_MAP_COLOURS = ("#FFFFFF", PALETTE["posterior_blend"])
COOCCURRENCE = (
    ("constrained_schedule", "best_fixed_support", "posterior_blend",
     "framewise_selection"),
    ("posterior_blend", "fixed_0", "fixed_1", "fixed_2", "fixed_3", "fixed_4"),
    ("dyadic_bank", "linear_bank"),
)


def check_palette(methods):
    """Fail rather than silently relax a requested perceptual threshold."""
    methods = tuple(dict.fromkeys(m for m in methods if m in PALETTE))
    coloured = [m for m in methods if ROLES[m] != "ORACLE"]
    for method in coloured:
        lab = colour_lab(PALETTE[method])
        if lab[0] > 75 or ciede2000(lab, colour_lab("#000000")) < 25:
            raise ValueError(f"{method} is too pale or too close to black")
    if len(methods) > 1:
        normal, protan, deutan = palette_distances([PALETTE[m] for m in methods])
        pairs = np.triu_indices(len(methods), 1)
        for metric, threshold, matrix in (
                ("CIEDE2000", 28, normal), ("protanopia Lab", 15, protan),
                ("deuteranopia Lab", 15, deutan)):
            if np.min(matrix[pairs]) < threshold - 1e-9:
                raise ValueError(f"{metric} threshold failed for {methods}")


def method_from_label(label):
    """Translate historical plot labels into the single chapter label map."""
    label = str(label)
    if label in PALETTE:
        return label
    for key, name in LABELS.items():
        if label == name:
            return key
    lower = label.lower()
    # Bounds are diagnostic references, never method colours or oracle black.
    if any(s in lower for s in ("bound", "certificate", "ceiling", "floor",
                               "descriptive", "regret", "mixture $-$")):
        return None
    if any(s in lower for s in ("hindsight schedule", "constrained", "best window schedule",
                               "prefix-optimal", "noncausal", "comparator")):
        return "constrained_schedule"
    if any(s in lower for s in ("framewise", "greedy")):
        return "framewise_selection"
    if any(s in lower for s in ("u-stft", "universal", "posterior", "mixture log loss")):
        return "posterior_blend"
    if any(s in lower for s in ("best fixed", "fixed causal", "fixed stft", "hindsight support")):
        return "best_fixed_support"
    if lower == "dyadic bank":
        return "dyadic_bank"
    if lower == "linear bank":
        return "linear_bank"
    return None


def line_style(method, slides=False, offset=0, count=1, points=100):
    """Equal-width curves; dissertation markers have staggered white rims."""
    result = {"color": PALETTE[method], "linewidth": LINE_WIDTH,
              "linestyle": LINESTYLES[method], "label": LABELS[method],
              "alpha": 1.0, "zorder": 2 + METHOD_ORDER.index(method) * 0.01}
    if not slides:
        every = max(1, int(np.ceil(points / 9)))
        result.update(marker=MARKERS[method], markersize=5,
                      markerfacecolor=PALETTE[method], markeredgecolor="white",
                      markeredgewidth=0.8,
                      markevery=(int(offset * every / max(count, 1)), every))
    else:
        result.update(marker="None")
    return result
