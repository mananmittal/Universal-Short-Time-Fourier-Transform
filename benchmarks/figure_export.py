"""Plot-only chapter exporter: immutable editions and data-only figure replay.

Historical plotting functions may stage Matplotlib artists, but their colours,
widths, fonts, annotations and legends are deliberately not exported. Only
their plotted arrays, coordinates and metric descriptions enter this renderer.
Every delivered figure is drawn with :mod:`benchmarks.plot_style`.
"""

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
import hashlib
import inspect
import json
import math
import re
import textwrap
import warnings
from dateutil.tz import gettz

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import QuadMesh, PathCollection, PolyCollection
from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from PIL import Image
import numpy as np

from benchmarks.plot_style import (
    PALETTE, LABELS, ROLES, METHOD_ORDER, REFERENCE_GREY, LINE_WIDTH,
    LINESTYLES, SLIDE_SIZE, SLIDE_PIXELS, PROBABILITY_COLOURS, PROBABILITY_MAP_COLOURS, check_palette, line_style,
    method_from_label,
)


ROOT = Path(__file__).resolve().parents[1]
_ACTIVE = None
_DEFAULTS = {}
_RESOLVED = {}


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "figure"


def _array(value):
    return np.ma.asarray(value, dtype=float).filled(np.nan)


def _method(label, hop=None):
    if "regret" in str(label).lower() and "bound" not in str(label).lower():
        return "posterior_blend"
    key = method_from_label(label)
    if key:
        return key
    match = re.search(r"(?:L|length)\s*[=:]\s*(\d+)", str(label))
    if match and hop:
        ratio = int(match.group(1)) / float(hop)
        index = int(round(np.log2(ratio))) if ratio > 0 else -1
        if 0 <= index <= 4 and np.isclose(ratio, 2 ** index):
            return f"fixed_{index}"
    lower = str(label).lower()
    if "short" in lower and "fixed" in lower:
        return "fixed_0"
    if "long" in lower and "fixed" in lower:
        return "fixed_4"
    return None


def _curve_label(label, key):
    lower = str(label).lower()
    if "bound" in lower:
        return "Path-code bound"
    if "certificate" in lower:
        return "Loss certificate"
    if "regret" in lower:
        return "U-STFT regret"
    if "descriptive" in lower or "mixture $-$" in lower:
        return "Mixture − framewise" if "framewise" in lower else "Mixture − fixed"
    return LABELS[key] if key else str(label)


def _context(path):
    """Provenance from the plotting caller, without evaluating new experiments."""
    frame = inspect.currentframe().f_back
    context = {}
    for _ in range(6):
        if frame is None:
            break
        if frame.f_code.co_name.startswith("_plot") or frame.f_code.co_name == "plot_suite":
            context = frame.f_locals.copy()
            break
        frame = frame.f_back
    record = context.get("record", {})
    case = context.get("case", context.get("clean_case", {}))
    lengths = context.get("lengths", case.get("lengths", ()))
    hop = context.get("hop_size", record.get("hop_size"))
    if hop is None and len(lengths):
        hop = int(np.min(lengths))
    name = Path(path).stem
    dataset = record.get("slug") or context.get("title_prefix") or context.get("title")
    for token in ("hyperbolic-chirp", "humpback-whale", "snapping-shrimp", "tone-impulse-conflict"):
        if token in name:
            dataset = token
    if not dataset:
        if any(t in name for t in ("humpback", "whale")):
            dataset = "humpback-whale"
        elif "shrimp" in name:
            dataset = "snapping-shrimp"
        elif "chirp" in name:
            dataset = "hyperbolic-chirp"
        elif any(t in name for t in ("conflict", "controlled", "heisenberg", "transition", "ground_truth", "spectrogram")):
            dataset = "tone-impulse-conflict"
        else:
            dataset = "audit"
    trials = "single run"
    if any(s in name for s in ("snr", "dyadic_ablation", "concentration")):
        summary = context.get("snr_summary", ())
        runs = {int(row["runs"]) for row in summary if "runs" in row}
        if len(runs) == 1:
            trials = f"{runs.pop()} trials per SNR"
        elif context.get("rows"):
            seeds = {row["seed"] for row in context["rows"] if "seed" in row}
            trials = f"{len(seeds)} trials per SNR"
    return {"dataset": slug(dataset), "trials": trials, "hop": hop,
            "source": "Existing plotting arrays from " + frame.f_code.co_name if frame else "Plotted arrays"}


def capture_figure(fig, metadata):
    """Extract only numerical evidence and metric coordinates from staged artists."""
    hop = metadata.get("hop")
    colour_keys = {}
    colour_labels = {}
    for legend in [*fig.legends, *(axis.get_legend() for axis in fig.axes if axis.get_legend())]:
        handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", ()))
        for handle, text in zip(handles, legend.get_texts()):
            if hasattr(handle, "get_color"):
                colour_keys.setdefault(to_hex(handle.get_color()), _method(text.get_text(), hop))
                colour_labels.setdefault(to_hex(handle.get_color()), text.get_text())
    for axis in fig.axes:
        for line in axis.lines:
            key = _method(line.get_label(), hop)
            if key:
                colour_keys.setdefault(to_hex(line.get_color()), key)

    panels = []
    for axis in fig.axes:
        if hasattr(axis, "_colorbar") or axis.get_label() == "<colorbar>":
            continue
        title = axis.get_title()
        panel = {"title": re.sub(r"^\([a-z]\)\s*", "", title),
                 "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel(),
                 "xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim()),
                 "xscale": axis.get_xscale(), "yscale": axis.get_yscale(),
                 "lines": [], "images": [], "bars": [], "bands": [], "scatter": [], "arrows": [],
                 "xticks": list(axis.get_xticks()), "yticks": list(axis.get_yticks()),
                 "xticklabels": [t.get_text() for t in axis.get_xticklabels()],
                 "yticklabels": [t.get_text() for t in axis.get_yticklabels()]}
        banks = []
        for line in axis.lines:
            x, y = _array(line.get_xdata()), _array(line.get_ydata())
            if not len(x):
                continue
            label = line.get_label()
            colour = to_hex(line.get_color())
            key = _method(label, hop)
            unlabeled = label.startswith("_")
            if unlabeled:
                key = colour_keys.get(colour)
                if "posterior" in title.lower():
                    key = "posterior_blend"
                if "regret" in title.lower() and key == "best_fixed_support":
                    key, label = None, "Path-code bound"
                    unlabeled = False
                elif "regret" in title.lower() and key == "posterior_blend":
                    label = "U-STFT regret"
                    unlabeled = False
            transform = line.get_transform()
            ref_axis = ("horizontal" if transform == axis.get_yaxis_transform() else
                        "vertical" if transform == axis.get_xaxis_transform() else None)
            if ref_axis:
                key = None
            if unlabeled and key is None and not ref_axis and len(x) > 2 and colour in {"#9e9e9e", "#a8a8a8", "#999999"}:
                banks.append({"x": x, "y": y})
                continue
            if unlabeled:
                label = LABELS[key] if key else ("Reference" if ref_axis else "Signal")
            else:
                label = _curve_label(label, key)
            panel["lines"].append({"x": x, "y": y, "method": key,
                                    "label": label, "reference_axis": ref_axis,
                                    "linestyle": line.get_linestyle() if key is None else None})
        for im in axis.images:
            panel["images"].append({"values": _array(im.get_array()),
                                    "extent": list(im.get_extent()), "clim": list(im.get_clim()),
                                    "cmap": im.get_cmap().name, "label": _image_label(im, title)})
        for collection in axis.collections:
            if isinstance(collection, QuadMesh):
                coords = collection.get_coordinates()
                values = _array(collection.get_array())
                if values.ndim == 1:
                    values = values.reshape(coords.shape[0] - 1, coords.shape[1] - 1)
                panel["images"].append({"values": values,
                    "extent": [float(coords[0, 0, 0]), float(coords[0, -1, 0]),
                               float(coords[0, 0, 1]), float(coords[-1, 0, 1])],
                    "clim": list(collection.get_clim()), "cmap": collection.get_cmap().name,
                    "label": _image_label(collection, title)})
            elif isinstance(collection, PathCollection):
                panel["scatter"].append(_array(collection.get_offsets()))
            elif isinstance(collection, PolyCollection):
                colours = collection.get_facecolors()
                key = colour_keys.get(to_hex(colours[0])) if len(colours) else None
                for path in collection.get_paths():
                    panel["bands"].append({"vertices": _array(path.vertices), "method": key})
        for container in axis.containers:
            if not hasattr(container, "patches"):
                continue
            for patch in container.patches:
                colour = to_hex(patch.get_facecolor())
                key = _method(container.get_label(), hop) or colour_keys.get(colour)
                if key is None:
                    tick = int(round(patch.get_x() + patch.get_width() / 2))
                    labels = panel["xticklabels"]
                    key = _method(labels[tick], hop) if 0 <= tick < len(labels) else None
                panel["bars"].append({"x": patch.get_x(), "width": patch.get_width(),
                                      "value": patch.get_height(), "bottom": patch.get_y(),
                                      "method": key})
        for text in axis.texts:
            if hasattr(text, "arrow_patch") and text.arrow_patch is not None and not text.get_text():
                panel["arrows"].append({"end": list(text.xy), "start": list(text.xyann),
                    "linestyle": text.arrow_patch.get_linestyle(),
                    "label": colour_labels.get(to_hex(text.arrow_patch.get_edgecolor()), "")})
        for table in axis.tables:
            panel["table"] = [[table.get_celld()[(row, col)].get_text().get_text()
                for col in sorted({c for r, c in table.get_celld() if r == row})]
                for row in sorted({r for r, c in table.get_celld()})]
        if any(not line["reference_axis"] for line in panel["lines"]) or panel["images"] or panel["bars"] or panel["scatter"] or "table" in panel:
            panels.append(panel)
        if banks:
            if len(banks) > 5:
                raise ValueError("The plotting palette supports up to five dyadic candidate states")
            bank_panel = {**panel, "title": "Fixed-window bank", "images": [], "bars": [],
                          "scatter": [], "arrows": [], "bands": [], "lines": []}
            for index, bank in enumerate(banks):
                key = f"fixed_{index}"
                bank_panel["lines"].append({**bank, "method": key, "label": LABELS[key],
                                             "reference_axis": None, "linestyle": None})
            posterior = next((line for line in panel["lines"] if line["method"] == "posterior_blend"), None)
            if posterior:
                bank_panel["lines"].append(posterior)
            panels.append(bank_panel)
    for panel in panels:
        if not panel["images"]:
            continue
        for label in ("xlabel", "ylabel"):
            if not panel[label]:
                sibling = next((other for other in panels if other["images"] and other[label]
                    and other["images"][0]["cmap"] == panel["images"][0]["cmap"]), None)
                panel[label] = sibling[label] if sibling else ("Time (s)" if label == "xlabel" else "Window (samples)")
    return {"metadata": metadata, "panels": panels}


def _image_label(image, title):
    colorbar = getattr(image, "colorbar", None)
    label = ""
    if colorbar is not None:
        label = colorbar.ax.get_ylabel() or colorbar.ax.get_xlabel() or colorbar.ax.get_title()
    if "prob" in label.lower() or "posterior" in title.lower():
        return "Probability"
    if "db" in label.lower() or (image.get_clim()[0] < 0 and image.get_clim()[1] == 0):
        return "Power (dB)"
    return label


def _save_evidence(spec, target):
    arrays = {}
    def encode(item):
        if isinstance(item, np.ndarray):
            key = f"array_{len(arrays)}"
            arrays[key] = item
            return {"array": key}
        if isinstance(item, dict):
            return {key: encode(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(value) for value in item]
        if isinstance(item, np.generic):
            return item.item()
        return item
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = encode(spec)
    np.savez_compressed(target.with_suffix(".npz"), **arrays)
    target.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def load_evidence(path):
    path = Path(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        def decode(item):
            if isinstance(item, dict) and set(item) == {"array"}:
                return arrays[item["array"]].copy()
            if isinstance(item, dict):
                return {key: decode(value) for key, value in item.items()}
            if isinstance(item, list):
                return [decode(value) for value in item]
            return item
        return decode(json.loads(path.with_suffix(".json").read_text()))


def _methods(panels):
    keys = {item["method"] for panel in panels for item in panel["lines"] + panel["bars"]
            if item["method"]}
    keys.update(method_from_label(panel["title"]) for panel in panels if panel["images"])
    return [key for key in METHOD_ORDER if key in keys]


def _message(panels):
    panel = _zoom(panels[0])
    probability = next((image for image in panel["images"]
                        if image.get("label", "").lower() == "probability"
                        or ("posterior" in panel["title"].lower() and image.get("clim") == [0, 1])), None)
    if probability is not None:
        weights = np.asarray(probability["values"])
        labels = panel.get("yticklabels", [])
        if ("window" in panel["ylabel"].lower() and weights.ndim == 2
                and len(labels) == weights.shape[0] and weights.shape[0] > 1
                and np.all(np.isfinite(weights[:, -1]))
                and np.isclose(np.sum(weights[:, -1]), 1)):
            lead, runner = np.argsort(-weights[:, -1], kind="stable")[:2]
            return (f"At the end, the blend gives L={labels[lead]} weight {weights[lead, -1]:.3g}, "
                    f"then L={labels[runner]} weight {weights[runner, -1]:.3g}.")
        finite = np.asarray(probability["values"])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            return f"The largest posterior probability is {np.max(finite):.3g} in this run."
    regret = next((line for line in panel["lines"] if "regret" in line["label"].lower()), None)
    bound = next((line for line in panel["lines"] if "bound" in line["label"].lower()), None)
    if regret is not None and bound is not None and np.array_equal(regret["x"], bound["x"]):
        valid = np.isfinite(regret["y"]) & np.isfinite(bound["y"])
        if np.any(valid):
            r, b = regret["y"][valid], bound["y"][valid]
            if np.all(r <= b + 1e-9 * np.maximum(1, np.abs(b))):
                return f"The regret stays within its path-code bound ({r[-1]:.3g} at the end, versus {b[-1]:.3g})."
            return f"The final regret is {r[-1]:.3g}; its path-code bound is {b[-1]:.3g}."
    series = [line for line in panel["lines"] if line["method"] and np.any(np.isfinite(line["y"]))]
    values = {}
    for line in series:
        finite = (np.isfinite(line["x"]) & np.isfinite(line["y"])
                  & (line["x"] >= panel["xlim"][0]) & (line["x"] <= panel["xlim"][1]))
        if not np.any(finite):
            continue
        values[line["method"]] = float(line["y"][np.flatnonzero(finite)[-1]])
    if not values:
        for bar in panel["bars"]:
            if bar["method"] and np.isfinite(bar["value"]):
                values[bar["method"]] = bar["value"]
    if len(values) >= 2:
        if np.allclose(list(values.values()), next(iter(values.values())), rtol=1e-9, atol=1e-12):
            return f"All {len(values)} methods end at {next(iter(values.values())):.3g}."
        high = any(term in panel["ylabel"].lower() for term in ("higher", "tone separation", "weight"))
        ranked = sorted(values, key=lambda key: (-values[key] if high else values[key], METHOD_ORDER.index(key)))
        lead, runner = ranked[:2]
        if np.isclose(values[lead], values[runner], rtol=1e-9, atol=1e-12):
            return f"{LABELS[lead]} and {LABELS[runner]} end together at {values[lead]:.3g}."
        direction = "highest" if high else "lowest"
        precision = 3
        while precision < 8 and f"{values[lead]:.{precision}g}" == f"{values[runner]:.{precision}g}":
            precision += 1
        return (f"{LABELS[lead]} ends {direction} at {values[lead]:.{precision}g}, "
                f"then {LABELS[runner]} at {values[runner]:.{precision}g}.")
    if values:
        key = next(iter(values))
        return f"{LABELS[key]} ends at {values[key]:.3g} in this run."
    if panel["images"]:
        limits = panel["images"][0]["clim"]
        if all(np.isfinite(limits)):
            if "db" in panel["images"][0].get("label", "").lower():
                return f"The maps share a power scale from {limits[0]:.3g} to {limits[1]:.3g} dB."
            return f"The panels share a scale from {limits[0]:.3g} to {limits[1]:.3g}."
    if panel["arrows"]:
        nodes = sum(len(array) for array in panel["scatter"])
        return f"The trellis connects {nodes} states with {len(panel['arrows'])} transitions."
    if "table" in panel:
        return f"This run compares {max(0, len(panel['table']) - 1)} reconstruction results."
    return "The figure shows the recorded signal for one run."


def _nice_limits(limits):
    lo, hi = sorted(limits)
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return lo, hi
    step = 10 ** math.floor(math.log10(span)) / 5
    return math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step


def _round_label(value):
    return f"{value:.3g}".replace("-", "−")


def _custom_ticks(ticks, labels):
    if len(ticks) != len(labels) or not len(ticks):
        return False
    for tick, label in zip(ticks, labels):
        if "mathdefault" in label:
            continue
        try:
            if not np.isclose(float(label.replace("−", "-")), tick):
                return True
        except ValueError:
            return True
    return False


def _zoom(panel):
    """Preserve physical crops, trim convergence startup, exclude distant bounds."""
    result = {**panel, "lines": [dict(line) for line in panel["lines"]]}
    if panel["ylabel"] == "Blend weight":
        # Probability axes have physical bounds, also when replaying an older
        # plot cache that predates the explicit lock_ylim presentation flag.
        result.update(ylim=[0, 1], lock_ylim=True)
    lines = [line for line in result["lines"] if not line["reference_axis"] and len(line["x"]) > 2]
    if not lines or panel["images"] or panel["scatter"] or panel["bars"]:
        return result
    convergence = panel.get("convergence", False) or any(
        s in (panel["title"] + panel["ylabel"]).lower()
        for s in ("cumulative", "average", "prefix", "regret", "convergence"))
    if convergence:
        start = min(line["x"][max(0, int(0.05 * len(line["x"])))] for line in lines)
        result["xlim"] = [max(panel["xlim"][0], float(start)), panel["xlim"][1]]
        if result["xlim"][0] > 0 and result["xlim"][1] / result["xlim"][0] >= 10:
            result["xscale"] = "log"
    values = []
    for line in lines:
        visible = (line["x"] >= result["xlim"][0]) & (line["x"] <= result["xlim"][1]) & np.isfinite(line["y"])
        values.extend(line["y"][visible])
    for band in panel.get("bands", []):
        if band["method"]:
            vertices = band["vertices"]
            visible = (vertices[:, 0] >= result["xlim"][0]) & (vertices[:, 0] <= result["xlim"][1]) & np.isfinite(vertices[:, 1])
            values.extend(vertices[visible, 1])
    if values and not result.get("lock_ylim"):
        lo, hi = float(np.min(values)), float(np.max(values))
        pad = max((hi - lo) * 0.06, abs(hi) * 0.005, 1e-9)
        if panel["yscale"] == "log":
            result["ylim"] = [max(lo / 1.06, np.finfo(float).tiny), hi * 1.06]
        else:
            result["ylim"] = [lo - pad, hi + pad]
    return result


def _close_difference(panel, field=False):
    lines = [line for line in panel["lines"] if line["method"] and len(line["x"]) > 2]
    if len(lines) < 2 or panel["images"] or "difference" in panel["ylabel"].lower():
        return None
    reference = "best_fixed_support" if field else "constrained_schedule"
    ref = next((line for line in lines if line["method"] == reference), None)
    if ref is None:
        return None
    if not all(np.array_equal(line["x"], ref["x"]) for line in lines):
        return None
    stack = np.stack([line["y"] for line in lines])
    visible = (ref["x"] >= panel["xlim"][0]) & (ref["x"] <= panel["xlim"][1])
    valid = visible & np.all(np.isfinite(stack), axis=0)
    if not np.any(valid):
        return None
    span = np.ptp(stack[:, valid])
    # Add a residual panel when the median across-method spread occupies less
    # than a quarter of the absolute range. Identical curves qualify too.
    if np.median(np.ptp(stack[:, valid], axis=0)) > 0.25 * max(span, 1e-12):
        return None
    delta = {**panel, "title": "", "ylabel": f"Difference from {LABELS[reference]}",
             "yscale": "linear", "lines": []}
    for line in lines:
        delta["lines"].append({**line, "y": line["y"] - ref["y"]})
    delta = _zoom(delta)
    return delta


def _direct_labels(axis, lines):
    def wrapped(label):
        return "\n".join(textwrap.fill(part, 22, break_long_words=False)
                         for part in label.splitlines())

    endpoints = []
    for line in lines:
        x, y = line.get_xdata(), line.get_ydata()
        valid = np.isfinite(x) & np.isfinite(y) & (x >= axis.get_xlim()[0]) & (x <= axis.get_xlim()[1])
        if not np.any(valid):
            continue
        index = np.flatnonzero(valid)[-1]
        point = axis.transAxes.inverted().transform(axis.transData.transform((x[index], y[index])))
        endpoints.append((float(point[1]), line, x[index], y[index]))
    endpoints.sort(key=lambda item: item[0])
    positions = np.array([np.clip(item[0], 0.06, 0.94) for item in endpoints])
    label_lines = max((len(wrapped(item[1].get_label()).splitlines())
                       for item in endpoints), default=1)
    height_points = axis.get_window_extent().height * 72 / axis.figure.dpi
    gap = min((18 * 1.2 * label_lines + 5) / height_points,
              0.82 / max(len(endpoints) - 1, 1))
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + gap)
    if len(positions) and positions[-1] > 0.94:
        positions -= positions[-1] - 0.94
    for i in range(len(positions) - 2, -1, -1):
        positions[i] = min(positions[i], positions[i + 1] - gap)
    for position, (_, line, x, y) in zip(positions, endpoints):
        label = wrapped(line.get_label())
        axis.annotate(label, xy=(x, y), xycoords="data", xytext=(1.025, position),
                      textcoords="axes fraction", ha="left", va="center", fontsize=18,
                      color=line.get_color(), annotation_clip=False,
                      arrowprops={"arrowstyle": "-", "color": line.get_color(),
                                  "linewidth": 1.1, "shrinkA": 2, "shrinkB": 0,
                                  "connectionstyle": "angle,angleA=0,angleB=90,rad=0"})


def render_figure(panels, metadata, message, slides=False):
    """Draw each metric equally, with all text outside its data rectangle."""
    methods = _methods(panels)
    check_palette(methods)
    count = len(panels)
    maps = all(panel["images"] for panel in panels)
    overlay = maps and any(any(not line["reference_axis"] for line in panel["lines"]) for panel in panels)
    triptych = metadata.get("layout") == "entropy_triptych" and count == 3
    columns = 3 if slides and triptych else (min(2, count) if maps else 1)
    rows = int(math.ceil(count / columns))
    width, height = SLIDE_SIZE if slides else (7.5, max(4.2, 2.7 * rows + 1.0))
    # Slides use Arial when installed, else Liberation Sans (same widths, common on
    # Linux), else DejaVu Sans; the slide margins leave room for the wider fallback.
    # Name the fonts directly: a generic "sans-serif" would be resolved again when
    # the figure is drawn after this rc_context has closed.
    rc = {"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"] if slides else ["DejaVu Sans"],
          "font.size": 18 if slides else 10,
          "axes.labelsize": 20 if slides else 11, "axes.titlesize": 18 if slides else 11,
          "xtick.labelsize": 18 if slides else 9, "ytick.labelsize": 18 if slides else 9,
          "text.color": REFERENCE_GREY, "axes.labelcolor": REFERENCE_GREY,
          "axes.edgecolor": REFERENCE_GREY,
          "xtick.color": REFERENCE_GREY, "ytick.color": REFERENCE_GREY,
          "axes.unicode_minus": True, "pdf.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(rows, columns, left=0.18 if not slides else 0.10,
            right=(0.88 if maps and not overlay else 0.69) if slides else (0.86 if maps else 0.94),
            bottom=0.16 if slides else (0.23 if maps else 0.30), top=0.79 if slides else 0.88,
            hspace=0.72 if slides else 0.85, wspace=0.40 if slides else 0.28)
        if triptych:
            grid.update(left=0.09 if slides else 0.18, right=0.82 if slides else 0.95,
                        bottom=0.22 if slides else 0.20, top=0.77 if slides else 0.93,
                        wspace=0.95, hspace=0.65)
            if slides:
                # One common scale makes the bottom-left endpoint ticks apply
                # to all three components in the slide's horizontal layout.
                limits = [_zoom(panel)["ylim"] for panel in panels]
                shared_ylim = (min(pair[0] for pair in limits), max(pair[1] for pair in limits))
        if slides:
            title_artist = fig.suptitle("", fontsize=28, fontweight="normal",
                x=0.055, y=0.965, ha="left", va="top")
            for wrap_width in range(86, 39, -1):
                title = textwrap.fill(re.sub(r"-(?=\d)", "−", message),
                    wrap_width, break_long_words=False)
                if len(title.splitlines()) > 2:
                    break
                title_artist.set_text(title)
                fig.canvas.draw()
                if title_artist.get_window_extent().x1 <= fig.bbox.width * 0.95:
                    break
            if len(title.splitlines()) > 2 or title_artist.get_window_extent().x1 > fig.bbox.width * 0.95:
                plt.close(fig)
                raise ValueError(f"Slide message cannot fit in two lines: {message}")
        handles = {}
        for index, raw in enumerate(panels):
            panel = _zoom(raw)
            if triptych and slides:
                panel = {**panel, "ylim": shared_ylim}
            axis = fig.add_subplot(grid[index // columns, index % columns])
            axis.set_xscale(panel["xscale"])
            if panel["yscale"] == "symlog":
                axis.set_yscale("symlog", linthresh=1e-4)
            else:
                axis.set_yscale(panel["yscale"])
            image = None
            for im in panel["images"]:
                cmap = im["cmap"]
                if im.get("label") == "Probability" or "posterior" in panel["title"].lower():
                    colours = PROBABILITY_COLOURS if panel["lines"] else PROBABILITY_MAP_COLOURS
                    cmap = LinearSegmentedColormap.from_list("chapter_probability", colours)
                image = axis.imshow(im["values"], origin="lower", aspect="auto", interpolation="nearest",
                    extent=im["extent"], vmin=im["clim"][0], vmax=im["clim"][1], cmap=cmap, rasterized=True)
            for band in panel.get("bands", []):
                axis.add_patch(Polygon(band["vertices"], closed=True,
                    facecolor=PALETTE[band["method"]] if band["method"] else REFERENCE_GREY,
                    edgecolor="none", alpha=0.12, zorder=1))
            drawn = []
            ordered = sorted(panel["lines"], key=lambda line: METHOD_ORDER.index(line["method"]) if line["method"] else 100)
            for offset, item in enumerate(ordered):
                key = item["method"]
                if key:
                    style = line_style(key, slides, offset, len(ordered), len(item["x"]))
                    style["label"] = item["label"]
                else:
                    style = {"color": REFERENCE_GREY, "linewidth": LINE_WIDTH,
                             "linestyle": item["linestyle"] or ":", "label": item["label"]}
                if item["reference_axis"]:
                    horizontal = item["reference_axis"] == "horizontal"
                    value = item["y"][0] if horizontal else item["x"][0]
                    limits = panel["ylim"] if horizontal else panel["xlim"]
                    if not min(limits) <= value <= max(limits):
                        label = style["label"] + (" (above axis)" if value > max(limits) else " (below axis)")
                        handles[label] = Line2D([], [], **{**style, "label": label})
                        continue
                    line = axis.axhline(value, **style) if horizontal else axis.axvline(value, **style)
                else:
                    line, = axis.plot(item["x"], item["y"], **style)
                    drawn.append(line)
                handles[line.get_label()] = line
            for bar in panel["bars"]:
                key = bar["method"]
                label = LABELS[key] if key else "Probability"
                container = axis.bar(bar["x"], bar["value"], width=bar["width"], align="edge",
                    bottom=bar["bottom"], color=PALETTE[key] if key else REFERENCE_GREY,
                    label=label)
                handles[label] = container
                if slides:
                    drawn.append(Line2D([bar["x"] + bar["width"] / 2], [bar["value"]],
                        label=label, color=PALETTE[key] if key else REFERENCE_GREY))
            for offsets in panel["scatter"]:
                axis.scatter(offsets[:, 0], offsets[:, 1], s=45, facecolors="white",
                             edgecolors=REFERENCE_GREY, linewidths=1)
            for arrow in panel["arrows"]:
                axis.annotate("", xy=arrow["end"], xytext=arrow["start"],
                    arrowprops={"arrowstyle": "-|>", "color": REFERENCE_GREY,
                                "linewidth": 1.0, "linestyle": arrow["linestyle"]})
                if arrow.get("label"):
                    key = arrow["label"]
                    handles[key] = Line2D([], [], color=REFERENCE_GREY, linewidth=1.0,
                                          linestyle=arrow["linestyle"], label=key)
            if slides and panel["arrows"]:
                last = {}
                for arrow in panel["arrows"]:
                    if arrow.get("label"):
                        last[arrow["label"]] = arrow
                for label, arrow in last.items():
                    drawn.append(Line2D([arrow["end"][0]], [arrow["end"][1]],
                                        label=label, color=REFERENCE_GREY))
            if "table" in panel:
                table = axis.table(cellText=panel["table"], cellLoc="left", loc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(18 if slides else 9)
                table.scale(1, 2.0 if slides else 1.6)
                for cell in table.get_celld().values():
                    cell.set_edgecolor("white")
                    cell.get_text().set_fontweight("normal")
                axis.set_axis_off()
            axis.set_xlim(panel["xlim"])
            if panel["bars"]:
                values = [bar["value"] for bar in panel["bars"] if np.isfinite(bar["value"])]
                if values:
                    lo, hi = min(values), max(values)
                    pad = max(0.08 * (hi - lo), 0.02 * abs(hi), 1e-9)
                    axis.set_ylim(lo - pad, hi + pad)
            else:
                axis.set_ylim(panel["ylim"])
            title = panel["title"]
            method = _method(title, metadata.get("hop"))
            if method and image is not None:
                # A data-derived rank may qualify a fixed support without
                # changing that support's shared identity, colour or order.
                title = panel.get("display_title", LABELS[method])
            if slides:
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.grid(False)
                # A horizontal metric name above the axis, with method names
                # on a second line for comparison maps. All are outside data.
                ylabel = panel["ylabel"].replace("\n", " ")
                axis.set_title(textwrap.fill(ylabel, 22 if triptych else 32) + ("\n" + title if maps else ""),
                               loc="left", pad=12, fontsize=20)
                # Round endpoint labels without inventing extra seconds or
                # frequencies outside a measured map's physical domain.
                physical_x = image is not None or panel["xlabel"].lower().startswith("time")
                xlimits = (_nice_limits(axis.get_xlim())
                           if panel["xscale"] == "linear" and not physical_x
                           else axis.get_xlim())
                ylimits = (_nice_limits(axis.get_ylim())
                           if panel["yscale"] == "linear" and image is None and not panel.get("lock_ylim")
                           else axis.get_ylim())
                axis.set_xlim(xlimits)
                axis.set_ylim(ylimits)
                axis.set_xticks(xlimits, [_round_label(x) for x in xlimits])
                axis.set_yticks(ylimits, [_round_label(y) for y in ylimits])
                if (image is not None or panel["arrows"]) and _custom_ticks(panel["yticks"], panel["yticklabels"]):
                    axis.set_ylim(panel["ylim"])
                    axis.set_yticks([panel["yticks"][0], panel["yticks"][-1]],
                                   [panel["yticklabels"][0], panel["yticklabels"][-1]])
                if panel["arrows"] and panel["scatter"] and len(panel["xticks"]):
                    axis.set_xticks([panel["xticks"][0], panel["xticks"][-1]],
                                   [panel["xticklabels"][0], panel["xticklabels"][-1]])
                axis.minorticks_off()
                axis.get_xticklabels()[0].set_ha("left")
                axis.get_xticklabels()[-1].set_ha("right")
                axis.get_yticklabels()[0].set_va("bottom")
                axis.get_yticklabels()[-1].set_va("top")
                # Only the bottom-left panel carries labels in a panel grid.
                if columns > 1 and index != (rows - 1) * columns:
                    axis.tick_params(labelbottom=False, labelleft=False)
                axis.set_xlabel(panel["xlabel"].replace("\n", " ") + " — " + metadata["trials"], labelpad=8)
                if columns > 1 and index != (rows - 1) * columns:
                    axis.set_xlabel("")
                _direct_labels(axis, drawn)
            else:
                axis.spines[["top", "right"]].set_visible(False)
                if image is None:
                    axis.grid(True, alpha=0.18, color=REFERENCE_GREY, linewidth=0.55)
                else:
                    axis.grid(False)
                axis.set_xlabel(panel["xlabel"] + "\n" + metadata["trials"])
                label_width = max(12, int(12 * axis.get_position().height * height / 2.5))
                label = panel["ylabel"].replace("\n", " ")
                label = label.replace("Difference from Hindsight schedule", "Δ vs hindsight")
                label = label.replace("Difference from Best fixed window", "Δ vs fixed")
                abbreviations = {"Cumulative": "Cumul.", "cumulative": "cumul.",
                    "Standardized": "Scaled", "standardized": "scaled",
                    "Normalized": "Norm.", "normalized": "norm.",
                    "Full-region": "Region", "Complete-region": "Region",
                    "Conditional": "Cond.", "conditional": "cond.",
                    "Candidate": "Window", "candidate": "window"}
                for phrase, short in abbreviations.items():
                    label = label.replace(phrase, short)
                axis.set_ylabel(textwrap.fill(label, label_width, break_long_words=False))
                if maps and columns > 1 and index % columns:
                    axis.set_ylabel("")
                    axis.tick_params(labelleft=False)
                if title and (image is not None or count > 1):
                    axis.set_title(textwrap.fill(title, 32), fontweight="normal", pad=9)
                for ticks, labels, setter in ((panel["xticks"], panel["xticklabels"], axis.set_xticks),
                                               (panel["yticks"], panel["yticklabels"], axis.set_yticks)):
                    if _custom_ticks(ticks, labels):
                        setter(ticks, [textwrap.fill(label.replace("\n", " "), 17, break_long_words=False) for label in labels])
                if panel["xscale"] == "log":
                    axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _round_label(value)))
                    axis.xaxis.set_minor_formatter(NullFormatter())
            if triptych:
                axis.set_xlabel("")
                if not slides and index < count - 1:
                    axis.tick_params(labelbottom=False)
            if image is not None:
                if slides and overlay:
                    position = axis.get_position()
                    cax = fig.add_axes((0.92, position.y0, 0.015, position.height))
                    colorbar = fig.colorbar(image, cax=cax)
                else:
                    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
                colorbar.outline.set_visible(False)
                if slides:
                    colorbar.set_ticks(image.get_clim(), labels=[_round_label(x) for x in image.get_clim()])
                    colorbar.ax.minorticks_off()
                else:
                    colorbar.set_label(panel["images"][0].get("label") or "Shared scale", fontsize=9)
        if slides:
            # A two-line message and a two-line map heading need separate
            # vertical space. Reserve it using their actual rendered extents.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            headline = title_artist.get_window_extent(renderer)
            heading_tops = [text.get_window_extent(renderer).y1
                            for axis in fig.axes
                            for text in (axis.title, axis._left_title)
                            if text.get_visible() and text.get_text()]
            overlap = max(heading_tops, default=0) + 8 - headline.y0
            if overlap > 0:
                grid.update(top=grid.top - overlap / fig.bbox.height)
        if triptych:
            fig.supxlabel("Time (s); hindsight: min scaled time + frequency H3 — " + metadata["trials"],
                          fontsize=20 if slides else 10, y=0.065 if slides else 0.115)
        if handles and not slides:
            labels = list(handles)
            for columns_legend in range(min(3, len(labels)), 0, -1):
                legend = fig.legend([handles[label] for label in labels], labels,
                    loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=columns_legend,
                    frameon=False, fontsize=9, handlelength=3.0)
                fig.canvas.draw()
                if legend.get_window_extent().width <= fig.bbox.width * 0.97:
                    break
                legend.remove()
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            xlabel_bottom = min(axis.xaxis.label.get_window_extent(renderer).y0
                                for axis in fig.axes if axis.axison and not hasattr(axis, "_colorbar"))
            overlap = legend.get_window_extent(renderer).y1 + 8 - xlabel_bottom
            if overlap > 0 and not triptych:
                grid.update(bottom=min(0.48, grid.bottom + overlap / fig.bbox.height))
        return fig


def _check_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_labels = [axis.xaxis.label for axis in fig.axes
                   if axis.axison and axis.xaxis.label.get_visible()
                   and axis.xaxis.label.get_text()]
    for index, label in enumerate(axis_labels):
        box = label.get_window_extent(renderer)
        for other in axis_labels[index + 1:]:
            peer = other.get_window_extent(renderer)
            if (min(box.x1, peer.x1) - max(box.x0, peer.x0) > 1
                    and min(box.y1, peer.y1) - max(box.y0, peer.y0) > 1):
                raise ValueError("Axis labels overlap between panels")
    if fig._suptitle is not None:
        headline = fig._suptitle.get_window_extent(renderer)
        for axis in fig.axes:
            for heading in (axis.title, axis._left_title):
                if heading.get_visible() and heading.get_text():
                    box = heading.get_window_extent(renderer)
                    if (min(box.x1, headline.x1) - max(box.x0, headline.x0) > 1
                            and min(box.y1, headline.y1) - max(box.y0, headline.y0) > 1):
                        raise ValueError("Slide message overlaps an axis heading")
    # Matplotlib keeps Text objects for log ticks outside the visible range;
    # Axis.draw does not paint those objects. Check only rendered tick labels.
    unpainted = set()
    for axis in fig.axes:
        for direction, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            for tick in direction.get_major_ticks() + direction.get_minor_ticks():
                if not axis.axison or not min(limits) <= tick.get_loc() <= max(limits):
                    unpainted.update((id(tick.label1), id(tick.label2)))
        if not axis.axison:
            unpainted.update((id(axis.xaxis.label), id(axis.yaxis.label),
                              id(axis.xaxis.offsetText), id(axis.yaxis.offsetText)))
    for artist in fig.findobj():
        if id(artist) in unpainted or not hasattr(artist, "get_text") or not artist.get_visible() or not artist.get_text().strip():
            continue
        box = Text.get_window_extent(artist, renderer) if isinstance(artist, Text) else artist.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        if box.x0 < 1 or box.y0 < 1 or box.x1 > fig.bbox.width - 1 or box.y1 > fig.bbox.height - 1:
            raise ValueError(f"Text runs off the figure: {artist.get_text()!r}")
        for axis in fig.axes:
            if not axis.axison or hasattr(axis, "_colorbar"):
                continue
            plot = axis.get_window_extent(renderer)
            if min(box.x1, plot.x1) - max(box.x0, plot.x0) > 1 and min(box.y1, plot.y1) - max(box.y0, plot.y0) > 1:
                raise ValueError(f"Text overlaps plotted data: {artist.get_text()!r}")


def _check_image(path, slides):
    with Image.open(path) as image:
        if slides and image.size != SLIDE_PIXELS:
            raise ValueError(f"Incorrect slide size {image.size}: {path}")
        pixels = np.asarray(image.convert("RGB"))
        if np.min(pixels) > 230 or np.std(pixels) < 1:
            raise ValueError(f"Blank figure: {path}")
        edges = np.concatenate((pixels[:3].reshape(-1, 3), pixels[-3:].reshape(-1, 3),
                                pixels[:, :3].reshape(-1, 3), pixels[:, -3:].reshape(-1, 3)))
        if np.any(edges < 245):
            raise ValueError(f"Ink touches the figure edge: {path}")


class FigureEdition:
    """One empty dated output folder; failed editions remain as evidence."""

    def __init__(self, parent=None, source="Plotted arrays from the released package"):
        parent = Path(parent or ROOT / "results" / "figures").resolve()
        stamp = datetime.now(gettz("America/New_York")).strftime("%Y-%m-%d_%H%M%S_%f")
        self.root = parent / stamp
        self.root.mkdir(parents=True, exist_ok=False)
        self.source = source
        self.entries = []
        self.evidence = []
        self._used = set()
        self.log = []
        self._write_readme("Build in progress")

    def save(self, spec, name):
        if spec["panels"] and all(panel["arrows"] and panel["scatter"] for panel in spec["panels"]):
            spec = {**spec, "metadata": {**spec["metadata"], "trials": "conceptual illustration"}}
        dataset = spec["metadata"]["dataset"]
        evidence = Path("data") / dataset / Path(name).stem
        if evidence in self._used:
            raise ValueError(f"Duplicate figure name in this edition: {evidence}")
        self._used.add(evidence)
        _save_evidence(spec, self.root / evidence)
        self.evidence.append(str(evidence.with_suffix(".json")))
        panels = spec["panels"]
        groups = []
        images = {}
        for panel in panels:
            method = _method(panel["title"], spec["metadata"].get("hop"))
            if panel["images"] and method and not panel["lines"]:
                im = panel["images"][0]
                recording = next((word for word in ("humpback", "shrimp") if word in panel["title"].lower()), "")
                signature = (recording, panel["ylabel"], tuple(panel["ylim"]), tuple(im["clim"]), im["cmap"])
                images.setdefault(signature, []).append(panel)
            else:
                groups.append([panel])
        groups = [sorted(group, key=lambda panel: METHOD_ORDER.index(_method(panel["title"], spec["metadata"].get("hop"))))
                  for group in images.values()] + groups
        if spec["metadata"].get("keep_panels_together"):
            groups = [panels]
        first = None
        for index, group in enumerate(groups):
            filename = Path(name).stem if index == 0 else Path(name).stem + "_" + slug(group[0]["title"] or f"metric-{index + 1}")
            panel_dataset = ("humpback-whale" if "humpback" in group[0]["title"].lower() else
                             "snapping-shrimp" if "shrimp" in group[0]["title"].lower() else dataset)
            relative = Path(panel_dataset) / (filename + ".png")
            if any(entry["path"] == str(relative) for entry in self.entries):
                relative = relative.with_name(relative.stem + f"_{index + 1}.png")
            self._render_pair(group, spec["metadata"], relative)
            if first is None:
                first = self.root / "dissertation" / relative
            # Every comparison map also has one shared-scale file per method.
            if (len(group) > 1 and all(panel["images"] for panel in group)
                    and spec["metadata"].get("export_individual_methods", True)):
                for panel in group:
                    key = _method(panel["title"], spec["metadata"].get("hop"))
                    method_path = relative.parent / "methods" / key / relative.name
                    self._render_pair([panel], spec["metadata"], method_path)
            if len(group) == 1 and spec["metadata"].get("export_difference", True):
                zoom = _zoom(group[0])
                delta = _close_difference(zoom, dataset in {"humpback-whale", "snapping-shrimp"})
                if delta:
                    path = relative.with_name(relative.stem + "_difference.png")
                    self._render_pair([zoom, delta], spec["metadata"], path)
        self._write_readme("Build in progress; exported figures checked")
        return first

    def _render_pair(self, panels, metadata, relative):
        message_panels = [panels[metadata["summary_panel"]]] if "summary_panel" in metadata else panels
        message = metadata.get("message_prefix", "") + _message(message_panels)
        methods = _methods(panels)
        check_palette(methods)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for slides, folder, dpi in ((False, "dissertation", 350), (True, "slides", 200)):
                fig = render_figure(panels, metadata, message, slides)
                try:
                    _check_text(fig)
                    target = self.root / folder / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(target, dpi=dpi)
                    fig.savefig(target.with_suffix(".pdf"), dpi=dpi)
                    _check_image(target, slides)
                    pdf = target.with_suffix(".pdf")
                    if pdf.stat().st_size < 1000 or not pdf.read_bytes().startswith(b"%PDF"):
                        raise ValueError(f"Invalid or blank PDF: {pdf}")
                finally:
                    plt.close(fig)
            # Fail on warnings about the figure itself (layout, glyphs, ...). Library
            # deprecation notices, such as Pillow warning about an older matplotlib,
            # say nothing about the plot and differ between Python versions.
            problems = [item for item in caught
                        if not issubclass(item.category, (DeprecationWarning, PendingDeprecationWarning))]
            if problems:
                raise RuntimeError("Plot build warnings: " + "; ".join(str(item.message) for item in problems))
        self.entries.append({"path": str(relative), "shows": "; ".join(panel["title"] or panel["ylabel"] for panel in panels),
                             "message": message, "methods": methods, "trials": metadata["trials"]})

    def _write_readme(self, status):
        counts = {}
        for entry in self.entries:
            dataset = Path(entry["path"]).parts[0]
            counts[dataset] = counts.get(dataset, 0) + 1
        lines = [f"# Figure edition {self.root.name}", "", status, "", f"Data source: {self.source}", "",
                 "Each indexed path exists in both `dissertation/` and `slides/`, as PNG and PDF.",
                 "Dissertation PNGs: 350 dpi. Slides: Arial, 2666 × 1500 pixels at 200 dpi.",
                 "Candidate labels H, 2H, … identify positions in the dyadic bank; H is recorded with the evidence.",
                 "The hindsight schedule is the finite-trellis oracle reference; no ideal asymptotic rate is asserted.", "",
                 "## File index", "", "| Dataset | Figures per set | Files per set |", "|---|---:|---:|"]
        if (self.root / "index.html").is_file():
            lines[2:2] = ["[Open all plots](index.html) · [All PNGs and PDFs in one folder](all_plots/)", ""]
        lines += [f"| {dataset} | {count} | {2 * count} |" for dataset, count in sorted(counts.items())]
        lines += ["", "| Path in each set (PNG and PDF) | Shows | Runs |", "|---|---|---|"]
        lines += [f"| `{e['path']}` | {e['shows'].replace('|', '/')} | {e['trials']} |" for e in self.entries]
        lines += ["", "## Colours and lines", "", "| Method | Colour | Line | Role |", "|---|---|---|---|"]
        lines += [f"| {LABELS[key]} | `{PALETTE[key]}` | `{LINESTYLES[key]}` | {ROLES[key]} |" for key in METHOD_ORDER]
        lines += ["", "All method lines have equal width. Dissertation markers have staggered positions and white rims.",
                  "Bounds, event markers, and other diagnostic references use #4D4D4D.", "", "## Slide messages", ""]
        lines += [f"- `{e['path']}`: {e['message']}" for e in self.entries]
        lines += ["", "## Checks", "", "- Perceptual palette thresholds checked for every exported comparison.",
                  "- Text extents checked before saving; PNG edges checked for ink.",
                  "- Nonblank PNGs, PDF headers, matching twins, and exact slide pixel sizes checked.",
                  "- Messages calculated from stored plot arrays; at most two title lines.",
                  "- No warnings permitted in the renderer.", "", "## Data and rebuild", "",
                  "Only plotted arrays and coordinates are cached as JSON + NPZ (no pickle).",
                  f"Rebuild without rerunning transforms: `python -m benchmarks.build_figures --from-edition '{self.root}'`.",
                  "A rebuild always creates a fresh edition. Previous editions remain intact.", ""]
        (self.root / "README.md").write_text("\n".join(lines))

    def finish(self):
        dissertation = {str(path.relative_to(self.root / "dissertation")) for path in (self.root / "dissertation").rglob("*.png")}
        slides = {str(path.relative_to(self.root / "slides")) for path in (self.root / "slides").rglob("*.png")}
        if not dissertation or dissertation != slides:
            raise ValueError("Dissertation/slide twins are missing")
        for folder in ("dissertation", "slides"):
            for relative in dissertation:
                if not (self.root / folder / Path(relative).with_suffix(".pdf")).is_file():
                    raise ValueError(f"Missing PDF: {relative}")
        hashes = {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (self.root / "data").rglob("*") if path.is_file()}
        (self.root / "manifest.json").write_text(json.dumps({"evidence": self.evidence, "sha256": hashes}, indent=2) + "\n")
        if (self.root / "data/wav_inventory.json").is_file():
            from benchmarks.plot_gallery import build_gallery
            build_gallery(self.root)
        (self.root / "build.log").write_text(f"PASS: {len(self.entries)} figures per set; twins, formats, dimensions, text, edges, palette.\n0 errors; 0 warnings.\n")
        self._write_readme("Build complete; automated export checks passed")
        return self.root


@contextmanager
def figure_edition(parent=None, source="Plotted arrays from the released package"):
    global _ACTIVE
    previous = _ACTIVE
    edition = FigureEdition(parent, source)
    _ACTIVE = edition
    try:
        yield edition
        edition.finish()
    except BaseException as error:
        (edition.root / "build.log").write_text(f"FAILED: {type(error).__name__}: {error}\n")
        edition._write_readme("Build failed; retained for inspection")
        raise
    finally:
        _ACTIVE = previous


def save_spec(spec, path):
    path = Path(path)
    if _ACTIVE:
        edition = _ACTIVE
    else:
        parent = path.parent
        edition = _DEFAULTS.setdefault(str(parent), None)
        if edition is None:
            edition = FigureEdition(parent)
            _DEFAULTS[str(parent)] = edition
    saved = edition.save(spec, path.name)
    _RESOLVED[str(path.resolve())] = saved
    if not _ACTIVE:
        edition.finish()
    return saved


def save_figure(fig, path, **ignored):
    return save_spec(capture_figure(fig, _context(path)), path)


def resolve_figure(path):
    path = Path(path)
    return _RESOLVED.get(str(path.resolve()), path)


def paper_figure_build(function):
    """Route only build outputs to a dated edition; experiment code is untouched."""
    @wraps(function)
    def wrapped(output_dir=None, quick=False, wav_dir=None, figure_dir=None):
        if _ACTIVE:
            return function(output_dir, quick=quick, wav_dir=wav_dir, figure_dir=figure_dir)
        parent = Path(figure_dir) if figure_dir else Path(output_dir or ROOT / "results" / "paper") / "figures"
        with figure_edition(parent, "benchmarks/run_paper.py; unchanged released transforms, seeds and metrics") as edition:
            result = function(edition.root / "data" / "audit", quick=quick, wav_dir=wav_dir,
                              figure_dir=edition.root / "dissertation")
        print(f"Wrote paired chapter figures to {edition.root}")
        return result
    return wrapped
