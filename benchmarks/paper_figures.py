"""
Render the seven manuscript figures in the dissertation style, sized for one
IEEE column.

The data come from a figure edition written by ``benchmarks.run_paper``
(``results/paper/figures/<date>/data``); nothing is recomputed except Fig. 1 and the
speech figure, which are built directly with the released package.

Style rules (after J.-L. Doumont, *Trees, Maps, and Theorems*):

* label curves directly, in their own colour, instead of using a legend;
* keep non-data ink light: grey axes, no top/right spines, no grid;
* one short, left-aligned title per panel; shared colour bars;
* render at the printed width so the text prints at 8-9 pt.

Usage::

    python -m benchmarks.paper_figures --edition results/paper/figures/<date> \
        --out Manan_Mittal___Universal_Short_Time_Fourier_Transform/figures/audit
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

from benchmarks.figure_export import load_evidence
from benchmarks.plot_style import PALETTE, PROBABILITY_COLOURS, REFERENCE_GREY
from universal_stft.numpy import stft

COLUMN_WIDTH = 3.5          # inches, IEEEtran \columnwidth
FULL_WIDTH = 7.16           # inches, IEEEtran \textwidth (figure*)
TEXT = "#222222"
LIGHT = "#BBBBBB"
SHORT_COLOUR = PALETTE["fixed_0"]
LONG_COLOUR = PALETTE["fixed_4"]

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": TEXT,
    "axes.labelcolor": REFERENCE_GREY,
    "axes.edgecolor": REFERENCE_GREY,
    "xtick.color": REFERENCE_GREY,
    "ytick.color": REFERENCE_GREY,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 4,
    "axes.unicode_minus": True,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "savefig.dpi": 600,
}


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
# One colour scheme per kind of quantity, used in every figure.
SPECTROGRAM_MAP = "magma"          # power in dB
SPECTROGRAM_DB = (-70, 0)
SPECTROGRAM_TICKS = [-70, -35, 0]
SPECTROGRAM_LABEL = "Relative power (dB)"
ENTROPY_MAP = "viridis"            # time/frequency entropy and the score
PROBABILITY_MAP = LinearSegmentedColormap.from_list("probability", PROBABILITY_COLOURS)
PATH_COLOUR = PALETTE["posterior_blend"]   # the U-STFT window path on posterior maps


def figure(width_fraction, height):
    """A figure at its printed size; constrained layout keeps text inside."""
    fig = plt.figure(figsize=(width_fraction * COLUMN_WIDTH, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    return fig


def title(axis, letter, text):
    axis.set_title(f"({letter}) {text}", loc="left", color=TEXT)


def spectrogram(axis, image, scale=1.0):
    """Draw a saved power map. ``scale`` converts the frequency unit."""
    left, right, bottom, top = image["extent"]
    return axis.imshow(
        image["values"], origin="lower", aspect="auto", cmap=SPECTROGRAM_MAP,
        extent=[left, right, bottom * scale, top * scale],
        vmin=SPECTROGRAM_DB[0], vmax=SPECTROGRAM_DB[1],
        interpolation="antialiased", rasterized=True)


def state_map(axis, image, cmap, clim=None, lengths=None):
    """Draw a (states x frames) map with window lengths as y tick labels."""
    clim = clim or image["clim"]
    artist = axis.imshow(
        image["values"], origin="lower", aspect="auto", cmap=cmap,
        extent=image["extent"], vmin=clim[0], vmax=clim[1],
        interpolation="nearest", rasterized=True)
    if lengths is not None:
        axis.set_yticks(range(len(lengths)))
        axis.set_yticklabels(lengths)
    return artist


def spectrogram_bar(fig, artist, axes):
    return colour_bar(fig, artist, axes, "bottom", SPECTROGRAM_LABEL, SPECTROGRAM_TICKS, shrink=0.6)


def attached_bar(fig, axis, artist, ticks=None):
    """A slim colour bar glued to the right edge of one panel."""
    bar = fig.colorbar(artist, cax=axis.inset_axes([1.03, 0.0, 0.045, 1.0]), ticks=ticks)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)
    return bar


def colour_bar(fig, artist, axes, location="right", label="", ticks=None, shrink=1.0):
    bar = fig.colorbar(artist, ax=axes, location=location, ticks=ticks, shrink=shrink,
                       aspect=12 if location == "right" else 30, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)
    if label:
        bar.set_label(label)
    return bar


def direct_labels(axis, items, min_gap=0.13):
    """Label curve ends at the right of an axis, nudged apart vertically.

    ``items`` is a list of ``(x_end, y_end, text, colour)`` in data units.
    """
    to_axes = axis.transAxes.inverted()
    placed = []
    for x, y, text, colour in items:
        _, y_axes = to_axes.transform(axis.transData.transform((x, y)))
        placed.append([float(np.clip(y_axes, 0.0, 1.0)), text, colour, y_axes])
    placed.sort(key=lambda item: item[0])
    for i in range(1, len(placed)):
        placed[i][0] = max(placed[i][0], placed[i - 1][0] + min_gap)
    overflow = placed[-1][0] - 1.0 if placed else 0
    if overflow > 0:
        for item in placed:
            item[0] -= overflow
    for y_label, text, colour, y_curve in placed:
        axis.annotate(text, xy=(1.0, y_curve), xycoords="axes fraction",
                      xytext=(1.05, y_label), textcoords="axes fraction",
                      va="center", ha="left", color=colour,
                      annotation_clip=False,
                      arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.5,
                                  "shrinkA": 0, "shrinkB": 0})


def line_end(line, xlim):
    x, y = np.asarray(line["x"]), np.asarray(line["y"])
    valid = np.isfinite(x) & np.isfinite(y) & (x <= xlim[1])
    index = np.flatnonzero(valid)[-1]
    return x[index], y[index]


def event_marks(axis, panel):
    """Thin grey vertical lines at the controlled-experiment events."""
    for line in panel["lines"]:
        if line["reference_axis"] == "vertical":
            axis.axvline(line["x"][0], color=REFERENCE_GREY, lw=0.6,
                         ls=line["linestyle"] or ":", zorder=3)


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_dir / f"{name}{suffix}", dpi=600 if suffix == ".pdf" else 300)
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf")


def panels_by_title(spec):
    return {panel["title"]: panel for panel in spec["panels"]}


def multiple_of_hop(length, hop):
    ratio = int(length) // int(hop)
    return "H" if ratio == 1 else f"{ratio}H"


# ---------------------------------------------------------------------------
# Fig. 1: the window-length trade-off
# ---------------------------------------------------------------------------
def window_length_tradeoff(out_dir):
    """Two tones 80 Hz apart plus a click, seen through a short and a long window.

    Top row: the two windows in time and their spectra (main lobes).
    Bottom row: the resulting spectrograms. The short window keeps the click
    sharp but blurs the two tones into one wide band; the long window
    separates the tones but smears the click over its whole length.
    """
    fs, hop, n_fft = 8000, 8, 1024
    short, long = 32, 512
    t = np.arange(int(0.8 * fs)) / fs
    low, high = 500, 580
    x = 0.3 * np.sin(2 * np.pi * low * t) + 0.18 * np.sin(2 * np.pi * high * t)
    x[int(0.4 * fs)] += 40.0                     # one-sample click at 0.4 s

    lengths = stft._candidate_lengths(n_fft, hop)
    num_frames = int(np.ceil(len(x) / hop))
    frame_time = (np.arange(num_frames) + 1) * hop / fs

    def fixed_power_db(length):
        blend = [(lengths, (lengths == length).astype(float))] * num_frames
        S = stft.generate_analysis_spectrogram(x, blend, n_fft=n_fft, hop_size=hop)
        power = np.abs(S) ** 2
        return 10 * np.log10(power / np.max(power) + 1e-12)

    fig = figure(0.85, 3.6)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2])

    # (a) the windows in time, both ending on the newest sample
    axis = fig.add_subplot(grid[0, 0])
    for length, colour in ((short, SHORT_COLOUR), (long, LONG_COLOUR)):
        window = stft._gaussian_analysis_window(length)
        lag_ms = (np.arange(length) - length + 1) / fs * 1000
        axis.plot(lag_ms, window / window.max(), color=colour)
    axis.text(-60, 1.0, "long", color=LONG_COLOUR, va="top")
    axis.text(-5, 1.0, "short", color=SHORT_COLOUR, ha="right", va="top")
    axis.set_xlim(-long / fs * 1000, 1)
    axis.set_ylim(0, 1.08)
    axis.set_yticks([0, 1])
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Weight")
    title(axis, "a", "Window")

    # (b) their spectra: the main-lobe width sets the frequency resolution
    axis = fig.add_subplot(grid[0, 1])
    for length, colour in ((short, SHORT_COLOUR), (long, LONG_COLOUR)):
        window = np.zeros(8192)
        window[:length] = stft._gaussian_analysis_window(length)
        response = np.abs(np.fft.rfft(window)) ** 2
        offset = np.fft.rfftfreq(8192, 1 / fs)
        axis.plot(offset, 10 * np.log10(response / response.max() + 1e-12), color=colour)
    axis.axvline(high - low, color=REFERENCE_GREY, lw=0.6, ls=":")
    axis.text(high - low + 10, -38, "tone\nspacing", color=REFERENCE_GREY, va="bottom")
    axis.text(395, -2, "short", color=SHORT_COLOUR, ha="right", va="top")
    axis.text(55, -20, "long", color=LONG_COLOUR, ha="left")
    axis.set_xlim(0, 400)
    axis.set_ylim(-40, 2)
    axis.set_yticks([-40, -20, 0])
    axis.set_xlabel("Offset (Hz)")
    axis.set_ylabel("Gain (dB)")
    title(axis, "b", "Main lobe")

    # (c, d) spectrograms of the same signal
    axes = []
    for column, (length, letter, name, note) in enumerate((
            (short, "c", "Short", "sharp click,\nblurred tones"),
            (long, "d", "Long", "two tones,\nsmeared click"))):
        axis = fig.add_subplot(grid[1, column])
        image = axis.imshow(
            fixed_power_db(length).T, origin="lower", aspect="auto", cmap=SPECTROGRAM_MAP,
            vmin=SPECTROGRAM_DB[0], vmax=SPECTROGRAM_DB[1], extent=[frame_time[0], frame_time[-1], 0, fs / 2],
            interpolation="antialiased", rasterized=True)
        axis.set_xlim(0.25, 0.6)
        axis.set_ylim(0, 1100)
        axis.set_xticks([0.3, 0.4, 0.5, 0.6])
        axis.set_yticks([0, 500, 1000])
        axis.set_xlabel("Time (s)")
        if column == 0:
            axis.set_ylabel("Frequency (Hz)")
        else:
            axis.tick_params(labelleft=False)
        axis.text(0.03, 0.97, note, transform=axis.transAxes, ha="left", va="top",
                  color="white", fontsize=7.5,
                  bbox={"facecolor": "black", "alpha": 0.5, "lw": 0, "pad": 1.5})
        title(axis, letter, f"$L={length}$")
        axes.append(axis)
    spectrogram_bar(fig, image, axes)
    save(fig, out_dir, "window_length_tradeoff")


# ---------------------------------------------------------------------------
# Fig. 2: the causal trellis
# ---------------------------------------------------------------------------
def window_length_trellis(spec, out_dir):
    panel = spec["panels"][0]
    colours = {"stay": REFERENCE_GREY, "grow": TEXT, "reset": "#8C8C8C"}
    styles = {"stay": "-", "grow": ":", "reset": "--"}

    fig = figure(0.85, 2.3)
    axis = fig.add_subplot()
    for arrow in panel["arrows"]:
        kind = arrow["label"]
        axis.add_patch(FancyArrowPatch(
            arrow["start"], arrow["end"], arrowstyle="-|>", mutation_scale=6,
            color=colours[kind], linestyle=styles[kind], lw=0.9, shrinkA=5, shrinkB=5,
            zorder=1))
    for point in panel["scatter"]:
        m, s = np.asarray(point).reshape(-1)[:2]
        axis.plot(m, s, "o", ms=9, mfc="white", mec=REFERENCE_GREY, mew=0.7, zorder=2)
        axis.text(m, s, str(int(s)), ha="center", va="center", fontsize=7, zorder=3)

    # A key above the plot, written as coloured words rather than a legend box.
    for i, kind in enumerate(("stay", "grow", "reset")):
        x0 = 0.02 + 0.24 * i
        axis.plot([x0, x0 + 0.07], [1.08, 1.08], transform=axis.transAxes,
                  color=colours[kind], ls=styles[kind], lw=0.9, clip_on=False)
        axis.text(x0 + 0.09, 1.08, kind, transform=axis.transAxes, color=colours[kind],
                  va="center")
    axis.set_xlim(panel["xlim"])
    axis.set_ylim(panel["ylim"])
    axis.set_xticks(panel["xticks"])
    axis.set_xticklabels(panel["xticklabels"])
    axis.set_yticks(panel["yticks"])
    axis.set_yticklabels(panel["yticklabels"])
    axis.set_xlabel("Frame $m$")
    axis.set_ylabel("Window length")
    save(fig, out_dir, "window_length_trellis")


# ---------------------------------------------------------------------------
# Fig. 3: chirp spectrograms
# ---------------------------------------------------------------------------
def chirp_spectrograms(methods_spec, bank_spec, out_dir):
    """Fixed H, U-STFT and Fixed 16H: the full record and a zoom on the fast onset."""
    methods = panels_by_title(methods_spec)
    bank = panels_by_title(bank_spec)
    rows = [("Fixed H", bank["Fixed H"]), ("U-STFT", methods["U-STFT"]),
            ("Fixed 16H", bank["Fixed 16H"])]
    zoom_time, zoom_freq = (0, 1.5), (2.8, 4.0)        # seconds, kHz

    fig = figure(0.95, 4.4)
    axes = fig.subplots(3, 2, sharex="col", width_ratios=[1.6, 1])
    letters = iter("abcdef")
    for row, (name, panel) in enumerate(rows):
        full, zoom = axes[row]
        artist = spectrogram(full, panel["images"][0], scale=1e-3)
        full.set_xlim(panel["xlim"])
        full.set_ylim(0, 4)
        full.set_ylabel("Freq. (kHz)")
        full.add_patch(plt.Rectangle((zoom_time[0], zoom_freq[0]), zoom_time[1] - zoom_time[0],
                                     zoom_freq[1] - zoom_freq[0], fill=False,
                                     edgecolor="white", lw=0.8, ls="--"))
        title(full, next(letters), name)

        spectrogram(zoom, panel["images"][0], scale=1e-3)
        zoom.set_xlim(zoom_time)
        zoom.set_ylim(zoom_freq)
        zoom.set_yticks([3.0, 3.5, 4.0])
        title(zoom, next(letters), "Onset")
    for axis in axes[:-1].flat:
        axis.tick_params(labelbottom=False)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    spectrogram_bar(fig, artist, list(axes[-1]))
    save(fig, out_dir, "hyperbolic-chirp_spectrogram_comparison")


# ---------------------------------------------------------------------------
# Fig. 4: chirp diagnostics (full page width)
# ---------------------------------------------------------------------------
SHORT_LABELS = {"posterior_blend": "U-STFT", "framewise_selection": "Framewise",
                "constrained_schedule": "Hindsight"}
METHOD_ORDER = ("posterior_blend", "framewise_selection", "best_fixed_support",
                "constrained_schedule")
HINDSIGHT_STYLE = (0, (3, 1.5))


def method_curves(axis, method_panel, bank_panel, xlim, best_fixed_label):
    """Fixed-window bank in light grey, the four methods in colour, labelled."""
    for line in bank_panel["lines"]:
        if line["method"] != "posterior_blend":
            axis.plot(line["x"], line["y"], color=LIGHT, lw=0.7, zorder=1)
    labels = []
    by_method = {line["method"]: line for line in method_panel["lines"]}
    for method in METHOD_ORDER:
        line = by_method[method]
        axis.plot(line["x"], line["y"], color=PALETTE[method], lw=1.2, zorder=3,
                  ls=HINDSIGHT_STYLE if method == "constrained_schedule" else "-")
        text = best_fixed_label if method == "best_fixed_support" else SHORT_LABELS[method]
        labels.append((*line_end(line, xlim), text, PALETTE[method]))
    top_bank = max(bank_panel["lines"], key=lambda line: np.nanmax(line["y"]))
    labels.append((*line_end(top_bank, xlim), "Fixed bank", "#999999"))
    axis.set_xlim(xlim)
    direct_labels(axis, labels)


def joint_regret_curves(axis, rows, xlim, hop):
    """2-D Renyi regret against the per-prefix optimal schedule."""
    labels = []
    states = sorted({row["method"] for row in rows if row["method"].startswith("fixed_")})
    for method in states + ["posterior_blend"]:
        points = [(float(r["time_s"]), float(r["regret_bits"])) for r in rows if r["method"] == method]
        t, regret = np.array(points).T
        if method == "posterior_blend":
            axis.plot(t, regret, color=PATH_COLOUR, lw=1.3, zorder=3)
            labels.append((t[-1], regret[-1], "U-STFT", PATH_COLOUR))
        else:
            axis.plot(t, regret, color=LIGHT, lw=0.8, zorder=1)
            ratio = 2 ** int(method.split("_")[1])
            labels.append((t[-1], regret[-1], f"Fixed {multiple_of_hop(ratio * hop, hop)}", "#999999"))
    axis.axhline(0, color=REFERENCE_GREY, lw=0.6, ls=":")
    axis.set_xlim(xlim)
    axis.set_ylim(bottom=-0.1)
    direct_labels(axis, labels, min_gap=0.11)


def chirp_diagnostics(spec, story_spec, joint_regret_rows, best_fixed_length, out_dir):
    panels = spec["panels"]
    hop = spec["metadata"]["hop"]
    score, posterior = panels[2], panels[3]
    excess, excess_bank, regret, joint, joint_bank = panels[4:9]
    lengths = score["yticklabels"]
    xlim = [0, excess["xlim"][1]]
    fixed_label = f"Fixed {multiple_of_hop(best_fixed_length, hop)}"
    time = np.asarray(posterior["lines"][0]["x"])
    hindsight = next(line for line in story_spec["panels"][1]["lines"]
                     if line["method"] == "constrained_schedule")

    fig = figure(FULL_WIDTH / COLUMN_WIDTH, 4.6)
    rows = fig.subfigures(2, 1)                      # each row lays out on its own
    axes = np.array([rows[0].subplots(1, 3), rows[1].subplots(1, 3)])

    # (a) where the best window is: the score map and the framewise minimum
    axis = axes[0, 0]
    artist = state_map(axis, score["images"][0], ENTROPY_MAP, lengths=lengths)
    values = np.where(np.isnan(score["images"][0]["values"]), np.inf, score["images"][0]["values"])
    axis.step(time, np.argmin(values, axis=0), where="mid",
              color=PALETTE["framewise_selection"], lw=1.2,
              path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")])
    axis.text(8, 3.55, "framewise\nminimum", color="white", fontsize=7.5, va="top")
    axis.set_xlim(xlim)
    axis.set_ylabel("Window $L$")
    attached_bar(rows[0], axis, artist)
    title(axis, "a", "Score $\\ell_{m,s}$")

    # (b) what the U-STFT does: blend weights and its path, next to hindsight
    axis = axes[0, 1]
    artist = state_map(axis, posterior["images"][0], PROBABILITY_MAP, [0, 1], lengths)
    axis.plot(time, posterior["lines"][0]["y"], color=PATH_COLOUR, lw=1.3)
    axis.plot(hindsight["x"], hindsight["y"], color=PALETTE["constrained_schedule"],
              lw=1.0, ls=HINDSIGHT_STYLE)
    axis.text(6, 2.1, "U-STFT", color=PATH_COLOUR)
    axis.text(6, 3.35, "hindsight", color=PALETTE["constrained_schedule"])
    axis.set_xlim(xlim)
    axis.set_ylabel("Window $L$")
    attached_bar(rows[0], axis, artist, ticks=[0, 1])
    title(axis, "b", "Blend weights $\\mu_m(s)$")

    # (c) the result: global concentration of the spectrogram so far
    axis = axes[0, 2]
    method_curves(axis, joint, joint_bank, xlim, fixed_label)
    axis.set_ylabel("Bits")
    title(axis, "c", "Joint 2-D $\\mathcal{H}_3^{\\mathrm{TF}}$")

    # (d) cumulative loss above the framewise minimum
    axis = axes[1, 0]
    method_curves(axis, excess, excess_bank, xlim, fixed_label)
    axis.set_ylabel("Nats")
    title(axis, "d", "Cumulative excess loss")

    # (e) regret per frame under its path-code bound
    axis = axes[1, 1]
    ustft = next(line for line in regret["lines"] if line["label"] == "U-STFT regret")
    bound = next(line for line in regret["lines"] if line["label"] == "Path-code bound")
    axis.fill_between(ustft["x"], ustft["y"], bound["y"], color=PATH_COLOUR, alpha=0.08, lw=0)
    axis.plot(bound["x"], bound["y"], color=REFERENCE_GREY, ls=(0, (4, 2)), lw=1.0)
    axis.plot(ustft["x"], ustft["y"], color=PATH_COLOUR, lw=1.3)
    axis.set_xlim(xlim)
    axis.set_ylim(bottom=0)
    direct_labels(axis, [(*line_end(ustft, xlim), "U-STFT", PATH_COLOUR),
                         (*line_end(bound, xlim), "Bound", REFERENCE_GREY)], min_gap=0.2)
    axis.set_ylabel("Nats/frame")
    title(axis, "e", "Mix-loss regret per frame")

    # (f) 2-D Renyi regret against the best schedule for each prefix
    axis = axes[1, 2]
    joint_regret_curves(axis, joint_regret_rows, xlim, hop)
    axis.set_ylabel("Bits")
    title(axis, "f", "2-D $\\mathcal{H}_3^{\\mathrm{TF}}$ regret")

    for axis in axes.flat:
        axis.set_xticks([0, 5, 10, 15])
    for axis in axes[1]:
        axis.set_xlabel("Time (s)")
    save(fig, out_dir, "hyperbolic-chirp_diagnostics")


# ---------------------------------------------------------------------------
# Fig. 5: controlled tone and impulse experiment
# ---------------------------------------------------------------------------
def controlled_evidence(spec, best_fixed_length, out_dir):
    panels = panels_by_title(spec)
    lengths = panels["Candidate score"]["yticklabels"]
    xlim = [0, 4]

    fig = figure(0.95, 4.8)
    axes = fig.subplots(3, 2, sharex=True)
    for axis in axes[:-1].flat:
        axis.tick_params(labelbottom=False)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")

    axis = axes[0, 0]
    signal_panel = panels["Signal and events"]
    signal_line = next(line for line in signal_panel["lines"] if line["label"] == "Signal")
    axis.plot(signal_line["x"], signal_line["y"], color=REFERENCE_GREY, lw=0.3, rasterized=True)
    event_marks(axis, signal_panel)
    axis.set_xlim(xlim)
    axis.set_ylabel("Amplitude")
    title(axis, "a", "Signal and events")

    axis = axes[0, 1]
    artist = state_map(axis, panels["Candidate score"]["images"][0], ENTROPY_MAP,
                       lengths=lengths)
    event_marks(axis, panels["Candidate score"])
    axis.set_ylabel("Window $L$")
    colour_bar(fig, artist, axis)
    title(axis, "b", "Score $\\ell_{m,s}$")

    axis = axes[1, 0]
    panel = panels["Posterior support"]
    artist = state_map(axis, panel["images"][0], PROBABILITY_MAP, [0, 1], lengths)
    for line in panel["lines"]:
        if line["reference_axis"] is None:
            axis.plot(line["x"], line["y"], color=PATH_COLOUR, lw=1.1)
    event_marks(axis, panel)
    axis.set_ylabel("Window $L$")
    colour_bar(fig, artist, axis, ticks=[0, 1])
    title(axis, "c", "Posterior $\\mu_m(s)$")

    for axis, name, letter in ((axes[1, 1], "U-STFT", "d"),
                               (axes[2, 0], "Framewise selection", "e"),
                               (axes[2, 1], "Best fixed window", "f")):
        artist = spectrogram(axis, panels[name]["images"][0], scale=1e-3)
        event_marks(axis, panels[name])
        axis.set_ylim(0, 2)
        axis.set_ylabel("Freq. (kHz)")
        label = f"Fixed $L={best_fixed_length}$" if name == "Best fixed window" else name
        title(axis, letter, label)
    axes[0, 0].set_xlim(xlim)
    spectrogram_bar(fig, artist, list(axes[2]))
    save(fig, out_dir, "controlled_evidence")


# ---------------------------------------------------------------------------
# Fig. 6: field recordings
# ---------------------------------------------------------------------------
def field_spectrograms(spec, best_fixed_lengths, out_dir):
    panels = panels_by_title(spec)
    rows = ("U-STFT", "Framewise selection", "Best fixed window")
    recordings = (("Humpback", "Humpback whale", 4), ("Shrimp", "Snapping shrimp", 24))

    fig = figure(0.95, 4.6)
    axes = fig.subplots(3, 2, sharex=True)
    letters = iter("abcdef")
    for row, method in enumerate(rows):
        for column, (key, heading, top) in enumerate(recordings):
            axis = axes[row, column]
            artist = spectrogram(axis, panels[f"{key}: {method}"]["images"][0])
            axis.set_xlim(0, 5)
            axis.set_ylim(0, top)
            if column == 0:
                axis.set_ylabel("Frequency (kHz)")
            name = {"Framewise selection": "Framewise",
                    "Best fixed window": f"Fixed $L={best_fixed_lengths[key]}$"}.get(method, method)
            title(axis, next(letters), name)
    for axis in axes[:-1].flat:
        axis.tick_params(labelbottom=False)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    for axis, (_, heading, _) in zip(axes[0], recordings):
        axis.annotate(heading, xy=(0.5, 1.0), xycoords="axes fraction",
                      xytext=(0, 16), textcoords="offset points",
                      ha="center", va="bottom", fontweight="bold", color=TEXT)
    spectrogram_bar(fig, artist, list(axes[-1]))
    save(fig, out_dir, "field_spectrogram_comparison")


# ---------------------------------------------------------------------------
# Fig. 7: English speech
# ---------------------------------------------------------------------------
SPEECH_WAV = Path(__file__).resolve().parent / "wav" / "english_speech_1.wav"
SPEECH_EXCERPT = (1.2, 6.2)        # seconds of the original recording
SPEECH_ZOOM = (0.30, 0.50)         # seconds of the excerpt: stop bursts, then a vowel


def speech_example(out_dir, wav_path=SPEECH_WAV):
    """VCTK speaker p227 at 16 kHz with H = 2 ms and 16H = 32 ms.

    Rows: Fixed H, U-STFT and Fixed 16H, each on the full excerpt and on a
    zoom where two stop bursts lead into a vowel; bottom row: the blend weights.
    Computed here with the released package. Prints the numbers quoted in
    the text (global Renyi-3 after warm-up, switching rates, reconstruction).
    """
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from benchmarks.run_paper import _joint_spectrogram_renyi3_prefix

    if not Path(wav_path).exists():
        raise FileNotFoundError(f"{wav_path} is not bundled; see benchmarks/wav/README.md")
    fs, x = wavfile.read(wav_path)
    x = x / float(np.iinfo(x.dtype).max) if np.issubdtype(x.dtype, np.integer) else x.astype(float)
    x = x.mean(axis=1) if x.ndim > 1 else x
    x = resample_poly(x, 1, 3)                    # 48 kHz -> 16 kHz
    fs //= 3
    x = x[int(SPEECH_EXCERPT[0] * fs):int(SPEECH_EXCERPT[1] * fs)]

    hop, n_fft = 32, 1024                         # candidate windows H, 2H, ..., 16H
    result = stft.universal_short_time_fourier_transform(x, n_fft, hop, beta=1.5, mode="filtering")
    lengths = result.blend_info[0][0]
    shortest, longest = result.all_candidate_spectrograms[0], result.all_candidate_spectrograms[-1]

    # Numbers for the text, after the warm-up where the long windows are filling.
    warm_up = int(lengths[-1] // hop)
    scores = np.where(result.candidate_entropies == 0, np.inf, result.candidate_entropies)
    framewise_state = np.argmin(scores, axis=1)
    frames = np.arange(len(framewise_state))
    framewise = result.all_candidate_spectrograms[framewise_state, frames]
    bits = lambda S: _joint_spectrogram_renyi3_prefix(S[warm_up:], n_fft)["entropy_bits"][-1]
    print("speech: global H3 (bits)")
    for i, length in enumerate(lengths):
        print(f"  Fixed {multiple_of_hop(length, hop):>4}: {bits(result.all_candidate_spectrograms[i]):.2f}")
    print(f"  Framewise : {bits(framewise):.2f}")
    print(f"  U-STFT    : {bits(result.analysis_spectrogram):.2f}")
    print(f"  switches per frame: framewise {np.mean(np.diff(framewise_state) != 0):.3f},"
          f" U-STFT {np.mean(np.diff(result.dominant_lengths) != 0):.3f}")
    print(f"  frames on 16H: {np.mean(result.dominant_lengths == lengths[-1]):.2f}")
    rebuilt = stft.inverse_universal_spectrogram(result.filtering_spectrogram, len(x), hop, n_fft)
    print(f"  reconstruction error: {np.linalg.norm(x - rebuilt) / np.linalg.norm(x):.1e}")

    # One shared dB scale for the three spectrograms.
    powers = [np.abs(S.T) ** 2 for S in (shortest, result.analysis_spectrogram, longest)]
    reference = max(power.max() for power in powers)
    images = [10 * np.log10(power / reference + 1e-12) for power in powers]
    frame_time = (frames + 1) * hop / fs
    extent = [frame_time[0], frame_time[-1], 0, fs / 2000]

    fig = figure(0.95, 5.2)
    grid = fig.add_gridspec(4, 2, width_ratios=[1.6, 1], height_ratios=[1, 1, 1, 0.75])
    letters = iter("abcdefg")
    names = ("Fixed H", "U-STFT", "Fixed 16H")
    spectrogram_axes = []
    for row, (name, image) in enumerate(zip(names, images)):
        full = fig.add_subplot(grid[row, 0], sharex=spectrogram_axes[0][0] if row else None)
        zoom = fig.add_subplot(grid[row, 1], sharex=spectrogram_axes[0][1] if row else None)
        for axis in (full, zoom):
            artist = axis.imshow(image, origin="lower", aspect="auto", cmap=SPECTROGRAM_MAP,
                                 vmin=SPECTROGRAM_DB[0], vmax=SPECTROGRAM_DB[1], extent=extent,
                                 interpolation="antialiased", rasterized=True)
        full.set_xlim(0, 5)
        full.set_ylim(0, 8)
        full.set_ylabel("Freq. (kHz)")
        full.add_patch(plt.Rectangle((SPEECH_ZOOM[0], 0.05), SPEECH_ZOOM[1] - SPEECH_ZOOM[0], 7.9,
                                     fill=False, edgecolor="white", lw=0.8, ls="--"))
        zoom.set_xlim(SPEECH_ZOOM)
        zoom.set_ylim(0, 8)
        zoom.set_xticks([0.3, 0.4, 0.5])
        title(full, next(letters), name)
        title(zoom, next(letters), "Onset")
        spectrogram_axes.append((full, zoom))
    for full, zoom in spectrogram_axes:
        full.tick_params(labelbottom=False)
    for zoom in [pair[1] for pair in spectrogram_axes[:-1]]:
        zoom.tick_params(labelbottom=False)
    spectrogram_axes[-1][1].set_xlabel("Time (s)")

    # The spectrogram colour bar sits in the free cell beside the blend weights.
    holder = fig.add_subplot(grid[3, 1])
    holder.axis("off")
    bar = fig.colorbar(artist, cax=holder.inset_axes([0.12, 0.62, 0.76, 0.1]),
                       orientation="horizontal", ticks=SPECTROGRAM_TICKS)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)
    bar.set_label(SPECTROGRAM_LABEL.replace(" ", "\n", 1))   # two lines fit the cell

    # (g) blend weights and the U-STFT path, on the same time axis as (a), (c), (e)
    axis = fig.add_subplot(grid[3, 0], sharex=spectrogram_axes[0][0])
    weights = state_map(axis, {"values": result.pmf_history.T, "clim": [0, 1],
                               "extent": [frame_time[0], frame_time[-1], -0.5, len(lengths) - 0.5]},
                        PROBABILITY_MAP, [0, 1], [multiple_of_hop(L, hop) for L in lengths])
    path = np.searchsorted(lengths, result.dominant_lengths)
    axis.step(frame_time, path, where="mid", color=PATH_COLOUR, lw=0.9)
    axis.set_xlim(0, 5)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Window $L$")
    attached_bar(fig, axis, weights, ticks=[0, 1])
    title(axis, next(letters), "Blend weights $\\mu_m(s)$")
    save(fig, out_dir, "english-speech_spectrogram_comparison")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def best_fixed_lengths(data_dir):
    lengths = {}
    with open(data_dir / "audit" / "joint_renyi3.csv") as file:
        for row in csv.DictReader(file):
            lengths.setdefault(row["case"], int(row["best_fixed_length"]))
    return lengths


def read_joint_regret(path):
    """Rows of joint_entropy_regret.csv from benchmarks.wav_collection."""
    with open(path) as file:
        return list(csv.DictReader(file))


def latest_joint_regret_csv():
    """The newest chirp 2-D regret table written by benchmarks.wav_collection."""
    candidates = sorted(Path("results/paper_wavs").glob("*/data/*hyperbolic-chirp/joint_entropy_regret.csv"))
    if not candidates:
        raise FileNotFoundError("run benchmarks.wav_collection first, or pass --chirp-regret")
    return candidates[-1]


def build(edition, out_dir, joint_regret_csv=None):
    data = Path(edition) / "data"
    out_dir = Path(out_dir)
    fixed = best_fixed_lengths(data)
    joint_regret_csv = joint_regret_csv or latest_joint_regret_csv()
    with plt.rc_context(STYLE):
        window_length_tradeoff(out_dir)
        window_length_trellis(load_evidence(data / "audit/window_length_trellis"), out_dir)
        chirp_spectrograms(
            load_evidence(data / "hyperbolic-chirp/hyperbolic-chirp_spectrogram_comparison"),
            load_evidence(data / "hyperbolic-chirp/hyperbolic-chirp_spectrogram_comparison_fixed_bank"),
            out_dir)
        chirp_diagnostics(load_evidence(data / "hyperbolic-chirp/hyperbolic-chirp_diagnostics"),
                          load_evidence(data / "hyperbolic-chirp/chirp_regret_story"),
                          read_joint_regret(joint_regret_csv), fixed["hyperbolic-chirp"], out_dir)
        controlled_evidence(load_evidence(data / "tone-impulse-conflict/controlled_evidence"),
                            fixed["tone-impulse-conflict"], out_dir)
        field_spectrograms(load_evidence(data / "tone-impulse-conflict/field_spectrogram_comparison"),
                           {"Humpback": fixed["humpback-whale"], "Shrimp": fixed["snapping-shrimp"]},
                           out_dir)
        speech_example(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edition", type=Path, required=True,
                        help="figure edition folder written by benchmarks.run_paper")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--chirp-regret", type=Path, default=None,
                        help="joint_entropy_regret.csv for the chirp (default: newest in results/paper_wavs)")
    args = parser.parse_args()
    build(args.edition, args.out, args.chirp_regret)
