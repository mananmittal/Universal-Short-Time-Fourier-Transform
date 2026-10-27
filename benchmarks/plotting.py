"""Compatibility spectrogram helpers using the shared chapter exporter."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.figure_export import save_spec, render_figure, _message, slug
from benchmarks.plot_suites import panel


def save_power_map(power, title, path, fs, hop_size, extent=None,
                   shared_reference=None, shared_limits=(-60.0, 0.0), pdf=None):
    """Export a map; comparisons supply their pooled reference explicitly."""
    power = np.asarray(power, dtype=float)
    reference = (float(shared_reference) if shared_reference is not None else
                 float(np.percentile(power[np.isfinite(power)], 99.8)))
    reference = max(reference, np.finfo(float).tiny)
    db = 10 * np.log10(np.maximum(power, reference * 10 ** (shared_limits[0] / 10)) / reference)
    extent = list(extent) if extent is not None else [0, power.shape[0] * hop_size / fs, 0, fs / 2]
    item = panel(title, 'Time (s)', 'Frequency (Hz)', extent[:2], extent[2:])
    item['images'] = [{'values': db.T, 'extent': extent, 'clim': list(shared_limits),
                       'cmap': 'magma', 'label': 'Power (dB)'}]
    metadata = {'dataset': slug(title), 'trials': 'single run', 'hop': hop_size,
                'source': 'Power array supplied to the shared spectrogram helper'}
    spec = {'metadata': metadata, 'panels': [item]}
    return _export_map(spec, path, pdf)


def _export_map(spec, path, pdf=None):
    saved = save_spec(spec, path)
    if pdf is not None:
        fig = render_figure(spec['panels'], spec['metadata'], _message(spec['panels']))
        try:
            pdf.savefig(fig, dpi=350)
        finally:
            plt.close(fig)
    return saved


def plot_single_spectrogram(S, title, pdf, fs, hop_size, extent=None,
                            t_mesh=None, f_mesh=None, mode='imshow',
                            shared_reference=None, shared_limits=(-60.0, 0.0)):
    if mode not in {'imshow', 'pcolormesh'}:
        raise ValueError('mode must be imshow or pcolormesh')
    if extent is None and mode == 'pcolormesh':
        extent = [np.min(t_mesh), np.max(t_mesh), np.min(f_mesh), np.max(f_mesh)]
    destination = Path(getattr(pdf, '_filename', 'results/figures/report.pdf')).parent / (slug(title) + '.png')
    return save_power_map(S, title, destination, fs, hop_size, extent,
                          shared_reference, shared_limits, pdf)


def show_mel_individual(data, title, pdf, fs, hop_size, mel_centers,
                        shared_reference=None, shared_limits=(-70.0, 0.0)):
    """Keep the Mel-bin coordinates and label them with the supplied Hz centres."""
    data = np.asarray(data, dtype=float)
    reference = float(shared_reference) if shared_reference is not None else float(np.max(data))
    reference = max(reference, np.finfo(float).tiny)
    db = 10 * np.log10(np.maximum(data, reference * 10 ** (shared_limits[0] / 10)) / reference)
    duration, bins = data.shape[0] * hop_size / fs, data.shape[1]
    item = panel(title, 'Time (s)', 'Frequency (Hz)', (0, duration), (0, bins))
    item['images'] = [{'values': db.T, 'extent': [0, duration, 0, bins],
                       'clim': list(shared_limits), 'cmap': 'magma', 'label': 'Power (dB)'}]
    ticks = np.linspace(0, bins - 1, 6)
    item['yticks'] = ticks.tolist()
    item['yticklabels'] = [str(int(mel_centers[int(i) + 1])) for i in ticks]
    metadata = {'dataset': slug(title), 'trials': 'single run', 'hop': hop_size,
                'source': 'Supplied Mel-power array and frequency centres'}
    destination = Path(getattr(pdf, '_filename', 'results/figures/report.pdf')).parent / (slug(title) + '.png')
    return _export_map({'metadata': metadata, 'panels': [item]}, destination, pdf)
