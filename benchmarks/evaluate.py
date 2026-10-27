import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from universal_stft.numpy import stft
from benchmarks.baselines import fixed_causal_spectrogram, inverse_spectrogram
from benchmarks.metrics import running_2d_renyi, phase_jitter, spectral_convergence, envelope_spectrum_kurtosis
from benchmarks.plotting import plot_single_spectrogram, show_mel_individual


def _safe_name(title):
    """Filesystem-safe slug for a signal title."""
    return "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).replace(" ", "_")


def _save_spectrogram_png(S_power, title, png_path, fs, hop_size):
    """Compatibility helper using the shared paired exporter."""
    from benchmarks.plotting import save_power_map
    return save_power_map(S_power, title, png_path, fs, hop_size)


def run_full_evaluation(x, fs, title, out_dir, n_fft, hop_size, beta=0.4):
    """
    Unified benchmarking pipeline for both real-life audio and synthetic sequences.
    """
    print(f"================ Processing {title} ================")
    
    # 1. Run Universal STFT
    print("-> Running Universal STFT...")
    S_univ, S_filtering, dom_lengths, pmf_history, blend_info, cand_ent, blend_ent, blend_h_time, blend_h_spec, cand_h_time, cand_h_spec, _ = \
        stft.universal_short_time_fourier_transform(x, n_fft=n_fft, hop_size=hop_size, beta=beta, mode="filtering")
    
    cand_lengths = blend_info[0][0]
    n_frames = min(S_univ.shape[0], len(x) // hop_size)
    
    # Compute baseline candidate spectrograms using fixed causal spectrogram
    print("-> Computing baseline fixed causal spectrograms...")
    all_cand_specs = []
    for L in cand_lengths:
        S_cand_fixed, _ = fixed_causal_spectrogram(x, L=int(L), n_fft=n_fft, hop_size=hop_size)
        all_cand_specs.append(S_cand_fixed)
    
    # Enforce the same causal availability mask in all candidate-entropy plots.
    print("-> Enforcing strict causality limits on candidates...")
    samples_seen = (np.arange(cand_ent.shape[0]) + 1) * hop_size
    valid_by_frame = cand_lengths[None, :] <= samples_seen[:, None]
    for t_idx in range(cand_ent.shape[0]):
        valid_indices = np.flatnonzero(valid_by_frame[t_idx])
        
        # If no window is fully active yet, default to the shortest possible window (c=0)
        max_active = valid_indices[-1] if len(valid_indices) > 0 else 0
        
        cand_ent[t_idx, max_active + 1:] = cand_ent[t_idx, max_active]
        cand_h_time[t_idx, max_active + 1:] = cand_h_time[t_idx, max_active]
        cand_h_spec[t_idx, max_active + 1:] = cand_h_spec[t_idx, max_active]
        
        # Note: No need to overwrite candidate spectrograms for causality —
        # fixed_causal_spectrogram is inherently causal by construction.

    # Pre-calculate candidate metrics for dynamic selection
    cand_2d_renyi_over_time = [running_2d_renyi(S) for S in all_cand_specs]
    
    # Select representative fixed windows by final cumulative 2-D Rényi entropy.
    print("-> Selecting representative fixed windows by 2-D Rényi entropy...")
    final_entropies = [r[-1] for r in cand_2d_renyi_over_time]
    sorted_indices = np.argsort(final_entropies)
    
    best_fixed_idx = sorted_indices[0]
    mid_fixed_idx = sorted_indices[len(sorted_indices) // 2]
    worst_fixed_idx = sorted_indices[-1]
    
    S_fixed_best = all_cand_specs[best_fixed_idx]
    best_L = int(cand_lengths[best_fixed_idx])
    
    target_indices = [best_fixed_idx, mid_fixed_idx, worst_fixed_idx]
    target_labels = ['Lowest entropy', 'Median entropy', 'Highest entropy']
    
    print(
        f"   -> Selected lowest (L={best_L}), median "
        f"(L={int(cand_lengths[mid_fixed_idx])}), and highest "
        f"(L={int(cand_lengths[worst_fixed_idx])}) final-entropy baselines."
    )

    # Apply explicit recursive warmstarting fix to perfectly mirror the cumulative 2D-Renyi tracking
    for t_idx in range(n_frames):
        valid_indices = np.flatnonzero(valid_by_frame[t_idx])
        max_act = valid_indices[-1] if len(valid_indices) > 0 else 0
        for c in range(max_act + 1, len(cand_lengths)):
            if t_idx < len(cand_2d_renyi_over_time[c]) and t_idx < len(cand_2d_renyi_over_time[max_act]):
                cand_2d_renyi_over_time[c][t_idx] = cand_2d_renyi_over_time[max_act][t_idx]

    # Entropy tracking on the same causal, dyadic candidate bank.
    u_renyi_2d_run = running_2d_renyi(S_univ)
    
    # 5. Reconstruction & Metrics
    print("-> Calculating Waveform Reconstruction Metrics & Candidate Tracking...")
    
    x_rec_univ = stft.inverse_universal_spectrogram(S_filtering, len(x), hop_size, n_fft, int(np.max(blend_info[0][0])))
    univ_mse, univ_pj, univ_sc = np.mean((x - x_rec_univ) ** 2), phase_jitter(x, x_rec_univ), spectral_convergence(x, x_rec_univ)

    min_len_stft = min(len(x), len(x_rec_univ)) # we use the best cand below
    x_rec_stft = inverse_spectrogram(S_fixed_best, len(x), np.full(S_fixed_best.shape[0], best_L), hop_size, n_fft)
    stft_mse = np.mean((x[:min_len_stft] - x_rec_stft[:min_len_stft])**2)
    stft_sc = spectral_convergence(x[:min_len_stft], x_rec_stft[:min_len_stft])
    stft_pj = phase_jitter(x[:min_len_stft], x_rec_stft[:min_len_stft])

    u_kurt_2d = envelope_spectrum_kurtosis(S_univ)

    cand_metrics = []

    for c, L in enumerate(cand_lengths):
        S_cand = all_cand_specs[c]
        x_rec_cand = inverse_spectrogram(S_cand, len(x), np.full(S_cand.shape[0], L), hop_size, n_fft)
        mse, sc, pj = np.mean((x - x_rec_cand) ** 2), spectral_convergence(x, x_rec_cand), phase_jitter(x, x_rec_cand)
        kurt = envelope_spectrum_kurtosis(S_cand)
        cand_metrics.append((mse, sc, kurt, pj))

    u_renyi_2d_run = running_2d_renyi(S_univ)



    # Generate the table
    table_data = [
        ["Universal STFT", f"{univ_mse:.2e}", f"{univ_sc:.2e}", f"{u_kurt_2d:.2f}", f"{univ_pj:.2e}"],
        [f"Causal STFT (L={best_L})", f"{stft_mse:.2e}", f"{stft_sc:.2e}", "-", f"{stft_pj:.2e}"],
    ]
    for c, L in enumerate(cand_lengths):
        table_data.append([f"Fixed L={int(L)}", f"{cand_metrics[c][0]:.2e}", f"{cand_metrics[c][1]:.2e}",
                           f"{cand_metrics[c][2]:.2f}", f"{cand_metrics[c][3]:.2e}"])

    # 6. Plot-only exports: same computed data, all candidates in fixed order.
    # Prefix 2-D Rényi Entropy remains a single raw-prefix calculation.
    from benchmarks.figure_export import figure_edition
    from benchmarks.plot_suites import evaluation_figures
    safe_title = _safe_name(title)
    with figure_edition(out_dir, f"{title}; unchanged run_full_evaluation data") as edition:
        fig_dir = edition.root / "data" / safe_title
        fig_dir.mkdir(parents=True, exist_ok=True)
        with (fig_dir / f"{safe_title}_metrics.csv").open("w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["Model", "MSE", "Spectral Conv", "Env Kurtosis", "Phase Jitter"])
            writer.writerows(table_data)
        evaluation_figures(
            title, fs, hop_size, S_univ, np.asarray(pmf_history), cand_lengths,
            all_cand_specs, cand_2d_renyi_over_time, u_renyi_2d_run,
            cand_metrics, (univ_mse, univ_sc, u_kurt_2d, univ_pj),
            ((cand_ent, blend_ent, "Total"),
             (cand_h_time, blend_h_time, "Time"),
             (cand_h_spec, blend_h_spec, "Frequency")),
            n_frames, edition.root,
        )
    print(f"Finished. Paired figures and original metrics: {edition.root}")
