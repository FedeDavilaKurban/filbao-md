"""
FilBAO, plan steps 1-2: make the subsamples and measure xi_0(s) and P(k) for each.

    1. Load the MultiDark mock, apply the magnitude cut, compute rho_3 if needed.
    2. Subsamples: split by `split_vars` (one or two variables), plus random dilutions of the
       full sample with the same number of galaxies as a bin (number-density controls).
    3. For the full sample and every subsample: DD(s, mu) -> xi(s, mu) -> monopole xi_0(s),
       saved to data/processed/monopoles/box/.
    4. Auto- and cross-power spectra of all of them in one file, data/processed/pk/box/
       (folded at high k), used by src/bao_fitting.py for the covariance of the fit; the
       error bands of the plots use that same covariance.
    5. Optional diagnostic plots (make_plots), in figures/diagnostics/<run name>/.

One run per split: split by dist_fil alone -> Full, Fil, NonFil (+ their dilution controls);
split by dist_fil and log_rho_3 -> the four filament x density cells.

Run:  python scripts/multidark_analysis.py   (from any folder)
"""
import os
import re
import sys
import shutil
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))

from multidark_data_loader import load_catalog, select_sample, split_by_variable, compute_knn_distance
from multidark_correlation import (get_save_filename, compute_xi_s_mu,
                                   compute_monopole_from_xi_s_mu, compute_xi_s)
from multidark_power import measure_spectra
import bao_fitting as bf
from multidark_plotter import (plot_magnitude_hist, plot_bin_data, plot_xi_s_mu,
                               plot_monopoles_combined, plot_xi_s_combined,
                               plot_monopole_to_xi_s_ratio)

# ---------------------------
# CONFIGURATION
# ---------------------------
L = 1000.0                        # box side, h^-1 Mpc
mag_max = -21.2                   # absolute-magnitude cut

# Correlation function binning
min_sep_2d = 5.0
max_sep_2d = 200.0
bin_size_2d = 2.5

# Pair counts are reused if their file exists; set True to recompute them
force_recompute_full = False
force_recompute_bin = False

# Split configuration (one or two variables; two -> all combinations)
split_vars = [
    {
        'col': 'dist_fil',
        'mode': 'custom_intervals',
        'custom_intervals': [(0, 3), (10, 100)]
    },
    # Uncomment to also split by rho_3 (the four filament x density cells):
    # {
    #     'col': 'log_rho_3',
    #     'mode': 'percentile_intervals',
    #     'percentile_intervals': [(0, 84), (84, 100)]
    # }
]

# Random dilutions of the full sample (number-density controls)
# For each bin chosen below, add a random subsample of the FULL sample with exactly the same
# number of galaxies: Full's clustering and BAO with a lower nbar.  Comparing a bin with its
# control separates 'this environment' from 'fewer galaxies'.  Treated as one more bin.
#   'all' : one control per bin      [0, 1] : only for these bins (index in the split)      [] : none
# The seed is part of the file name, so changing it never reuses old pair counts.
dilution_controls = 'all'
n_dilution_realisations = 1       # > 1: independent controls (seed, seed+1, ...)
dilution_seed = 20261016

# Power spectra (for the covariance of the BAO fit)
compute_spectra = True            # auto- and cross-spectra of Full + all bins (~1 min per fold level, ~1-2 GB)
pk_ngrid = 384                    # FFT grid per side (k_Nyquist = 1.2 h/Mpc for L = 1000)
pk_n_fold = 2                     # foldings of the box (each doubles the k reach at the same memory)
pk_kuse_frac = 0.6                # each fold level is used up to this fraction of its Nyquist frequency
                                  # (384, 2, 0.6 -> measured to k = 2.9 h/Mpc)

# Diagnostic plots
make_plots = True                 # magnitude histogram, sample maps, xi(s, mu), xi_0, xi(s), ratio
plot_errors = True                # error bands on xi_0 and xi(s): the covariance of the BAO fit
                                  # (bao_fitting.gaussian_cov_pk), from the measured spectra

# ---------------------------
# PATHS
# ---------------------------
RAW_DIR = os.path.join(REPO, 'data', 'raw')
PROC_DIR = os.path.join(REPO, 'data', 'processed')
paircounts_dir = os.path.join(PROC_DIR, 'pair_counts', 'box')
monopoles_dir = os.path.join(PROC_DIR, 'monopoles', 'box')
xis_dir = os.path.join(PROC_DIR, 'xis', 'box')
spectra_dir = os.path.join(PROC_DIR, 'pk', 'box')

split_cols = "".join(re.sub(r'[^\w]', '', v['col']) for v in split_vars)
run_name = f"XISMU_box_mag{mag_max:.1f}_{split_cols}"
plots_dir = os.path.join(REPO, 'figures', 'diagnostics', run_name)


# ---------------------------
# HELPERS
# ---------------------------
def make_bin_desc_from_label(label):
    """
    Bin label -> the <bin_desc> part of the file names.  The fitting notebook uses the same
    string to find each sample's spectrum in the spectra file.
    """
    label = label.replace('d_{fil}', 'dfil')
    label = label.replace('log_{10} rho_3', 'log10rho3')
    for ch in ('$', '\\', '{', '}', '(', ')'):
        label = label.replace(ch, '')
    label = label.replace(',', '_')        # '_' between the two variables ('-' would eat a minus sign)
    label = label.replace(' ', '')
    label = label.replace('<', '-')
    label = label.replace('>', '-')
    label = label.replace('=', '-')
    label = re.sub(r'[^\w\-_\.]', '_', label)     # only letters, numbers, -, _, .
    label = re.sub(r'_+', '_', label)
    label = re.sub(r'-+', '-', label)
    return label


def make_samples(cat):
    """[(label, bin_desc, DataFrame, force_recompute)], the full sample first."""
    if split_vars:
        bins, labels = split_by_variable(cat, split_vars)
    else:
        bins, labels = [], []

    # Random dilutions of the full sample, appended as extra bins
    if dilution_controls:
        targets = range(len(bins)) if dilution_controls == 'all' else list(dilution_controls)
        new_bins, new_labels = [], []
        for t in targets:
            n_target = len(bins[t])
            for r in range(n_dilution_realisations):
                seed = dilution_seed + r
                rng = np.random.default_rng(seed)
                idx = np.sort(rng.choice(len(cat), size=n_target, replace=False))
                new_bins.append(cat.iloc[idx])
                new_labels.append(f"Full random N={n_target} r{seed}")
                print(f"Control: random {n_target} of {len(cat)} Full galaxies "
                      f"(matches '{labels[t]}', seed {seed})")
        bins = list(bins) + new_bins
        labels = list(labels) + new_labels

    samples = [('Full Sample', 'full', cat, force_recompute_full)]
    samples += [(lab, make_bin_desc_from_label(lab), g, force_recompute_bin)
                for g, lab in zip(bins, labels)]
    return samples


def measure_monopole(gxs, bin_desc, force):
    """xi(s, mu) and its monopole for one sample.  The pair counts are reused if saved
    (unless force); the monopole file is written if missing or if recomputed."""
    params = dict(mag_max=mag_max, min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d)
    pc_file = get_save_filename(bin_desc, params, paircounts_dir, filetype='paircounts')
    mono_file = get_save_filename(bin_desc, params, monopoles_dir, filetype='monopole')

    xi, s_bins, mu_bins = compute_xi_s_mu(
        gxs["x"].values, gxs["y"].values, gxs["z"].values,
        min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
        paircounts_filename=pc_file, force_recompute=force)
    s = 0.5 * (s_bins[:-1] + s_bins[1:])
    xi0 = compute_monopole_from_xi_s_mu(xi)

    if force or not os.path.exists(mono_file):
        np.savez(mono_file, s=s, xi0=xi0, n_gal=len(gxs))
        print(f"Saved monopole to {mono_file}")
    return dict(xi_smu=xi, s_bins=s_bins, mu_bins=mu_bins, s=s, xi0=xi0)


# ---------------------------
# DIAGNOSTIC PLOTS
# ---------------------------
def make_diagnostic_plots(cat_full, samples, results, pk_file, pk_labels):
    """All plots of the run, in plots_dir (emptied first)."""
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    subsample_cols = [v['col'] for v in split_vars] if split_vars else None

    # Magnitude cut and the samples (map + histograms of the split variables)
    plot_magnitude_hist(cat_full, mag_max, plots_dir)
    plot_bin_data(samples[0][2], "Full Sample", plots_dir, filename="bin_full_data_randoms.png",
                  subsample_cols=subsample_cols)
    for lab, desc, gxs, _ in samples[1:]:
        plot_bin_data(gxs, lab, plots_dir, filename=f"bin_{desc}_data_randoms.png",
                      subsample_cols=subsample_cols)

    # 1D xi(s), computed directly from DD(s) (checks the monopole: ratio should be 1)
    params = dict(mag_max=mag_max, min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d)
    xi_s_list = []
    for lab, desc, gxs, force in samples:
        f = get_save_filename(desc, params, xis_dir, filetype='xi_s')
        xi_s, s_c = compute_xi_s(gxs["x"].values, gxs["y"].values, gxs["z"].values,
                                 min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
                                 paircounts_filename=f, force_recompute=force)
        np.savez(f, s=s_c, xi_s=xi_s)
        xi_s_list.append((s_c, xi_s))

    # xi(s, mu) of every sample, with a common colour scale
    all_xi_flat = np.concatenate([r['xi_smu'].ravel() for r in results])
    vmin_global = np.percentile(all_xi_flat, 1)
    vmax_global = np.percentile(all_xi_flat, 99)
    if vmin_global >= 0:
        vmin_global = -vmax_global / 2
    if vmax_global <= 0:
        vmax_global = -vmin_global / 2
    contour_levels = np.concatenate(([-0.01, -0.005, -0.001], np.logspace(-3, 0, 10)))
    print(f"Global color limits: vmin={vmin_global:.3f}, vmax={vmax_global:.3f}")
    for (lab, desc, _, _), r in zip(samples, results):
        title = r"$\xi(s,\mu)$ Full Sample" if desc == 'full' else rf"$\xi(s,\mu)$ {lab}"
        plot_xi_s_mu(r['xi_smu'], r['s_bins'], r['mu_bins'], title=title,
                     output_folder=plots_dir, plotname=f"xi_s_mu_{desc}.png", min_sep=min_sep_2d,
                     vmin_global=vmin_global, vmax_global=vmax_global,
                     contour_levels=contour_levels, contour_colors='black')

    # Error bands: the same covariance as the BAO fit (bao_fitting.gaussian_cov_pk, with each
    # sample's bias from estimate_bias_independent, as in analyse_sample), joint over all
    # samples so it also contains the cross-covariances between samples; saved for later use
    errors = None
    if plot_errors:
        if os.path.exists(pk_file):
            s_edges = bf.edges_from_centres(results[0]['s'])
            nb_s = len(s_edges) - 1
            lin = bf.linear_theory()
            T = bf.BAOTemplate(*bf.generate_templates(lin=lin), lin)
            spec = bf.load_spectra(pk_file)
            biases = [bf.estimate_bias_independent(r['s'], r['xi0'], s_edges, len(g) / L**3, T, L)['b']
                      for (_, _, g, _), r in zip(samples, results)]
            C_xi = bf.gaussian_cov_pk(s_edges, spec, [spec['labels'].index(l) for l in pk_labels],
                                      biases, lin, L)
            cov_file = os.path.join(spectra_dir, f"cov_xi0_{run_name}_binsep{bin_size_2d:.1f}.npz")
            np.savez(cov_file, cov=C_xi, labels=np.array(pk_labels), s_edges=s_edges, biases=biases,
                     note="joint Gaussian covariance of the xi monopoles (bao_fitting.gaussian_cov_pk), "
                          "blocks ordered as labels")
            errors = [np.sqrt(np.diag(C_xi)[i*nb_s:(i+1)*nb_s]) for i in range(len(pk_labels))]
            print(f"Covariance of the fit (gaussian_cov_pk), b = "
                  + ", ".join(f"{b:.2f}" for b in biases) + f" -> {cov_file}")
        else:
            print(f"WARNING: {pk_file} not found (set compute_spectra = True once); no error bands")

    # Combined xi_0, xi(s) and their ratio
    monopoles_list = [(r['s'], r['xi0']) for r in results]
    labels_list = [lab for lab, _, _, _ in samples]
    plot_monopoles_combined(monopoles_list, labels_list, plots_dir,
                            filename='xi0_combined.png', errors=errors)
    plot_xi_s_combined(xi_s_list, labels_list, plots_dir,
                       filename='xi_s_combined.png', errors=errors)
    plot_monopole_to_xi_s_ratio(monopoles_list, xi_s_list, labels_list, plots_dir)
    print(f"Plots in {plots_dir}")


# ---------------------------
# MAIN
# ---------------------------
def main():
    for d in (paircounts_dir, monopoles_dir, spectra_dir) + ((xis_dir,) if make_plots else ()):
        os.makedirs(d, exist_ok=True)

    # 1. Catalogue
    cat_full = load_catalog(os.path.join(RAW_DIR, 'to_mock.csv'),
                            os.path.join(RAW_DIR, 'mock_withfilament.csv'))
    cat = select_sample(cat_full, mag_max)
    print(f"Selected {len(cat)} galaxies.")
    if any(v['col'] == 'log_rho_3' for v in split_vars) and 'log_rho_3' not in cat.columns:
        print("Computing rho_3 (distance to 3rd nearest neighbor)...")
        nn_dist = compute_knn_distance(cat[['x', 'y', 'z']].values, k=3, boxsize=L)
        cat['log_rho_3'] = np.log10(3 / ((4 * np.pi * nn_dist**3) / 3))

    # 2. Subsamples
    samples = make_samples(cat)
    print(f"Split into {len(samples) - 1} bins:")
    for i, (lab, _, g, _) in enumerate(samples[1:]):
        print(f"  Bin {i}: {len(g)} galaxies, {lab}")

    # 3. Power spectra of all samples (labels = <bin_desc>, as in the monopole file names)
    pk_labels = [desc for _, desc, _, _ in samples]
    pk_file = os.path.join(spectra_dir, f"pk_{run_name}.npz")
    if compute_spectra:
        print(f"\nMeasuring auto/cross power spectra of {pk_labels} ...")
        measure_spectra([g[['x', 'y', 'z']].values for _, _, g, _ in samples],
                        pk_labels, L, ng=pk_ngrid, n_fold=pk_n_fold, kuse_frac=pk_kuse_frac, save=pk_file)

    # 4. Monopoles
    results = []
    for lab, desc, gxs, force in samples:
        print(f"Processing: {lab} -> {desc}")
        results.append(measure_monopole(gxs, desc, force))

    # 5. Diagnostic plots
    if make_plots:
        make_diagnostic_plots(cat_full, samples, results, pk_file, pk_labels)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
