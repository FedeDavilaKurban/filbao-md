import os
import shutil
import numpy as np
import re
from multidark_data_loader import (load_catalog, select_sample, split_by_variable,
                         plot_magnitude_hist, plot_bin_data,
                         compute_knn_distance)
from multidark_correlation import (compute_xi_s_mu, compute_xi_s, compute_monopole_from_xi_s_mu,
                         get_paircounts_filename)
from multidark_plotter import (plot_xi_s_mu, plot_xi_s_combined, plot_monopoles_combined,
                     plot_monopole_to_xi_s_ratio, plot_xi_sigma_pi_from_xi_s_mu)

# ---------------------------
# CONFIGURATION
# ---------------------------
L = 1000.0
mag_max = -21.2
test_dilute = 1.0
force_recompute_full = False
force_recompute_bin = False

# 2D correlation parameters
min_sep_2d = 1.0
max_sep_2d = 150.0
bin_size_2d = 2.0
pi_rebin = bin_size_2d

# Split configuration
split_vars = [
    {
        'col': 'dist_fil',
        'mode': 'custom_intervals',
        'custom_intervals': [(0, 4), (8, 100)]
    },
    # Uncomment to also split by Sigma_3:
    {
        'col': 'Sigma_3',
        'mode': 'percentile_intervals',
        'percentile_intervals': [(0, 30), (50, 100)]
    }
]

# Output folders
folderName = f'XISMU_box_mag{mag_max:.1f}'
output_folder = f"../plots/{folderName}/"
paircounts_dir = "../data/pair_counts/box"
monopoles_dir = "../data/monopoles/box"

# Create folders
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(paircounts_dir, exist_ok=True)
os.makedirs(monopoles_dir, exist_ok=True)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def clean_label_for_filename(label):
    """Convert LaTeX label to a filesystem-friendly string."""
    # Remove LaTeX math delimiters
    label = label.replace('$', '')
    # Replace common operators
    label = label.replace('\\leq', 'le')
    label = label.replace('\\in', 'in')
    label = label.replace('\\', '')
    # Remove spaces and special characters
    label = re.sub(r'[^\w\.,\-]', '_', label)
    # Collapse multiple underscores
    label = re.sub(r'_+', '_', label)
    # Remove leading/trailing underscores
    label = label.strip('_')
    return label

def make_bin_desc_from_label(label, var_names=None):
    """Create a concise bin description from a label."""
    cleaned = clean_label_for_filename(label)
    # Optionally shorten by taking first few characters? Not needed.
    return cleaned

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    # Load and select data
    cat_full = load_catalog()
    cat = select_sample(cat_full, mag_max, test_dilute=test_dilute)
    print(f"Selected {len(cat)} galaxies.")

    # Compute Sigma_3 if needed
    if split_vars:
        need_sigma3 = any(v['col'] == 'Sigma_3' for v in split_vars)
        if need_sigma3 and 'Sigma_3' not in cat.columns:
            print("Computing Sigma_3 (distance to 3rd nearest neighbor)...")
            coords = cat[['x', 'y', 'z']].values
            cat['Sigma_3'] = compute_knn_distance(coords, k=3, boxsize=L)
            print("Done.")

    # Split into bins
    if split_vars:
        bins, labels, bin_metadata = split_by_variable(cat, split_vars)
    else:
        bins = [cat]
        labels = ['Full Sample']
        bin_metadata = [None]

    print(f"Split into {len(bins)} bins:")
    for i, (b, lab) in enumerate(zip(bins, labels)):
        print(f"  Bin {i}: {len(b)} galaxies, {lab}")

    # Diagnostic plots
    plot_magnitude_hist(cat_full, mag_max, output_folder)
    plot_bin_data(cat, "Full Sample", output_folder, filename="bin_full_data_randoms.png")
    for i, (gxs, lab) in enumerate(zip(bins, labels)):
        # Use a cleaned label for filename
        clean_lab = clean_label_for_filename(lab)
        plot_bin_data(gxs, lab, output_folder, filename=f"bin_{clean_lab}_data_randoms.png")

    # Base parameters for paircounts filenames
    base_params = {
        'mag_max': mag_max,
        'min_sep': min_sep_2d,
        'max_sep': max_sep_2d,
        'bin_size': bin_size_2d,
        'split_vars': split_vars,   # includes split info
    }

    # ------------------------------------------------------------------
    # 1. Full sample
    # ------------------------------------------------------------------
    params_full = base_params.copy()
    params_full['dist_bin_mode'] = 'full'  # keep for compatibility
    paircounts_file_full = get_paircounts_filename("full", params_full, paircounts_dir)
    xi_full, s_bins_full, mu_bins_full = compute_xi_s_mu(
        cat["x"].values, cat["y"].values, cat["z"].values,
        min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
        paircounts_file=paircounts_file_full, force_recompute=force_recompute_full
    )

    xi0_full = compute_monopole_from_xi_s_mu(xi_full, mu_bins_full)
    s_centers_full = 0.5 * (s_bins_full[:-1] + s_bins_full[1:])
    monopoles_list = [(s_centers_full, xi0_full)]
    labels_list = ['Full Sample']

    # xi(s) for full sample
    xi_s_full, s_bins_xi_s_full = compute_xi_s(
        cat["x"].values, cat["y"].values, cat["z"].values,
        min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
        paircounts_file=get_paircounts_filename("full_xi_s", params_full, paircounts_dir),
        force_recompute=force_recompute_full
    )
    s_centers_xi_s_full = 0.5 * (s_bins_xi_s_full[:-1] + s_bins_xi_s_full[1:])
    xi_s_list = [(s_centers_xi_s_full, xi_s_full)]

    # Save full sample results
    np.savez(os.path.join(monopoles_dir, f"monopole_full_{params_full['dist_bin_mode']}.npz"),
             s=s_centers_full, xi0=xi0_full)
    np.savez(os.path.join(paircounts_dir, f"xi_s_full_{params_full['dist_bin_mode']}.npz"),
             s=s_centers_xi_s_full, xi_s=xi_s_full)

    # ξ(σ, π) for full sample
    sigma_bins = np.linspace(min_sep_2d, max_sep_2d, int((max_sep_2d - min_sep_2d) / bin_size_2d) + 1)
    plot_xi_sigma_pi_from_xi_s_mu(
        xi_full, s_bins_full, mu_bins_full, sigma_bins, pi_rebin,
        title=r"$\xi(\sigma,\pi)$ Full Sample", output_folder=output_folder,
        plotname="xi_sigma_pi_full.png", min_sep=min_sep_2d
    )

    # ------------------------------------------------------------------
    # 2. Distance bins
    # ------------------------------------------------------------------
    bin_results = []   # each: (xi, s_bins, mu_bins, label, bin_desc)
    for i, (gxs, lab, meta) in enumerate(zip(bins, labels, bin_metadata)):
        # Create a descriptive bin name from the label (clean)
        bin_desc = make_bin_desc_from_label(lab)
        # For safety, if the cleaned label is empty or too long, fallback to i
        if not bin_desc:
            bin_desc = f"bin_{i}"
        print(f"Processing bin: {lab} -> {bin_desc}")

        # 2D correlation function
        params_bin = base_params.copy()
        paircounts_file = get_paircounts_filename(bin_desc, params_bin, paircounts_dir)
        xi_bin, s_bins_bin, mu_bins_bin = compute_xi_s_mu(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
            paircounts_file=paircounts_file, force_recompute=force_recompute_bin,
            dfil_bin_metadata=meta
        )
        bin_results.append((xi_bin, s_bins_bin, mu_bins_bin, lab, bin_desc))

        # Monopole
        xi0_bin = compute_monopole_from_xi_s_mu(xi_bin, mu_bins_bin)
        s_centers_bin = 0.5 * (s_bins_bin[:-1] + s_bins_bin[1:])
        monopoles_list.append((s_centers_bin, xi0_bin))
        labels_list.append(lab)

        # Save monopole
        np.savez(os.path.join(monopoles_dir, f"monopole_{bin_desc}_{base_params['split_vars'][0]['mode']}.npz"),
                 s=s_centers_bin, xi0=xi0_bin, **meta if meta else {})

        # 1D correlation
        paircounts_xi_s = get_paircounts_filename(f"{bin_desc}_xi_s", params_bin, paircounts_dir)
        xi_s_bin, s_bins_xi_s_bin = compute_xi_s(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
            paircounts_file=paircounts_xi_s, force_recompute=force_recompute_bin,
            dfil_bin_metadata=meta
        )
        s_centers_xi_s_bin = 0.5 * (s_bins_xi_s_bin[:-1] + s_bins_xi_s_bin[1:])
        xi_s_list.append((s_centers_xi_s_bin, xi_s_bin))

        # ξ(σ,π) plot
        plot_xi_sigma_pi_from_xi_s_mu(
            xi_bin, s_bins_bin, mu_bins_bin, sigma_bins, pi_rebin,
            title=rf"$\xi(\sigma,\pi)$ {lab}", output_folder=output_folder,
            plotname=f"xi_sigma_pi_{bin_desc}.png", min_sep=min_sep_2d
        )

    # ------------------------------------------------------------------
    # Global color limits for ξ(s, μ) plots
    # ------------------------------------------------------------------
    all_xi = [xi_full] + [xi for (xi, _, _, _, _) in bin_results]
    all_xi_flat = np.concatenate([xi.ravel() for xi in all_xi])
    vmin_global = np.percentile(all_xi_flat, 1)
    vmax_global = np.percentile(all_xi_flat, 99)
    if vmin_global >= 0:
        vmin_global = -vmax_global / 2
    if vmax_global <= 0:
        vmax_global = -vmin_global / 2
    contour_levels = np.concatenate(([-0.01, -0.005, -0.001], np.logspace(-3, 0, 10)))
    print(f"Global color limits: vmin={vmin_global:.3f}, vmax={vmax_global:.3f}")

    # Plot ξ(s, μ) for full sample
    plot_xi_s_mu(xi_full, s_bins_full, mu_bins_full,
                 title=r"$\xi(s,\mu)$ Full Sample", output_folder=output_folder,
                 plotname="xi_s_mu_full.png", min_sep=min_sep_2d,
                 vmin_global=vmin_global, vmax_global=vmax_global,
                 contour_levels=contour_levels, contour_colors='black')

    # Plot ξ(s, μ) for each distance bin
    for xi, s_bins, mu_bins, lab, bin_desc in bin_results:
        plot_xi_s_mu(xi, s_bins, mu_bins,
                     title=rf"$\xi(s,\mu)$ {lab}", output_folder=output_folder,
                     plotname=f"xi_s_mu_{bin_desc}.png", min_sep=min_sep_2d,
                     vmin_global=vmin_global, vmax_global=vmax_global,
                     contour_levels=contour_levels, contour_colors='black')

    # ------------------------------------------------------------------
    # Combined plots
    # ------------------------------------------------------------------
    plot_monopoles_combined(monopoles_list, labels_list, output_folder, filename='xi0_combined.png')
    plot_xi_s_combined(xi_s_list, labels_list, output_folder, filename='xi_s_combined.png')
    plot_monopole_to_xi_s_ratio(monopoles_list, xi_s_list, labels_list, output_folder)

    print("Analysis complete.")

if __name__ == "__main__":
    main()