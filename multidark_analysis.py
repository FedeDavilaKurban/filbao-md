import os
import shutil
import numpy as np
import re
from multidark_data_loader import (load_catalog, select_sample, split_by_variable,
                         plot_magnitude_hist, plot_bin_data,
                         compute_knn_distance)
from multidark_correlation import (compute_xi_s_mu, compute_xi_s, compute_monopole_from_xi_s_mu,
                         get_save_filename)
from multidark_plotter import (plot_xi_s_mu, plot_xi_s_combined, plot_monopoles_combined,
                     plot_monopole_to_xi_s_ratio, plot_xi_sigma_pi_from_xi_s_mu)

# ---------------------------
# CONFIGURATION
# ---------------------------
L = 1000.0
mag_max = -21.2
test_dilute = 1.
force_recompute_full = False
force_recompute_bin = True
interpolate_to_xirppi = False

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
        'custom_intervals': [(0, 3), (5, 10), (10, 100)]
    },
    # Uncomment to also split by rho_3:
    # {
    #     'col': 'log_rho_3',
    #     'mode': 'percentile_intervals',
    #     'percentile_intervals': [(0, 84), (84, 100)]
    # }
]

# Output folders
# Collect split column names without any underscores
split_cols = "".join(
    re.sub(r'[^\w]', '', v['col'])  # remove anything not a letter or number
    for v in split_vars
)

folderName = f"XISMU_box_mag{mag_max:.1f}_{split_cols}"
if test_dilute < 1.0:
    folderName += f"_dilute{test_dilute}"
output_folder = f"../plots/{folderName}/"
paircounts_dir = "../data/pair_counts/box"
monopoles_dir = "../data/monopoles/box"
xis_dir = "../data/xis/box"

# Create folders
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(paircounts_dir, exist_ok=True)
os.makedirs(monopoles_dir, exist_ok=True)
os.makedirs(xis_dir, exist_ok=True)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
# def clean_label_for_filename(label):
#     """Convert LaTeX label to a filesystem-friendly string."""
#     # Remove LaTeX math delimiters
#     label = label.replace('$', '')
#     # Replace common operators
#     label = label.replace('\\leq', 'le')
#     label = label.replace('\\in', 'in')
#     label = label.replace('\\', '')
#     # Remove spaces and special characters
#     label = re.sub(r'[^\w\.,\-]', '_', label)
#     # Collapse multiple underscores
#     label = re.sub(r'_+', '_', label)
#     # Remove leading/trailing underscores
#     label = label.strip('_')
#     return label

def make_bin_desc_from_label(label, dilute=None):
    """
    Convert a bin label into a concise, safe filename string.
    """
    # Replace variable names
    label = label.replace('d_{fil}', 'dfil')
    label = label.replace('log_{10} rho_3', 'log10rho3')
    
    # Remove LaTeX symbols and spaces
    label = label.replace('$', '')
    label = label.replace('\\', '')
    label = label.replace('{', '')
    label = label.replace('}', '')
    label = label.replace('(', '')
    label = label.replace(')', '')
    label = label.replace(',', '-')
    label = label.replace(' ', '')
    label = label.replace('<', '-')
    label = label.replace('<=', '-')
    label = label.replace('>', '-')
    label = label.replace('=', '-')

    # Remove any remaining illegal filename characters
    label = re.sub(r'[^\w\-_\.]', '_', label)  # only allow letters, numbers, -, _, .
    
    # Collapse multiple underscores/hyphens
    label = re.sub(r'_+', '_', label)
    label = re.sub(r'-+', '-', label)
    
    # Optional: add dilute parameter
    if dilute is not None and dilute < 1.0:
        label = f"dilute{dilute}_{label}"
    else:
        label = f"{label}"
    
    return label

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    # Load and select data
    cat_full = load_catalog()
    cat = select_sample(cat_full, mag_max, test_dilute=test_dilute)
    print(f"Selected {len(cat)} galaxies.")

    # Compute rho_3 if needed
    if split_vars:
        need_rho3 = any(v['col'] == 'log_rho_3' for v in split_vars)
        if need_rho3 and 'log_rho_3' not in cat.columns:
            print("Computing rho_3 (distance to 3rd nearest neighbor)...")
            coords = cat[['x', 'y', 'z']].values
            nn_dist = compute_knn_distance(coords, k=3, boxsize=L)
            rho3 = 3 / ((4 * np.pi * nn_dist**3) / 3)
            cat['log_rho_3'] = np.log10(rho3)
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
    subsample_cols = [v['col'] for v in split_vars] if split_vars else None
    plot_magnitude_hist(cat_full, mag_max, output_folder)
    plot_bin_data(cat, "Full Sample", output_folder, filename="bin_full_data_randoms.png",\
                   subsample_cols=subsample_cols)
    for i, (gxs, lab) in enumerate(zip(bins, labels)):
        # Use bin description for filename
        bin_desc = make_bin_desc_from_label(lab, dilute=test_dilute)
        plot_bin_data(gxs, lab, output_folder,
                    filename=f"bin_{bin_desc}_data_randoms.png",
                    subsample_cols=subsample_cols)

    # Base parameters for paircounts filenames
    base_params = {
        'mag_max': mag_max,
        'min_sep': min_sep_2d,
        'max_sep': max_sep_2d,
        'bin_size': bin_size_2d,
        'split_vars': split_vars,   # includes split info
    }

    # ----------------------------
    # 1. Full sample
    # ----------------------------
    params_full = base_params.copy()

    # Paircounts filename
    xismu_paircounts_filename_full = get_save_filename("full", params_full, paircounts_dir, filetype='paircounts')

    # xi_s_mu
    xi_full, s_bins_full, mu_bins_full = compute_xi_s_mu(
        cat["x"].values, cat["y"].values, cat["z"].values,
        min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
        paircounts_filename=xismu_paircounts_filename_full, force_recompute=force_recompute_full
    )

    # Monopole
    xi0_full = compute_monopole_from_xi_s_mu(xi_full, mu_bins_full)
    s_centers_full = 0.5 * (s_bins_full[:-1] + s_bins_full[1:])
    monopoles_list = [(s_centers_full, xi0_full)]
    labels_list = ['Full Sample']

    monopole_filename_full = get_save_filename("full", params_full, monopoles_dir, filetype='monopole')
    np.savez(monopole_filename_full, s=s_centers_full, xi0=xi0_full)

    # xi_s output filename
    xis_filename_full = get_save_filename("full", params_full, xis_dir, filetype='xi_s')

    # 1D correlation using paircounts file
    xi_s_full, s_centers_xi_s_full = compute_xi_s(
        cat["x"].values, cat["y"].values, cat["z"].values,
        min_sep_2d, max_sep_2d, bin_size_2d,
        boxsize=L,
        paircounts_filename=xis_filename_full,  # separate file from xi_s_mu paircounts
        force_recompute=force_recompute_full
    )
    xi_s_list = [(s_centers_xi_s_full, xi_s_full)]   # <-- keep this for combined plots
    np.savez(xis_filename_full, s=s_centers_xi_s_full, xi_s=xi_s_full)


    # ----------------------------
    # 2. Subsamples
    # ----------------------------
    bin_results = []  # keep for xi(s, mu) plots
    for i, (gxs, lab, meta) in enumerate(zip(bins, labels, bin_metadata)):
        bin_desc = make_bin_desc_from_label(lab, dilute=test_dilute)
        print(f"Processing bin: {lab} -> {bin_desc}")

        params_bin = base_params.copy()

        # Paircounts filename
        xismu_paircounts_filename_bin = get_save_filename(bin_desc, params_bin, paircounts_dir, filetype='paircounts')

        # xi_s_mu
        xi_bin, s_bins_bin, mu_bins_bin = compute_xi_s_mu(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
            paircounts_filename=xismu_paircounts_filename_bin, force_recompute=force_recompute_bin,
            dfil_bin_metadata=meta
        )

        # Store for xi(s, mu) plots
        bin_results.append((xi_bin, s_bins_bin, mu_bins_bin, lab, bin_desc))

        # Monopole
        xi0_bin = compute_monopole_from_xi_s_mu(xi_bin, mu_bins_bin)
        s_centers_bin = 0.5 * (s_bins_bin[:-1] + s_bins_bin[1:])
        monopoles_list.append((s_centers_bin, xi0_bin))
        labels_list.append(lab)

        monopole_filename_bin = get_save_filename(bin_desc, params_bin, monopoles_dir, filetype='monopole')
        np.savez(monopole_filename_bin, s=s_centers_bin, xi0=xi0_bin)

        # xi_s output filename
        xi_s_filename_bin = get_save_filename(bin_desc, params_bin, xis_dir, filetype='xi_s')

        # 1D correlation 
        xi_s_bin, s_centers_xi_s_bin = compute_xi_s(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            min_sep_2d, max_sep_2d, bin_size_2d, boxsize=L,
            paircounts_filename=xi_s_filename_bin,  
            force_recompute=force_recompute_bin,
            dfil_bin_metadata=meta
        )
        xi_s_list.append((s_centers_xi_s_bin, xi_s_bin)) # keep for combined plots
        np.savez(xi_s_filename_bin, s=s_centers_xi_s_bin, xi_s=xi_s_bin)

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