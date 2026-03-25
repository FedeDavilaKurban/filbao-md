"""
3D box version – Multidark simulation.
No lightcone, no RSD – pure real‑space periodic box.
"""

import os
import shutil
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from matplotlib.colors import LogNorm, SymLogNorm
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import gaussian_kde
from astropy.table import Table

# ---------------------------
# PARAMETERS
# ---------------------------

force_recompute_full = True
force_recompute_bin = True

# ---- Sample ----------
L = 1000.0                     # box size in Mpc/h (hardcoded)
h = 1                          # H0 = 100, so h = 1
mag_max = -21.5                 # absolute magnitude cut
ran_method = 'random_choice'    # for 3D box we can simply choose uniform random points
test_dilute = .5   

# ------ Weighting options ------
use_dec_weights = False         # no declination weights in 3D box

# ------ 2D correlation function parameters ------
min_sep_2d = 1.0                # minimum σ, π in Mpc/h
max_sep_2d = 150.0
bin_size_2d = 3.0
pi_rebin = 3                    # rebin factor for π direction

# ------ dist_fil binning ------
dist_bin_mode = "percentile_intervals"
dist_bin_percentile_intervals = [   # used only for "percentile_intervals" mode
    (0, 20),      # a–bth percentile
    (60, 80)      # c–dth percentile
]
nbins_dist = 4                   # used only for percentile / equal_width
dist_bin_edges = [0, 5, 10, 15, 100]  # used only for "fixed"

# ------ Random catalog parameters ------
nrand_mult = 10                  # Nr / Nd

# ------ Output folder --------
folderName = f'XISIGMAPI_3Dbox_mag{mag_max:.1f}_nrand{nrand_mult}'
output_folder = f"../plots/{folderName}/"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# ------ Pair counts folder ----
paircounts_dir = "../data/pair_counts/"
os.makedirs(paircounts_dir, exist_ok=True)

# ------ Monopoles folder ----
monopoles_dir = "../data/monopoles/3dbox"
os.makedirs(monopoles_dir, exist_ok=True)

# ---------------------------
# COSMOLOGY (not used but kept for compatibility)
# ---------------------------
cosmo = FlatLambdaCDM(H0=h * 100, Om0=0.31)

# ---------------------------
# GENERAL HELPER FUNCTIONS (unchanged)
# ---------------------------

def safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

def ensure_dir_exists(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def save_figure(fig, path: str, dpi: int = 300):
    ensure_dir_exists(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def get_paircounts_filename(bin_name, params):
    """
    Generate a filename for saved pair counts based on key parameters.
    params should be a dict containing at least:
        sample, mag_max, min_sep_2d, max_sep_2d, bin_size_2d, pi_rebin,
        nrand_mult, and the bin identifier (bin_name).
    """
    parts = [
        f"mag={params['mag_max']:.1f}",
        f"sep={params['min_sep_2d']}-{params['max_sep_2d']}",
        f"bin={params['bin_size_2d']}",
        f"pi_rebin={params['pi_rebin']}",
        f"nrand={params['nrand_mult']}",
        f"bin={bin_name}",
        f"distbinmode={params['dist_bin_mode']}"
    ]
    fname = "_".join(parts).replace('.', 'p')
    return os.path.join(paircounts_dir, fname + ".npz")

# ---------------------------
# REDSHIFT DISTRIBUTION FITTING (not used, but keep for compatibility)
# ---------------------------
def build_cdf_from_line(data, vmin, vmax, num_points=10000):
    # dummy return
    z_vals = np.linspace(vmin, vmax, num_points)
    cdf_vals = np.linspace(0, 1, num_points)
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, np.ones_like(z_vals), cdf_vals

def build_cdf_from_parabola(data, vmin, vmax, deg, num_points=10000):
    z_vals = np.linspace(vmin, vmax, num_points)
    cdf_vals = np.linspace(0, 1, num_points)
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, np.ones_like(z_vals), cdf_vals

def generate_random_red(redshift, nrand, ran_method, deg=None):
    # Not used – we generate uniform positions directly.
    return np.random.uniform(0, 1, nrand)

# ---------------------------
# WEIGHT COMPUTATION FUNCTIONS (not used, but keep)
# ---------------------------
def compute_dec_weights(data_dec, rand_dec, alpha=1.0, method="auto",
                        kde_threshold=1_000_000, nbins=40, spline_s=0.5,
                        bw_factor=1.2, n_grid=300):
    return np.ones_like(rand_dec)

# ---------------------------
# RA/DEC GENERATION FUNCTIONS (not used)
# ---------------------------
def generate_master_radec(full_cat, nrand_total, nside, ran_radec_method,
                          ra_preload=None, dec_preload=None):
    # Not used – we generate uniform positions directly.
    return np.random.uniform(0, 360, nrand_total), np.random.uniform(-90, 90, nrand_total)

# ---------------------------
# PLOTTING FUNCTIONS (adapted for 3D box)
# ---------------------------

def plot_redshift_k(cat):
    # Not used in 3D box – skip or create a magnitude histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cat["mag_abs_r"], bins=40, color="C00", alpha=0.8)
    ax.axvline(mag_max, color="k", linestyle=":")
    ax.set_xlabel("Absolute magnitude r")
    ax.set_ylabel("Count")
    filename = f"../plots/{folderName}/magnitude_hist.png"
    save_figure(fig, filename, dpi=100)

def plot_radec_distribution(cat, randoms, subsample=None):
    # For 3D box, we can plot the distribution of x and y (projection)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes[0].hist(randoms["x"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    axes[0].hist(cat["x"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
    axes[0].set_xlabel("x [Mpc/h]")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].hist(cat["y"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
    axes[1].hist(randoms["y"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    axes[1].set_xlabel("y [Mpc/h]")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    if subsample is None:
        filename = f"../plots/{folderName}/xy_distribution.png"
        axtitle = 'Total sample'
    else:
        filename = f"../plots/{folderName}/xy_distribution_bin{subsample}.png"
        axtitle = rf"r_{{fil}} \in {cat['dist_fil'].min():.1f}-{cat['dist_fil'].max():.1f} Mpc/h"
    fig.suptitle(axtitle)
    plt.tight_layout()
    save_figure(fig, filename, dpi=100)

def plot_bin_data_and_randoms(gxs, rxs, label, plotname):
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    
    # Top: x-y scatter
    ax_map = axes[0]
    ax_map.scatter(rxs["x"], rxs["y"], s=1.5, color="k", alpha=0.5, label="Randoms")
    ax_map.scatter(gxs["x"], gxs["y"], s=1, color="C00", label="Galaxies")
    ax_map.set_xlabel("x [Mpc/h]")
    ax_map.set_ylabel("y [Mpc/h]")
    ax_map.legend(loc='upper right')
    ax_map.set_title(label)
    ax_map.set_aspect('equal')
    
    # Bottom: filament distance histogram
    ax_hist_dist = axes[1]
    ax_hist_dist.hist(gxs["dist_fil"], bins=40, density=True, color="C03", alpha=0.9)
    ax_hist_dist.set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
    ax_hist_dist.set_ylabel("PDF")
    
    plt.tight_layout()
    save_figure(fig, plotname, dpi=200)

# ---------------------------
# CATALOG LOADING / MANIPULATION
# ---------------------------

def load_catalog():
    """
    Load and merge the two input files:
    - to_mock.csv: galaxy data (x, y, z, vx, vy, vz, magstarsdssr)
    - mock_withfilament.csv: filament distance (dfil or dist_fil)
    """
    df = pd.read_csv("../data/to_mock.csv", usecols=['x','y','z','vx','vy','vz','magstarsdssr'])
    df_fil = pd.read_csv("../data/mock_withfilament.csv")
    # Attach dist_fil
    if "dist_fil" in df_fil.columns:
        df["dist_fil"] = df_fil["dist_fil"].values
    elif "dfil" in df_fil.columns:
        df["dist_fil"] = df_fil["dfil"].values
    else:
        raise ValueError("No dist_fil/dfil column found in filament file")
    # Rename magnitude column for consistency
    df.rename(columns={'magstarsdssr': 'mag_abs_r'}, inplace=True)
    return df

def select_sample(cat):
    """Apply magnitude cut only (no redshift cut in 3D box)."""
    cat_mag = cat[cat["mag_abs_r"] < mag_max].copy()
    return cat_mag

def split_by_dist_fil_bins(cat_mag):
    values = cat_mag["dist_fil"].values

    if dist_bin_mode == "custom_intervals":
        # (unchanged) ...
        pass
    elif dist_bin_mode == "percentile_intervals":
        if not hasattr(dist_bin_percentile_intervals, '__iter__'):
            raise ValueError("dist_bin_percentile_intervals must be a list of (low, high) tuples")
        bins = []
        labels = []
        for (lo_pct, hi_pct) in dist_bin_percentile_intervals:
            lo_val = np.percentile(values, lo_pct)
            hi_val = np.percentile(values, hi_pct)
            mask = (values >= lo_val) & (values <= hi_val)
            subset = cat_mag.loc[mask].copy()
            bins.append(subset)
            labels.append(f"$r_{{fil}} \\in [{lo_val:.1f}-{hi_val:.1f}]$ Mpc/h")
        return bins, labels, None
    elif dist_bin_mode == "percentile":
        percentiles = np.linspace(0, 100, nbins_dist + 1)
        edges = np.percentile(values, percentiles)
    elif dist_bin_mode == "equal_width":
        vmin, vmax = values.min(), values.max()
        edges = np.linspace(vmin, vmax, nbins_dist + 1)
    elif dist_bin_mode == "fixed":
        edges = np.array(dist_bin_edges)
    else:
        raise ValueError(f"Unknown dist_bin_mode: {dist_bin_mode}")

    bins = []
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        bins.append(cat_mag.loc[mask].copy())
        labels.append(f"${lo:.1f} < r_{{fil}} \\leq {hi:.1f}$")
    return bins, labels, edges

# ---------------------------
# 2D CORRELATION FUNCTION ξ(σ, π) – CARTESIAN VERSION
# ---------------------------

def compute_xi_sigmapi_cartesian(x_data, y_data, z_data,
                                 x_rand, y_rand, z_rand,
                                 pi_rebin,
                                 data_weights=None, rand_weights=None,
                                 min_sep=0.0, max_sep=50.0, bin_size=2.0,
                                 paircounts_file=None, force_recompute=False,
                                 dfil_bin_metadata=None):
    """
    Compute ξ(σ, π) using Corrfunc for a periodic box.
    The line‑of‑sight direction is taken as the z‑axis.
    """
    import multiprocessing
    try:
        from Corrfunc.theory.DDrppi import DDrppi
    except ImportError:
        raise ImportError("Corrfunc is not installed. Please run: conda install -c conda-forge corrfunc")

    if data_weights is None:
        data_weights = np.ones(len(x_data))
    if rand_weights is None:
        rand_weights = np.ones(len(x_rand))

    # Try to load precomputed pair counts
    if paircounts_file and os.path.exists(paircounts_file) and not force_recompute:
        print(f"Loading precomputed pair counts from {paircounts_file}")
        data = np.load(paircounts_file)
        rp_bins = data['rp_bins']
        pi_rebin = data['pi_rebin']
        max_pimax = data['max_pimax']
        H_dd_rebinned = data['H_dd_rebinned']
        H_dr_rebinned = data['H_dr_rebinned']
        H_rr_rebinned = data['H_rr_rebinned']
        WD = data['WD']
        WR = data['WR']
        WD2 = data['WD2']
        WR2 = data['WR2']
    else:
        # Compute from scratch
        nbins = int((max_sep - min_sep) / bin_size)
        rp_bins = np.linspace(min_sep, max_sep, nbins + 1)
        pimax = int(max_sep)
        nthreads = min(multiprocessing.cpu_count(), 16)

        # Periodic box: set boxsize = L
        boxsize = L

        dd_counts = DDrppi(1, nthreads, pimax, rp_bins,
                           x_data, y_data, z_data,
                           weights1=data_weights,
                           weight_type='pair_product',
                           periodic=True, boxsize=boxsize, verbose=False)
        H_dd = (dd_counts['npairs'] * dd_counts['weightavg']).reshape(nbins, pimax)

        rr_counts = DDrppi(1, nthreads, pimax, rp_bins,
                           x_rand, y_rand, z_rand,
                           weights1=rand_weights,
                           weight_type='pair_product',
                           periodic=True, boxsize=boxsize, verbose=False)
        H_rr = (rr_counts['npairs'] * rr_counts['weightavg']).reshape(nbins, pimax)

        dr_counts = DDrppi(0, nthreads, pimax, rp_bins,
                           x_data, y_data, z_data,
                           X2=x_rand, Y2=y_rand, Z2=z_rand,
                           weights1=data_weights,
                           weights2=rand_weights,
                           weight_type='pair_product',
                           periodic=True, boxsize=boxsize, verbose=False)
        H_dr = (dr_counts['npairs'] * dr_counts['weightavg']).reshape(nbins, pimax)

        max_pimax = (pimax // pi_rebin) * pi_rebin
        H_dd_rebinned = H_dd[:, :max_pimax].reshape(nbins, max_pimax // pi_rebin, pi_rebin).sum(axis=2)
        H_dr_rebinned = H_dr[:, :max_pimax].reshape(nbins, max_pimax // pi_rebin, pi_rebin).sum(axis=2)
        H_rr_rebinned = H_rr[:, :max_pimax].reshape(nbins, max_pimax // pi_rebin, pi_rebin).sum(axis=2)

        WD = np.sum(data_weights)
        WR = np.sum(rand_weights)
        WD2 = np.sum(data_weights**2)
        WR2 = np.sum(rand_weights**2)

        if paircounts_file:
            print(f"Saving pair counts to {paircounts_file}")
            os.makedirs(os.path.dirname(paircounts_file), exist_ok=True)
            save_dict = {
                'rp_bins': rp_bins,
                'pi_rebin': pi_rebin,
                'max_pimax': max_pimax,
                'H_dd_rebinned': H_dd_rebinned,
                'H_dr_rebinned': H_dr_rebinned,
                'H_rr_rebinned': H_rr_rebinned,
                'WD': WD, 'WR': WR, 'WD2': WD2, 'WR2': WR2,
            }
            if dfil_bin_metadata is not None:
                save_dict.update(dfil_bin_metadata)
            np.savez(paircounts_file, **save_dict)

    # Normalize to get xi
    norm_DD = WD*WD - WD2
    norm_RR = WR*WR - WR2
    norm_DR = WD*WR

    DD = H_dd_rebinned / norm_DD
    RR = H_rr_rebinned / norm_RR
    DR = H_dr_rebinned / norm_DR

    with np.errstate(divide='ignore', invalid='ignore'):
        xi = (DD - 2*DR + RR) / RR
        xi[RR == 0] = 0

    return xi, rp_bins, max_pimax

def plot_xi_sigmapi(xi, sigma_edges, max_pimax, pi_rebin,
                    title=None, output_folder=None, plotname="xi_sigma_pi.png",
                    min_sep=0.0, vmin_global=None, vmax_global=None):
    """
    Plot ξ(σ, π) (unchanged from original).
    """
    pi_edges_rebinned = np.arange(0, max_pimax + pi_rebin, pi_rebin)
    X, Y = np.meshgrid(sigma_edges, pi_edges_rebinned)
    C = xi.T

    linthresh = 0.001
    if vmin_global is not None and vmax_global is not None:
        vmin = vmin_global
        vmax = vmax_global
    else:
        vmin = np.percentile(xi, 1)
        vmax = np.percentile(xi, 99)
        if vmin >= 0:
            vmin = -vmax / 2
        if vmax <= 0:
            vmax = -vmin / 2

    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X, Y, C, shading='flat', cmap='plasma', norm=norm)
    ax.set_xlabel(r'$\sigma$ [$h^{-1}$ Mpc]')
    ax.set_ylabel(r'$\pi$ [$h^{-1}$ Mpc]')
    ax.set_title(title if title else r'$\xi(\sigma, \pi)$')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\xi(\sigma,\pi)$')

    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    pi_centers = 0.5 * (pi_edges_rebinned[:-1] + pi_edges_rebinned[1:])
    Xc, Yc = np.meshgrid(sigma_centers, pi_centers)
    mask_region = (Xc >= 50) & (Xc <= 150) & (Yc >= 50) & (Yc <= 150)
    values_region = C[mask_region]
    if len(values_region) > 0:
        levels = np.percentile(values_region, [50, 70, 90])
        ax.contour(Xc, Yc, C, levels=levels, colors='k', linewidths=1.5)

    ax.set_xlim(left=min_sep)
    ax.set_ylim(bottom=min_sep)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        outpath = os.path.join(output_folder, plotname)
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"Saved ξ(σ, π) plot to {outpath}")
    else:
        plt.show()
    plt.close(fig)

# ---------------------------
# Monopole functions (unchanged)
# ---------------------------
from scipy.interpolate import RegularGridInterpolator

def compute_monopole(xi, sigma_edges, pi_edges, s_bins=None):
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])

    interp = RegularGridInterpolator(
        (sigma_centers, pi_centers), xi,
        bounds_error=False, fill_value=0.0
    )

    if s_bins is None:
        max_s = pi_edges[-1]
        s_bins = sigma_edges[sigma_edges <= max_s]
        if s_bins[-1] < max_s:
            s_bins = np.append(s_bins, max_s)
    s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])

    xi0 = np.zeros_like(s_centers)

    for i, s in enumerate(s_centers):
        π_max = min(s, pi_edges[-1])
        if π_max <= 0:
            xi0[i] = 0.0
            continue

        n_int = 100
        π_vals = np.linspace(0, π_max, n_int)
        σ_vals = np.sqrt(s**2 - π_vals**2)

        points = np.column_stack((σ_vals, π_vals))
        ξ_vals = interp(points)

        integral = np.trapz(ξ_vals, π_vals)
        xi0[i] = integral / s

    return s_centers, xi0

def plot_monopoles_combined(monopoles_list, labels, output_folder=None, filename='xi0_combined.png'):
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.tab10.colors
    for i, ((s, xi0), label) in enumerate(zip(monopoles_list, labels)):
        ax.plot(s, xi0*s**2, marker='o', linestyle='-', color=colors[i % len(colors)], label=label)
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^{2}\xi_0(s)$')
    ax.set_title('Monopoles ξ₀(s)')
    if np.any(monopoles_list[0][0] > 100):
        ax.axvline(102, color='k', linestyle=':', label='BAO scale')
    ax.legend(loc='lower left')
    plt.tight_layout()
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        outpath = os.path.join(output_folder, filename)
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"Saved combined monopole plot to {outpath}")
    else:
        plt.show()
    plt.close(fig)

# ---------------------------
# MAIN PROCEDURE
# ---------------------------

def main():
    print(f"""
    Running 3D box analysis:
    - Box size L = {L} Mpc/h
    - Magnitude cut: {mag_max}
    - nrand_mult: {nrand_mult}
    - dist_bin_mode: {dist_bin_mode}
    """)

    # Load data
    cat_full = load_catalog()
    if test_dilute < 1.0:
        cat_full = cat_full.sample(frac=test_dilute, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(cat_full)} galaxies (dilution factor {test_dilute})")
    cat_mag = select_sample(cat_full)

    print(f"Number of galaxies after magnitude cut: {len(cat_mag)}")

    # Plot magnitude histogram
    plot_redshift_k(cat_mag)

    # Split by filament distance
    bins, labels, _ = split_by_dist_fil_bins(cat_mag)

    # Generate random catalog: uniform in cube [0, L]^3
    nrand_full = nrand_mult * len(cat_mag)
    x_rand_full = np.random.uniform(0, L, nrand_full)
    y_rand_full = np.random.uniform(0, L, nrand_full)
    z_rand_full = np.random.uniform(0, L, nrand_full)
    random_full = pd.DataFrame({"x": x_rand_full, "y": y_rand_full, "z": z_rand_full})
    random_full["weight"] = 1.0   # uniform weights

    # Generate random catalogs for each bin
    randoms_bins_list = []
    for bin_df in bins:
        nrand_bin = nrand_mult * len(bin_df)
        x_rand = np.random.uniform(0, L, nrand_bin)
        y_rand = np.random.uniform(0, L, nrand_bin)
        z_rand = np.random.uniform(0, L, nrand_bin)
        rand_bin = pd.DataFrame({"x": x_rand, "y": y_rand, "z": z_rand})
        rand_bin["weight"] = 1.0
        randoms_bins_list.append(rand_bin)

    # Diagnostic plots
    plot_radec_distribution(cat_mag, random_full)  # shows x,y distribution
    plot_bin_data_and_randoms(cat_mag, random_full, label="Full Sample",
                              plotname=f"../plots/{folderName}/bin_full_data_randoms.png")
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        plot_bin_data_and_randoms(gxs, rxs, label=lab,
                                  plotname=f"../plots/{folderName}/bin_{i}_data_randoms.png")
        plot_radec_distribution(gxs, rxs, subsample=i)

    # First pass: compute ξ(σ, π) for all samples
    print("\nFirst pass: computing ξ(σ, π) for all samples...")
    all_results = []

    base_params = {
        'mag_max': mag_max,
        'min_sep_2d': min_sep_2d,
        'max_sep_2d': max_sep_2d,
        'bin_size_2d': bin_size_2d,
        'pi_rebin': pi_rebin,
        'nrand_mult': nrand_mult,
        'dist_bin_mode': dist_bin_mode
    }

    # Full sample
    params_full = base_params.copy()
    params_full['dist_bin_mode'] = 'full'
    paircounts_file_full = get_paircounts_filename("full", params_full)

    xi_full, rp_bins, max_pimax = compute_xi_sigmapi_cartesian(
        cat_mag["x"].values, cat_mag["y"].values, cat_mag["z"].values,
        random_full["x"].values, random_full["y"].values, random_full["z"].values,
        pi_rebin,
        data_weights=None,
        rand_weights=random_full["weight"].values,
        min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
        paircounts_file=paircounts_file_full,
        force_recompute=force_recompute_full
    )
    all_results.append( (xi_full, rp_bins, max_pimax, pi_rebin,
                        rf"$\xi(\sigma, \pi)$ Full Sample", "xi_sigma_pi_full.png") )

    # Monopole
    n_pi = xi_full.shape[1]
    pi_edges = np.arange(n_pi + 1) * pi_rebin
    s_centers, xi0_full = compute_monopole(xi_full, rp_bins, pi_edges)
    monopoles_list = [(s_centers, xi0_full)]
    labels_list = ['Full Sample']
    monopole_file = os.path.join(monopoles_dir, f"monopole_full_{params_full['dist_bin_mode']}.npz")
    np.savez(monopole_file, s=s_centers, xi0=xi0_full)

    # Distance bins
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        params_bin = base_params.copy()
        params_bin['dist_bin_mode'] = dist_bin_mode
        paircounts_file_bin = get_paircounts_filename(f"bin{i}", params_bin)

        dfil_bin_metadata = {
            'dfil_min': gxs["dist_fil"].min(),
            'dfil_max': gxs["dist_fil"].max(),
        }

        xi_bin, rp_bins_bin, max_pimax_bin = compute_xi_sigmapi_cartesian(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            rxs["x"].values, rxs["y"].values, rxs["z"].values,
            pi_rebin,
            data_weights=gxs["weight"].values if "weight" in gxs.columns else None,
            rand_weights=rxs["weight"].values,
            min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
            paircounts_file=paircounts_file_bin,
            force_recompute=force_recompute_bin,
            dfil_bin_metadata=dfil_bin_metadata
        )
        all_results.append( (xi_bin, rp_bins_bin, max_pimax_bin, pi_rebin,
                             rf"$\xi(\sigma, \pi)$ {lab}", f"xi_sigma_pi_bin{i}.png") )

        # Monopole
        n_pi_bin = xi_bin.shape[1]
        pi_edges_bin = np.arange(n_pi_bin + 1) * pi_rebin
        s_centers_bin, xi0_bin = compute_monopole(xi_bin, rp_bins_bin, pi_edges_bin)
        monopoles_list.append((s_centers_bin, xi0_bin))
        labels_list.append(lab)
        monopole_file = os.path.join(monopoles_dir, f"monopole_bin{i}_{params_bin['dist_bin_mode']}.npz")
        np.savez(monopole_file, s=s_centers_bin, xi0=xi0_bin, **dfil_bin_metadata)

    # Determine global color limits
    all_xi_flat = np.concatenate([xi.ravel() for xi, _, _, _, _, _ in all_results])
    vmin_global = np.percentile(all_xi_flat, 1)
    vmax_global = np.percentile(all_xi_flat, 99)
    if vmin_global >= 0:
        vmin_global = -vmax_global / 2
    if vmax_global <= 0:
        vmax_global = -vmin_global / 2
    print(f"Global color limits: vmin={vmin_global:.3f}, vmax={vmax_global:.3f}")

    # Second pass: plot
    print("\nSecond pass: plotting with fixed color scale...")
    for xi, sigma_edges, max_pimax, pi_rebin_val, title, plotname in all_results:
        plot_xi_sigmapi(xi, sigma_edges, max_pimax, pi_rebin_val,
                        title=title, output_folder=output_folder, plotname=plotname,
                        min_sep=min_sep_2d,
                        vmin_global=vmin_global, vmax_global=vmax_global)

    # Combined monopoles
    plot_monopoles_combined(monopoles_list, labels_list,
                            output_folder=output_folder,
                            filename='xi0_combined.png')

    print("All done.")

if __name__ == "__main__":
    main()