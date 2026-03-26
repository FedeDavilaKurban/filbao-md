"""
Unified pipeline for 2D correlation function and monopole analysis.
Supports both lightcone data (with RA, Dec, redshift) and 3D box data (cartesian coordinates).

MODE = 'lightcone'   or   'box'
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline, RegularGridInterpolator
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from matplotlib.colors import LogNorm, SymLogNorm
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs   # needed for lightcone skymap, safe to import always

# ================================
# MODE SELECTION
# ================================
MODE = "lightcone"          # "lightcone" or "box"

# ================================
# COMMON PARAMETERS
# ================================
force_recompute_full = True
force_recompute_bin = True

# ---- 2D correlation function parameters ------
min_sep_2d = 1.0          # minimum σ, π in Mpc/h
max_sep_2d = 150.0
bin_size_2d = 3.0
pi_rebin = 3                # rebin factor for π direction

# ------ dist_fil binning ------
dist_bin_mode = "percentile_intervals"
dist_bin_percentile_intervals = [   # used for "percentile_intervals"
    (0, 20),      # a–bth percentile
    (60, 80)      # c–dth percentile
]
nbins_dist = 4               # used only for percentile / equal_width
dist_bin_edges = [0, 5, 10, 15, 100]  # used only for "fixed"

# ------ Random catalog parameters ------
nrand_mult = 15                # Nr / Nd

# ================================
# MODE‑SPECIFIC PARAMETERS
# ================================
if MODE == "lightcone":
    # Lightcone sample parameters
    lightcone_filename = '../data/lightcone_real_and_rsd_withfil.csv'
    test_dilute = 0.5
    h = 1
    zmin, zmax = 0.07, 0.2
    mag_max = -21.2
    ran_method = 'poly'           # ['random_choice', 'piecewise', 'poly']
    if ran_method == 'poly':
        deg = 5
    gr_min = 0
    use_dec_weights = False
    nside = 256
    ran_radec_method = 'file'
    RADec_filepath = '../../data/lss_randoms_combined_cut_LARGE.csv'

    # ------ RSD / real-space redshift selection ------
    include_rsd = True               # True for RSD (z_obs), False for real-space (z_cos)
    real_redshift_col = 'z_cos'      # column name for real-space redshift (if include_rsd=False)

    # Output folders – add suffix to indicate RSD/real
    suffix = "_rsd" if include_rsd else "_real"
    folderName = f'XISIGMAPI_z{zmin:.2f}-{zmax:.2f}_mag{mag_max:.1f}_gr{gr_min:.1f}_nrand{nrand_mult}_RADECmethod{ran_radec_method}{suffix}'
    output_folder = f"../plots/{folderName}/"
    paircounts_dir = "../data/pair_counts/"
    monopoles_dir = "../data/monopoles/lightcone"

    # Cosmology (for comoving distance)
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=0.31)

else:  # MODE == "box"
    # Box parameters
    L = 1000.0               # box size in Mpc/h (hardcoded)
    h = 1
    mag_max = -21.5
    test_dilute = 1.0

    # Output folders
    folderName = f'XISIGMAPI_3Dbox_mag{mag_max:.1f}_nrand{nrand_mult}'
    output_folder = f"../plots/{folderName}/"
    paircounts_dir = "../data/pair_counts/"
    monopoles_dir = "../data/monopoles/3dbox"

    # Cosmology not used in box, but keep for compatibility
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=0.31)

# ================================
# CREATE OUTPUT FOLDERS
# ================================
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)
os.makedirs(paircounts_dir, exist_ok=True)
os.makedirs(monopoles_dir, exist_ok=True)

# ================================
# HELPER FUNCTIONS (mode‑agnostic)
# ================================
def safe_trapz(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

def ensure_dir_exists(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def save_figure(fig, path, dpi=300):
    ensure_dir_exists(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def get_paircounts_filename(bin_name, params):
    """
    Build a unique filename for saved pair counts.
    The dictionary 'params' may contain different keys depending on the mode.
    """
    parts = []
    if 'zmin' in params and 'zmax' in params:
        parts.append(f"z={params['zmin']:.2f}-{params['zmax']:.2f}")
    if 'mag_max' in params:
        parts.append(f"mag={params['mag_max']:.1f}")
    if 'gr_min' in params:
        parts.append(f"gr={params['gr_min']:.1f}")
    if 'min_sep_2d' in params and 'max_sep_2d' in params:
        parts.append(f"sep={params['min_sep_2d']}-{params['max_sep_2d']}")
    if 'bin_size_2d' in params:
        parts.append(f"bin={params['bin_size_2d']}")
    if 'pi_rebin' in params:
        parts.append(f"pi_rebin={params['pi_rebin']}")
    if 'nrand_mult' in params:
        parts.append(f"nrand={params['nrand_mult']}")
    if 'ran_radec_method' in params:
        parts.append(f"radec={params['ran_radec_method']}")
    if 'include_rsd' in params:
        parts.append(f"rsd={params['include_rsd']}")
    parts.append(f"bin={bin_name}")
    if 'dist_bin_mode' in params:
        parts.append(f"distbinmode={params['dist_bin_mode']}")
    fname = "_".join(parts).replace('.', 'p')
    return os.path.join(paircounts_dir, fname + ".npz")

# ================================
# DATA LOADING (mode‑specific)
# ================================
def load_catalog():
    if MODE == "lightcone":
        cat = pd.read_csv(lightcone_filename)
        if test_dilute < 1.0:
            cat = cat.sample(frac=test_dilute, random_state=42).reset_index(drop=True)
        # Rename common columns
        cat.rename(columns={'mag_r': 'mag_abs_r',
                            'dfil': 'dist_fil',
                            'ra_deg': 'ra',
                            'dec_deg': 'dec'}, inplace=True)
        # Choose redshift column based on RSD flag
        if include_rsd:
            cat.rename(columns={'z_obs': 'red'}, inplace=True)
        else:
            cat.rename(columns={real_redshift_col: 'red'}, inplace=True)
        return cat
    else:  # box
        df = pd.read_csv("../data/to_mock.csv", usecols=['x','y','z','vx','vy','vz','magstarsdssr'])
        df_fil = pd.read_csv("../data/mock_withfilament.csv")
        if "dist_fil" in df_fil.columns:
            df["dist_fil"] = df_fil["dist_fil"].values
        elif "dfil" in df_fil.columns:
            df["dist_fil"] = df_fil["dfil"].values
        else:
            raise ValueError("No dist_fil/dfil column found in filament file")
        df.rename(columns={'magstarsdssr': 'mag_abs_r'}, inplace=True)
        if test_dilute < 1.0:
            df = df.sample(frac=test_dilute, random_state=42).reset_index(drop=True)
        return df

def select_sample(cat):
    if MODE == "lightcone":
        cat_z = cat[(cat["red"] >= zmin) & (cat["red"] <= zmax)]
        cat_z_mag = cat_z[cat_z["mag_abs_r"] < mag_max].copy()
        return cat_z_mag
    else:
        return cat[cat["mag_abs_r"] < mag_max].copy()

def add_cartesian_coords(df):
    """
    For lightcone: compute comoving distance (Mpc/h) and convert to cartesian.
    For box: already have x,y,z – nothing to do.
    """
    if MODE == "lightcone":
        df["r"] = cosmo.comoving_distance(df["red"].values).value * h
        df["x"] = df["r"] * np.cos(np.radians(df["dec"])) * np.cos(np.radians(df["ra"]))
        df["y"] = df["r"] * np.cos(np.radians(df["dec"])) * np.sin(np.radians(df["ra"]))
        df["z"] = df["r"] * np.sin(np.radians(df["dec"]))
    # For box, x,y,z already present – just ensure they are floats
    return df

# ================================
# REDSHIFT DISTRIBUTION (lightcone only)
# ================================
def build_cdf_from_line(data, vmin, vmax, num_points=10000):
    hist, bin_edges = np.histogram(data, bins=40, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    def LinearFunction(x, a, b): return a * x + b
    def BreakFunction(x, a1, b1, a2, xb):
        yi = lambda x_: LinearFunction(x_, a1, b1)
        yo = lambda x_: LinearFunction(xb, a1, b1) + ((x_ - xb) * a2)
        return np.piecewise(x, [x < xb, x >= xb], [yi, yo])
    bounds = [[-np.inf, -np.inf, -np.inf, vmin], [np.inf, np.inf, np.inf, vmax]]
    popt, _ = curve_fit(BreakFunction, bin_centers, hist, bounds=bounds)
    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = BreakFunction(z_vals, *popt)
    pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=None)
    integ = safe_trapz(pdf_vals, z_vals)
    if integ > 0:
        pdf_vals = pdf_vals / integ
    dz = z_vals[1] - z_vals[0]
    cdf_vals = np.cumsum(pdf_vals) * dz
    if cdf_vals[-1] > 0:
        cdf_vals = cdf_vals / cdf_vals[-1]
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals

def build_cdf_from_parabola(data, vmin, vmax, deg, num_points=10000):
    hist, bin_edges = np.histogram(data, bins=40, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    poly = Polynomial.fit(bin_centers, hist, deg=deg)
    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = poly(z_vals)
    pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=None)
    integ = safe_trapz(pdf_vals, z_vals)
    if integ > 0:
        pdf_vals = pdf_vals / integ
    dz = z_vals[1] - z_vals[0]
    cdf_vals = np.cumsum(pdf_vals) * dz
    if cdf_vals[-1] > 0:
        cdf_vals = cdf_vals / cdf_vals[-1]
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals

def generate_random_red(redshift, nrand, ran_method, deg=None):
    if ran_method == "poly":
        cdf_inv_z, _, _, _ = build_cdf_from_parabola(redshift, redshift.min(), redshift.max(), deg)
        u = np.random.uniform(0.0, 1.0, nrand)
        red_random = cdf_inv_z(u)
    elif ran_method == "piecewise":
        cdf_inv_z, _, _, _ = build_cdf_from_line(redshift, redshift.min(), redshift.max())
        u = np.random.uniform(0.0, 1.0, nrand)
        red_random = cdf_inv_z(u)
    elif ran_method == "random_choice":
        red_random = np.random.choice(redshift, nrand)
    else:
        raise ValueError(f"Unknown ran_method: {ran_method}")
    return red_random

# ================================
# WEIGHT COMPUTATION (lightcone only)
# ================================
def compute_dec_weights(data_dec, rand_dec, alpha=1.0, method="auto",
                        kde_threshold=1_000_000, nbins=40, spline_s=0.5,
                        bw_factor=1.2, n_grid=300):
    n_rand = len(rand_dec)
    if method == "auto":
        method = "kde" if n_rand < kde_threshold else "spline"
    print(f"Using {method} method for declination weights")
    epsilon = 1e-10

    if method == "kde":
        kde_data = gaussian_kde(data_dec)
        kde_data.set_bandwidth(kde_data.factor * bw_factor)
        kde_rand = gaussian_kde(rand_dec)
        kde_rand.set_bandwidth(kde_rand.factor * bw_factor)
        dec_min = min(data_dec.min(), rand_dec.min())
        dec_max = max(data_dec.max(), rand_dec.max())
        grid = np.linspace(dec_min, dec_max, n_grid)
        density_data_grid = kde_data(grid)
        density_rand_grid = kde_rand(grid)
        ratio_grid = (density_data_grid + epsilon) / (density_rand_grid + epsilon)
        weights = np.interp(rand_dec, grid, ratio_grid)
    elif method == "spline":
        hist_data, edges = np.histogram(data_dec, bins=nbins)
        hist_rand, _ = np.histogram(rand_dec, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ratio = (hist_data + epsilon) / (hist_rand + epsilon)
        spline = UnivariateSpline(centers, ratio, s=spline_s)
        weights = spline(rand_dec)
        weights = np.clip(weights, 0.01, None)
    else:
        raise ValueError("method must be 'auto', 'kde', or 'spline'")

    weights = 1.0 + alpha * (weights - 1.0)
    weights /= np.mean(weights)
    return weights

# ================================
# RA/DEC GENERATION (lightcone only)
# ================================
def generate_master_radec(full_cat, nrand_total, nside, ran_radec_method,
                          ra_preload=None, dec_preload=None):
    if ran_radec_method == 'file':
        if ra_preload is None or dec_preload is None:
            raise ValueError("Method 'file' requires ra_preload and dec_preload.")
        if len(ra_preload) < nrand_total:
            raise ValueError(f"RA/Dec arrays contain {len(ra_preload)} points but {nrand_total} are needed.")
        ra_master = ra_preload[:nrand_total]
        dec_master = dec_preload[:nrand_total]
    else:
        raise ValueError(f"Unknown ran_radec_method: {ran_radec_method}")
    return ra_master, dec_master

# ================================
# PLOTTING (mode‑specific)
# ================================
def plot_redshift_magnitude(cat):
    if MODE == "lightcone":
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist2d(cat["red"], cat["mag_abs_r"], bins=40, cmap="Blues", norm=LogNorm())
        ax.axvline(zmin, color="k", linestyle=":")
        ax.axvline(zmax, color="k", linestyle=":")
        ax.axhline(mag_max, color="k", linestyle=":")
        ax.invert_yaxis()
        ax.set_xlabel("Redshift")
        ax.set_ylabel("K")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Number of galaxies')
        filename = f"../plots/{folderName}/redshift_magnitude.png"
        save_figure(fig, filename, dpi=100)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(cat["mag_abs_r"], bins=40, color="C00", alpha=0.8)
        ax.axvline(mag_max, color="k", linestyle=":")
        ax.set_xlabel("Absolute magnitude r")
        ax.set_ylabel("Count")
        filename = f"../plots/{folderName}/magnitude_hist.png"
        save_figure(fig, filename, dpi=100)

def plot_radec_distribution(cat, randoms, subsample=None):
    if MODE == "lightcone":
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        axes[0].hist(randoms["ra"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
        axes[0].hist(cat["ra"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
        axes[0].set_xlabel("RA")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[1].hist(cat["dec"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
        axes[1].hist(randoms["dec"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
        if "weight" in randoms.columns:
            axes[1].hist(randoms["dec"], bins=40, density=True, weights=randoms["weight"],
                         linestyle='--', color='k', histtype='step', label="Weighted Randoms")
        axes[1].set_xlabel("Dec")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        if subsample is None:
            filename = f"../plots/{folderName}/radec_distribution.png"
            axtitle = 'Total sample'
        else:
            filename = f"../plots/{folderName}/radec_distribution_bin{subsample}.png"
            axtitle = rf"r_{{fil}} \in {cat['dist_fil'].min():.1f}-{cat['dist_fil'].max():.1f} Mpc/h"
    else:
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
    if MODE == "lightcone":
        fig = plt.figure(figsize=(7, 14))
        ax_map = fig.add_subplot(3, 1, 1)
        ax_map.scatter(rxs["ra"], rxs["dec"], s=1.5, color="k", alpha=0.5, label="Randoms")
        ax_map.scatter(gxs["ra"], gxs["dec"], s=1, color="C00", label="Galaxies")
        ax_map.set_xlabel("RA")
        ax_map.set_ylabel("DEC")
        ax_map.legend(loc='upper right')
        ax_map.set_title(label)
        ax_hist_z = fig.add_subplot(3, 1, 2)
        ax_hist_z.hist(gxs["red"], bins=40, density=True, histtype="stepfilled",
                       color="C00", alpha=0.8, label="Galaxies (unweighted)")
        ax_hist_z.hist(rxs["red"], bins=40, density=True, histtype="step",
                       color="k", lw=1.5, label="Randoms")
        if "weight" in gxs.columns:
            ax_hist_z.hist(gxs["red"], bins=40, density=True, weights=gxs["weight"],
                           histtype="step", color="C00", linestyle="--", lw=2,
                           label="Galaxies (weighted)")
        ax_hist_z.set_xlabel("Redshift")
        ax_hist_z.set_ylabel("PDF")
        ax_hist_z.legend()
        ax_hist_dist = fig.add_subplot(3, 1, 3)
        ax_hist_dist.hist(gxs["dist_fil"], bins=40, density=True, color="C03", alpha=0.9)
        ax_hist_dist.set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
        ax_hist_dist.set_ylabel("PDF")
        plt.tight_layout()
        save_figure(fig, plotname, dpi=200)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7, 10))
        ax_map = axes[0]
        ax_map.scatter(rxs["x"], rxs["y"], s=1.5, color="k", alpha=0.5, label="Randoms")
        ax_map.scatter(gxs["x"], gxs["y"], s=1, color="C00", label="Galaxies")
        ax_map.set_xlabel("x [Mpc/h]")
        ax_map.set_ylabel("y [Mpc/h]")
        ax_map.legend(loc='upper right')
        ax_map.set_title(label)
        ax_map.set_aspect('equal')
        ax_hist_dist = axes[1]
        ax_hist_dist.hist(gxs["dist_fil"], bins=40, density=True, color="C03", alpha=0.9)
        ax_hist_dist.set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
        ax_hist_dist.set_ylabel("PDF")
        plt.tight_layout()
        save_figure(fig, plotname, dpi=200)

# ================================
# 2D CORRELATION FUNCTION
# ================================
def compute_xi_sigmapi(x_data, y_data, z_data,
                       x_rand, y_rand, z_rand,
                       pi_rebin,
                       data_weights=None, rand_weights=None,
                       min_sep=0.0, max_sep=50.0, bin_size=2.0,
                       paircounts_file=None, force_recompute=False,
                       dfil_bin_metadata=None):
    """
    Compute ξ(σ, π) using Corrfunc.
    If MODE == "box", the calculation is periodic with boxsize L.
    If MODE == "lightcone", it is non‑periodic.
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

    periodic = (MODE == "box")
    boxsize = L if periodic else None

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
        nbins = int((max_sep - min_sep) / bin_size)
        rp_bins = np.linspace(min_sep, max_sep, nbins + 1)
        pimax = int(max_sep)
        nthreads = min(multiprocessing.cpu_count(), 16)

        # Weighted pair counts: using weight_type='pair_product'
        dd_counts = DDrppi(1, nthreads, pimax, rp_bins,
                           x_data, y_data, z_data,
                           weights1=data_weights,
                           weight_type='pair_product',
                           periodic=periodic, boxsize=boxsize, verbose=False)
        H_dd = dd_counts['npairs'].reshape(nbins, pimax)   # weighted sum of weights product

        rr_counts = DDrppi(1, nthreads, pimax, rp_bins,
                           x_rand, y_rand, z_rand,
                           weights1=rand_weights,
                           weight_type='pair_product',
                           periodic=periodic, boxsize=boxsize, verbose=False)
        H_rr = rr_counts['npairs'].reshape(nbins, pimax)

        dr_counts = DDrppi(0, nthreads, pimax, rp_bins,
                           x_data, y_data, z_data,
                           X2=x_rand, Y2=y_rand, Z2=z_rand,
                           weights1=data_weights,
                           weights2=rand_weights,
                           weight_type='pair_product',
                           periodic=periodic, boxsize=boxsize, verbose=False)
        H_dr = dr_counts['npairs'].reshape(nbins, pimax)

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

    # Normalization
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

# ================================
# MONOPOLE FUNCTIONS
# ================================
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

# ================================
# DISTANCE BIN SPLITTING (common)
# ================================
def split_by_dist_fil_bins(cat):
    values = cat["dist_fil"].values

    if dist_bin_mode == "custom_intervals":
        # Not implemented in original, but kept for completeness
        raise NotImplementedError("custom_intervals not implemented in combined version")
    elif dist_bin_mode == "percentile_intervals":
        if not hasattr(dist_bin_percentile_intervals, '__iter__'):
            raise ValueError("dist_bin_percentile_intervals must be a list of (low, high) tuples")
        bins = []
        labels = []
        for (lo_pct, hi_pct) in dist_bin_percentile_intervals:
            lo_val = np.percentile(values, lo_pct)
            hi_val = np.percentile(values, hi_pct)
            mask = (values >= lo_val) & (values <= hi_val)
            subset = cat.loc[mask].copy()
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
        bins.append(cat.loc[mask].copy())
        labels.append(f"${lo:.1f} < r_{{fil}} \\leq {hi:.1f}$")
    return bins, labels, edges

# ================================
# MAIN PROCEDURE
# ================================
def main():
    print(f"Running in {MODE} mode.")
    if MODE == "lightcone":
        print(f"Using {'RSD (z_obs)' if include_rsd else f'real-space ({real_redshift_col})'} redshifts.")

    # Load and select sample
    cat_full = load_catalog()
    cat = select_sample(cat_full)
    print(f"Selected {len(cat)} galaxies.")

    # Add cartesian coordinates (for lightcone, also compute r)
    cat = add_cartesian_coords(cat)

    # Plot magnitude/redshift diagnostic
    plot_redshift_magnitude(cat_full)

    # Split into distance bins
    bins, labels, _ = split_by_dist_fil_bins(cat)
    print(f"Split into {len(bins)} bins.")

    # ------------------------------------------------------------------
    # Generate random catalogs (mode‑specific)
    # ------------------------------------------------------------------
    if MODE == "lightcone":
        # Preload RA/Dec from file if needed
        if ran_radec_method == 'file':
            print("Reading RA/Dec file...")
            radec_data = pd.read_csv(RADec_filepath)
            ra_random_file = radec_data["ra"].values
            dec_random_file = radec_data["dec"].values
            idx = np.random.permutation(len(ra_random_file))
            ra_random_file = ra_random_file[idx]
            dec_random_file = dec_random_file[idx]
            print(f"Loaded {len(ra_random_file)} random RA/Dec points")
        else:
            ra_random_file = dec_random_file = None

        # Total number of randoms needed
        nrand_full = nrand_mult * len(cat)
        nrand_bins = [nrand_mult * len(b) for b in bins]
        total_rand = nrand_full + sum(nrand_bins)

        # Generate master RA/Dec array
        master_ra, master_dec = generate_master_radec(
            full_cat=cat,
            nrand_total=total_rand,
            nside=nside,
            ran_radec_method=ran_radec_method,
            ra_preload=ra_random_file,
            dec_preload=dec_random_file
        )

        # Full sample random
        ra_rand_full = master_ra[:nrand_full]
        dec_rand_full = master_dec[:nrand_full]
        ptr = nrand_full
        red_full = generate_random_red(cat["red"].values, nrand_full, ran_method,
                                       deg if ran_method == "poly" else None)
        random_full = pd.DataFrame({"ra": ra_rand_full, "dec": dec_rand_full, "red": red_full})
        random_full["r"] = cosmo.comoving_distance(random_full["red"].values).value * h
        # --- ADDED: compute x, y, z ---
        random_full["x"] = random_full["r"] * np.cos(np.radians(random_full["dec"])) * np.cos(np.radians(random_full["ra"]))
        random_full["y"] = random_full["r"] * np.cos(np.radians(random_full["dec"])) * np.sin(np.radians(random_full["ra"]))
        random_full["z"] = random_full["r"] * np.sin(np.radians(random_full["dec"]))
        # -----------------------------
        if use_dec_weights:
            rand_weights_full = compute_dec_weights(cat["dec"].values, random_full["dec"].values,
                                                    nbins=40, method="kde", alpha=1)
            random_full["weight"] = rand_weights_full
        else:
            random_full["weight"] = 1.0

        # Binned random catalogs
        randoms_bins_list = []
        for bin_df in bins:
            nrand_bin = nrand_bins[len(randoms_bins_list)]
            ra_bin = master_ra[ptr:ptr + nrand_bin]
            dec_bin = master_dec[ptr:ptr + nrand_bin]
            ptr += nrand_bin
            red_bin = generate_random_red(cat["red"].values, nrand_bin, ran_method,
                                          deg if ran_method == "poly" else None)
            rand_bin = pd.DataFrame({"ra": ra_bin, "dec": dec_bin, "red": red_bin})
            rand_bin["r"] = cosmo.comoving_distance(rand_bin["red"].values).value * h
            # --- ADDED: compute x, y, z ---
            rand_bin["x"] = rand_bin["r"] * np.cos(np.radians(rand_bin["dec"])) * np.cos(np.radians(rand_bin["ra"]))
            rand_bin["y"] = rand_bin["r"] * np.cos(np.radians(rand_bin["dec"])) * np.sin(np.radians(rand_bin["ra"]))
            rand_bin["z"] = rand_bin["r"] * np.sin(np.radians(rand_bin["dec"]))
            # -----------------------------
            if use_dec_weights:
                dec_weights = compute_dec_weights(bin_df["dec"].values, rand_bin["dec"].values,
                                                  nbins=40, method="kde", alpha=1)
                rand_bin["weight"] = dec_weights
            else:
                rand_bin["weight"] = 1.0
            randoms_bins_list.append(rand_bin)

        # For galaxies, we keep uniform weight (or could apply redshift weighting if desired)
        for bin_df in bins:
            bin_df["weight"] = 1.0
        cat["weight"] = 1.0

    else:  # MODE == "box"
        # Full sample random
        nrand_full = nrand_mult * len(cat)
        x_rand_full = np.random.uniform(0, L, nrand_full)
        y_rand_full = np.random.uniform(0, L, nrand_full)
        z_rand_full = np.random.uniform(0, L, nrand_full)
        random_full = pd.DataFrame({"x": x_rand_full, "y": y_rand_full, "z": z_rand_full})
        random_full["weight"] = 1.0

        # Binned random catalogs
        randoms_bins_list = []
        for bin_df in bins:
            nrand_bin = nrand_mult * len(bin_df)
            x_rand = np.random.uniform(0, L, nrand_bin)
            y_rand = np.random.uniform(0, L, nrand_bin)
            z_rand = np.random.uniform(0, L, nrand_bin)
            rand_bin = pd.DataFrame({"x": x_rand, "y": y_rand, "z": z_rand})
            rand_bin["weight"] = 1.0
            randoms_bins_list.append(rand_bin)

        # Galaxies already have x,y,z; assign uniform weight
        cat["weight"] = 1.0
        for bin_df in bins:
            bin_df["weight"] = 1.0

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    plot_radec_distribution(cat, random_full)
    plot_bin_data_and_randoms(cat, random_full, label="Full Sample",
                              plotname=f"../plots/{folderName}/bin_full_data_randoms.png")
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        plot_bin_data_and_randoms(gxs, rxs, label=lab,
                                  plotname=f"../plots/{folderName}/bin_{i}_data_randoms.png")
        plot_radec_distribution(gxs, rxs, subsample=i)

    # ------------------------------------------------------------------
    # First pass: compute ξ(σ, π) for all samples (store xi)
    # ------------------------------------------------------------------
    print("\nFirst pass: computing ξ(σ, π) for all samples...")
    all_results = []

    # Base parameters for paircounts file naming
    base_params = {
        'mag_max': mag_max,
        'min_sep_2d': min_sep_2d,
        'max_sep_2d': max_sep_2d,
        'bin_size_2d': bin_size_2d,
        'pi_rebin': pi_rebin,
        'nrand_mult': nrand_mult,
        'dist_bin_mode': dist_bin_mode,
    }
    if MODE == "lightcone":
        base_params.update({
            'zmin': zmin, 'zmax': zmax,
            'gr_min': gr_min,
            'ran_radec_method': ran_radec_method,
            'include_rsd': include_rsd,
        })

    # Full sample
    params_full = base_params.copy()
    params_full['dist_bin_mode'] = 'full'
    paircounts_file_full = get_paircounts_filename("full", params_full)

    xi_full, rp_bins, max_pimax = compute_xi_sigmapi(
        cat["x"].values, cat["y"].values, cat["z"].values,
        random_full["x"].values, random_full["y"].values, random_full["z"].values,
        pi_rebin,
        data_weights=cat["weight"].values,
        rand_weights=random_full["weight"].values,
        min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
        paircounts_file=paircounts_file_full,
        force_recompute=force_recompute_full
    )
    all_results.append( (xi_full, rp_bins, max_pimax, pi_rebin,
                         rf"$\xi(\sigma, \pi)$ Full Sample", "xi_sigma_pi_full.png") )

    # Monopole for full sample
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

        xi_bin, rp_bins_bin, max_pimax_bin = compute_xi_sigmapi(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            rxs["x"].values, rxs["y"].values, rxs["z"].values,
            pi_rebin,
            data_weights=gxs["weight"].values,
            rand_weights=rxs["weight"].values,
            min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
            paircounts_file=paircounts_file_bin,
            force_recompute=force_recompute_bin,
            dfil_bin_metadata=dfil_bin_metadata
        )
        all_results.append( (xi_bin, rp_bins_bin, max_pimax_bin, pi_rebin,
                             rf"$\xi(\sigma, \pi)$ {lab}", f"xi_sigma_pi_bin{i}.png") )

        # Monopole for this bin
        n_pi_bin = xi_bin.shape[1]
        pi_edges_bin = np.arange(n_pi_bin + 1) * pi_rebin
        s_centers_bin, xi0_bin = compute_monopole(xi_bin, rp_bins_bin, pi_edges_bin)
        monopoles_list.append((s_centers_bin, xi0_bin))
        labels_list.append(lab)
        monopole_file = os.path.join(monopoles_dir, f"monopole_bin{i}_{params_bin['dist_bin_mode']}.npz")
        np.savez(monopole_file, s=s_centers_bin, xi0=xi0_bin, **dfil_bin_metadata)

    # ------------------------------------------------------------------
    # Determine global color limits from all xi arrays
    # ------------------------------------------------------------------
    all_xi_flat = np.concatenate([xi.ravel() for xi, _, _, _, _, _ in all_results])
    vmin_global = np.percentile(all_xi_flat, 1)
    vmax_global = np.percentile(all_xi_flat, 99)
    if vmin_global >= 0:
        vmin_global = -vmax_global / 2
    if vmax_global <= 0:
        vmax_global = -vmin_global / 2
    print(f"Global color limits: vmin={vmin_global:.3f}, vmax={vmax_global:.3f}")

    # ------------------------------------------------------------------
    # Second pass: plot all ξ(σ, π) with fixed color scale
    # ------------------------------------------------------------------
    print("\nSecond pass: plotting with fixed color scale...")
    for xi, sigma_edges, max_pimax, pi_rebin_val, title, plotname in all_results:
        plot_xi_sigmapi(xi, sigma_edges, max_pimax, pi_rebin_val,
                        title=title, output_folder=output_folder, plotname=plotname,
                        min_sep=min_sep_2d,
                        vmin_global=vmin_global, vmax_global=vmax_global)

    # ------------------------------------------------------------------
    # Combined monopole plot
    # ------------------------------------------------------------------
    plot_monopoles_combined(monopoles_list, labels_list,
                            output_folder=output_folder,
                            filename='xi0_combined.png')

    print("All done.")

if __name__ == "__main__":
    main()