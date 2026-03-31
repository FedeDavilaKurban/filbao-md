"""
BOX analysis
- xi(s), xi(mu, s) and xi_0(s) w/analytical randoms

"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import SymLogNorm

# ================================
# BOX‑SPECIFIC PARAMETERS
# ================================
L = 1000.0               # box size in Mpc/h
mag_max = -21.2          # magnitude cut
test_dilute = 1.0        # fraction of data to use (for testing)
force_recompute_full = False
force_recompute_bin = False

# ---- 2D correlation function parameters (s, μ) ------
min_sep_2d = 1.0          # minimum s in Mpc/h
max_sep_2d = 150.0
bin_size_2d = 2.0
pi_rebin = bin_size_2d  # for xi(σ, π) rebinning (same as s bin size)
# Number of μ bins will be set equal to number of s bins (no rebinning)

# ------ dist_fil binning ------
dist_bin_mode = "percentile_intervals"  # options: "custom_intervals", "percentile_intervals", "percentile", "equal_width", "fixed"
dist_bin_percentile_intervals = [   # used for "percentile_intervals"
    (0, 30),      # a–bth percentile
    (50, 100)      # c–dth percentile
]
nbins_dist = 4               # used only for percentile / equal_width
dist_bin_edges = [0, 5, 10, 15, 100]  # used only for "fixed"

# Output folders
folderName = f'XISMU_box_mag{mag_max:.1f}'
output_folder = f"../plots/{folderName}/"
paircounts_dir = "../data/pair_counts/box"
monopoles_dir = "../data/monopoles/box"

# ================================
# CREATE OUTPUT FOLDERS
# ================================
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)
os.makedirs(paircounts_dir, exist_ok=True)
os.makedirs(monopoles_dir, exist_ok=True)

# ================================
# HELPER FUNCTIONS
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
    """Build a unique filename for saved pair counts."""
    parts = []
    if 'mag_max' in params:
        parts.append(f"mag={params['mag_max']:.1f}")
    if 'min_sep_2d' in params and 'max_sep_2d' in params:
        parts.append(f"sep={params['min_sep_2d']}-{params['max_sep_2d']}")
    if 'bin_size_2d' in params:
        parts.append(f"bin={params['bin_size_2d']}")
    parts.append(f"bin={bin_name}")
    if 'dist_bin_mode' in params:
        parts.append(f"distbinmode={params['dist_bin_mode']}")
    fname = "_".join(parts).replace('.', 'p')
    return os.path.join(paircounts_dir, fname + ".npz")

# ================================
# DATA LOADING (BOX)
# ================================
def load_catalog():
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
    return cat[cat["mag_abs_r"] < mag_max].copy()

# ================================
# PLOTTING (BOX)
# ================================
def plot_magnitude_hist(cat):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cat["mag_abs_r"], bins=40, color="C00", alpha=0.8)
    ax.set_yscale("log")
    ax.axvline(mag_max, color="k", linestyle=":")
    ax.set_xlabel("Absolute magnitude r")
    ax.set_ylabel("Count")
    filename = f"../plots/{folderName}/magnitude_hist.png"
    save_figure(fig, filename, dpi=100)

def plot_bin_data(gxs, label, plotname):
    # Define slice in z
    gxs_plot = gxs[(gxs["z"] < 100.)]
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    # Plot slice
    ax_map = axes[0]
    # hist2d returns (counts, xedges, yedges, image)
    h, xedges, yedges, im = ax_map.hist2d(
        gxs_plot["x"], gxs_plot["y"],
        bins=100, cmap="Blues",
        norm=SymLogNorm(linthresh=1, vmin=1, vmax=gxs_plot.shape[0]/500)
    )
    # Add colorbar
    fig.colorbar(im, ax=ax_map, label="N. of galaxies")

    ax_map.set_xlabel("x [Mpc/h]")
    ax_map.set_ylabel("y [Mpc/h]")
    ax_map.set_title(label)
    ax_map.set_aspect('equal')
    ax_map.set_title('z < 100 Mpc/h')  # this overwrites previous title – you may want to combine or remove one

    # Plot distfil histogram
    ax_hist_dist = axes[1]
    ax_hist_dist.hist(gxs["dist_fil"], bins=40, density=True, color="C03", alpha=0.9)
    ax_hist_dist.set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
    ax_hist_dist.set_ylabel("PDF")

    plt.tight_layout()
    save_figure(fig, plotname, dpi=200)


# ================================
# 2D CORRELATION FUNCTION (xi(s, μ))
# ================================
def compute_xi_s_mu(x_data, y_data, z_data,
                    min_sep=0.0, max_sep=150.0, bin_size=2.0,
                    paircounts_file=None, force_recompute=False,
                    dfil_bin_metadata=None):

    import multiprocessing
    from Corrfunc.theory.DDsmu import DDsmu

    periodic = True
    boxsize = L

    nbins_s = int((max_sep - min_sep) / bin_size)
    s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)
    nbins_mu = nbins_s*2
    mu_max = 1.0
    mu_bins = np.linspace(0.0, mu_max, nbins_mu + 1)
    nthreads = min(multiprocessing.cpu_count()-1, 16)

    # Load or compute DD
    if paircounts_file and os.path.exists(paircounts_file) and not force_recompute:
        print(f"Loading precomputed DD from {paircounts_file}")
        data = np.load(paircounts_file)
        H_dd = data['H_dd']
        s_bins = data['s_bins']
        mu_bins = data['mu_bins']
        nbins_s = len(s_bins) - 1
        nbins_mu = len(mu_bins) - 1
    else:
        print("Computing DDsmu...")
        dd_counts = DDsmu(autocorr=1, nthreads=nthreads, binfile=s_bins,
                          mu_max=mu_max, nmu_bins=nbins_mu,
                          X1=x_data, Y1=y_data, Z1=z_data,
                          periodic=periodic, boxsize=boxsize,
                          verbose=False)
        H_dd = dd_counts['npairs'].reshape(nbins_s, nbins_mu).astype(np.float64)

        if paircounts_file:
            print(f"Saving DD to {paircounts_file}")
            os.makedirs(os.path.dirname(paircounts_file), exist_ok=True)
            save_dict = {'s_bins': s_bins, 'mu_bins': mu_bins, 'H_dd': H_dd}
            if dfil_bin_metadata is not None:
                save_dict.update(dfil_bin_metadata)
            np.savez(paircounts_file, **save_dict)

    # Analytic RR (unordered pairs)
    N = len(x_data)
    V = boxsize**3

    RR = np.zeros((nbins_s, nbins_mu))

    for i in range(nbins_s):
        s_lo = s_bins[i]
        s_hi = s_bins[i+1]

        vol_shell = (4* np.pi / 3.0) * (s_hi**3 - s_lo**3)

        for j in range(nbins_mu):
            mu_lo = mu_bins[j]
            mu_hi = mu_bins[j+1]

            dmu = mu_hi - mu_lo

            dV = vol_shell * dmu

            RR[i, j] = (N * (N - 1) / V) * dV

    # Compute xi
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = H_dd / RR - 1.0
        xi[RR == 0] = np.nan

    print("DIAGNOSTICS:")
    print(f"Total DD pairs = {np.sum(H_dd):.3e}")
    print(f"Expected RR pairs = {np.sum(RR):.3e}")
    print(f"Ratio DD/RR (global) = {np.sum(H_dd)/np.sum(RR):.6f}")

    return xi, s_bins, mu_bins

def compute_xi_s(x_data, y_data, z_data,
                    min_sep=0.0, max_sep=150.0, bin_size=2.0,
                    paircounts_file=None, force_recompute=False,
                    dfil_bin_metadata=None):

    import multiprocessing
    from Corrfunc.theory.DD import DD

    periodic = True
    boxsize = L

    nbins_s = int((max_sep - min_sep) / bin_size)
    s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)
    nthreads = min(multiprocessing.cpu_count()-1, 16)

    # Load or compute DD
    if paircounts_file and os.path.exists(paircounts_file) and not force_recompute:
        print(f"Loading precomputed DD from {paircounts_file}")
        data = np.load(paircounts_file)
        H_dd = data['H_dd']
        s_bins = data['s_bins']
        nbins_s = len(s_bins) - 1
    else:
        print("Computing DD...")
        dd_counts = DD(autocorr=1, nthreads=nthreads, binfile=s_bins,
                          X1=x_data, Y1=y_data, Z1=z_data,
                          periodic=periodic, boxsize=boxsize,
                          verbose=False)
        H_dd = dd_counts['npairs'].astype(np.float64)

        if paircounts_file:
            print(f"Saving DD to {paircounts_file}")
            os.makedirs(os.path.dirname(paircounts_file), exist_ok=True)
            save_dict = {'s_bins': s_bins, 'H_dd': H_dd}
            if dfil_bin_metadata is not None:
                save_dict.update(dfil_bin_metadata)
            np.savez(paircounts_file, **save_dict)

    # Analytic RR (unordered pairs)
    N = len(x_data)
    V = boxsize**3
    smin = s_bins[:-1]
    smax = s_bins[1:]
    shell_vol = (4.0/3.0) * np.pi * (smax**3 - smin**3)

    # Correct factor 1/2 because pairs are unordered
    RR_s = shell_vol * (N * (N - 1) / V)

    # Compute xi
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = H_dd / RR_s - 1.0
        xi[RR_s == 0] = np.nan

    print("DIAGNOSTICS:")
    print(f"Total DD pairs = {np.sum(H_dd):.3e}")
    print(f"Expected RR pairs = {np.sum(RR_s):.3e}")
    print(f"Ratio DD/RR (global) = {np.sum(H_dd)/np.sum(RR_s):.6f}")

    return xi, s_bins


def plot_xi_s_mu(xi, s_edges, mu_edges,
                 title=None, output_folder=None, plotname="xi_s_mu.png",
                 min_sep=0.0, vmin_global=None, vmax_global=None,
                 contour_levels=None, contour_colors='black', contour_linewidths=1,
                 contour_kwargs=None):
    """
    Plot ξ(s, μ) with s on Y-axis, μ on X-axis (decreasing from 1 left to 0 right),
    optionally overlaying contour lines.
    """
    # Reverse mu_edges and xi along μ axis
    mu_edges_rev = mu_edges[::-1]
    xi_rev = xi[:, ::-1]  # shape (nbins_s, nbins_mu)

    # Meshgrid: X = μ (reversed edges), Y = s
    X, Y = np.meshgrid(mu_edges_rev, s_edges)
    C = xi_rev

    # Determine color scale
    linthresh = 0.001
    if vmin_global is not None and vmax_global is not None:
        vmin, vmax = vmin_global, vmax_global
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

    # Contours if requested
    if contour_levels is not None:
        ckwargs = {'colors': contour_colors, 'linewidths': contour_linewidths}
        if contour_kwargs is not None:
            ckwargs.update(contour_kwargs)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        mu_centers_rev = 0.5 * (mu_edges_rev[:-1] + mu_edges_rev[1:])
        Xc, Yc = np.meshgrid(mu_centers_rev, s_centers)
        ax.contour(Xc, Yc, C, levels=contour_levels, **ckwargs)

    ax.axhline(105, color='white', linestyle=':', linewidth=2, label='BAO scale', alpha=.8)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$s$ [$h^{-1}$ Mpc]')
    ax.set_title(title if title else r'$\xi(s, \mu)$')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\xi(s,\mu)$')

    ax.set_ylim(bottom=min_sep, top=s_edges[-1])
    ax.set_xlim(right=0, left=1)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        outpath = os.path.join(output_folder, plotname)
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"Saved ξ(s, μ) plot to {outpath}")
    else:
        plt.show()
    plt.close(fig)

def plot_xi_s_combined(xi_s_list, labels, output_folder=None, filename='xi_s_combined.png'):
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.tab10.colors
    if np.any(xi_s_list[0][0] > 100):
        ax.axvline(105, color='k', linestyle=':', label='BAO scale')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    for i, ((s_centers, xi_s), label) in enumerate(zip(xi_s_list, labels)):
        ax.plot(s_centers, xi_s * s_centers**2, marker='o', linestyle='-', color=colors[i % len(colors)], label=label)
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^{2}\xi(s)$')
    ax.set_title('ξ(s)')
    ax.legend(loc='upper right')
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
# MONOPOLE FUNCTIONS
# ================================
def compute_monopole_from_xi_s_mu(xi, mu_edges):
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    xi0 = np.trapz(xi, x=mu_centers, axis=1)
    return xi0

def plot_monopoles_combined(monopoles_list, labels, output_folder=None, filename='xi0_combined.png'):
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.tab10.colors
    if np.any(monopoles_list[0][0] > 100):
        ax.axvline(105, color='k', linestyle=':', label='BAO scale')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    for i, ((s_centers, xi0), label) in enumerate(zip(monopoles_list, labels)):
        ax.plot(s_centers, xi0 * s_centers**2, marker='o', linestyle='-', color=colors[i % len(colors)], label=label)
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^{2}\xi_0(s)$')
    ax.set_title('Monopoles ξ₀(s)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        outpath = os.path.join(output_folder, filename)
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"Saved combined monopole plot to {outpath}")
    else:
        plt.show()
    plt.close(fig)

def plot_monopole_to_xi_s_ratio(monopoles_list, xi_s_list, labels_list, output_folder):
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.tab10.colors
    ax.axvline(105, color='k', linestyle=':', label='BAO scale')
    ax.axhline(1, color='gray', linestyle='-', linewidth=1)
    for i, ((s_xi_s, xi_s), (s_xi0, xi0), label) in enumerate(zip(xi_s_list, monopoles_list, labels_list)):
        ratio = xi0 / xi_s
        ax.plot(s_xi_s, ratio, marker='o', linestyle='-', color=colors[i % len(colors)], label=label)
    ax.set_ylim(.8, 1.2)
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$\xi_0(s) / \xi(s)$')
    ax.set_title('Monopole to ξ(s) ratio')
    ax.legend(loc='upper right')
    plt.tight_layout()
    outpath = os.path.join(output_folder, 'monopole_to_xi_s_ratio.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"Saved monopole to xi(s) ratio plot to {outpath}")

# ================================
# DISTANCE BIN SPLITTING
# ================================
def split_by_dist_fil_bins(cat):
    values = cat["dist_fil"].values

    if dist_bin_mode == "custom_intervals":
        raise NotImplementedError("custom_intervals not implemented")
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

# ==========================
# Xi(sigma, pi) functions
# ==========================
from scipy.interpolate import RegularGridInterpolator

def xi_s_mu_to_xi_sigma_pi(xi_s_mu, s_edges, mu_edges, sigma_edges, pi_rebin):
    """
    Convert ξ(s, μ) to ξ(σ, π) via interpolation.

    Parameters:
    -----------
    xi_s_mu : 2D array, shape (len(s_centers), len(mu_centers))
        Correlation function on the (s, μ) grid.
    s_edges : 1D array
        Bin edges for s.
    mu_edges : 1D array
        Bin edges for μ.
    sigma_edges : 1D array
        Bin edges for σ (usually identical to s_edges).
    pi_rebin : float
        Bin width for π (used to define π bins).

    Returns:
    --------
    xi_sigma_pi : 2D array, shape (len(sigma_centers), len(pi_centers))
        Correlation function on the (σ, π) grid.
    sigma_edges : 1D array
        Input sigma_edges (unchanged).
    max_pimax : float
        Maximum π value covered by the π bins.
    """
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

    # Build interpolator
    interp = RegularGridInterpolator(
        (s_centers, mu_centers), xi_s_mu,
        bounds_error=False, fill_value=0.0
    )

    # σ grid
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    max_pimax = sigma_edges[-1]               # largest π we consider
    pi_edges = np.arange(0, max_pimax + pi_rebin, pi_rebin)
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])

    # Create meshgrid of (σ, π) points
    S, P = np.meshgrid(sigma_centers, pi_centers, indexing='ij')
    # Convert to (s, μ)
    s = np.sqrt(S**2 + P**2)
    mu = P / s
    mu[~np.isfinite(mu)] = 0.0

    points = np.stack([s.ravel(), mu.ravel()], axis=-1)
    xi_interp = interp(points).reshape(s.shape)

    return xi_interp, sigma_edges, max_pimax

def plot_xi_sigma_pi_from_xi_s_mu(xi_s_mu, s_edges, mu_edges, sigma_edges, pi_rebin,
                                  title=None, output_folder=None, plotname="xi_sigma_pi.png",
                                  min_sep=0.0, vmin_global=None, vmax_global=None):
    """
    Convert ξ(s, μ) to ξ(σ, π) and plot it with contours (if valid levels).
    """
    # Convert first
    xi, sigma_edges, max_pimax = xi_s_mu_to_xi_sigma_pi(
        xi_s_mu, s_edges, mu_edges, sigma_edges, pi_rebin
    )

    # Build π edges
    pi_edges = np.arange(0, max_pimax + pi_rebin, pi_rebin)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(sigma_edges, pi_edges)
    C = xi.T   # transpose because xi is (σ, π) but we want (π, σ) for meshgrid

    # Determine color scale
    linthresh = 0.001
    if vmin_global is not None and vmax_global is not None:
        vmin, vmax = vmin_global, vmax_global
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

    # Add contours only if we have valid data in the region
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])
    Xc, Yc = np.meshgrid(sigma_centers, pi_centers)

    # Region of interest (e.g., 50–150 Mpc/h in both axes)
    mask_region = (Xc >= 50) & (Xc <= 150) & (Yc >= 50) & (Yc <= 150)
    values_region = C[mask_region]
    if len(values_region) > 0:
        levels = np.percentile(values_region, [50, 70, 90])
        # Ensure levels are strictly increasing
        if len(np.unique(levels)) == len(levels) and np.all(np.diff(levels) > 0):
            ax.contour(Xc, Yc, C, levels=levels, colors='k', linewidths=1.5)
        else:
            print(f"Warning: Contour levels not strictly increasing for {plotname}. Skipping contour.")
    else:
        print(f"Warning: No data in region for {plotname}. Skipping contour.")

    # Plot BAO circle
    theta = np.linspace(0, np.pi/2, 100)  # only first quadrant because σ,π ≥ 0
    sigma_circle = 105 * np.cos(theta)
    pi_circle = 105 * np.sin(theta)
    ax.plot(sigma_circle, pi_circle, 'w--', linewidth=2, alpha=.8, \
            label=r'$s = {}$ Mpc/h'.format(int(105)))
    ax.legend(loc='upper right')

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
# MAIN PROCEDURE
# ================================
def main():
    print("Analyzing BOX.")

    # Load and select sample
    cat_full = load_catalog()
    cat = select_sample(cat_full)
    print(f"Selected {len(cat)} galaxies.")

    # Plot magnitude histogram
    plot_magnitude_hist(cat_full)

    # Split into distance bins
    bins, labels, _ = split_by_dist_fil_bins(cat)
    print(f"Split into {len(bins)} bins:")
    print("\n".join([f"  Bin {i}: {len(b)} galaxies, {label}" for i, (b, label) in enumerate(zip(bins, labels))]))
    print(f"Distance bin mode: {dist_bin_mode}")
    print(f"Mode parameters: {dist_bin_percentile_intervals if dist_bin_mode == 'percentile_intervals' else (nbins_dist if dist_bin_mode in ['percentile', 'equal_width'] else dist_bin_edges if dist_bin_mode == 'fixed' else 'N/A')}")

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    plot_bin_data(cat, label="Full Sample",
                              plotname=f"../plots/{folderName}/bin_full_data_randoms.png")
    for i, (gxs, lab) in enumerate(zip(bins, labels)):
        plot_bin_data(gxs, label=lab,
                                  plotname=f"../plots/{folderName}/bin_{i}_data_randoms.png")


    # ------------------------------------------------------------------
    # First pass: compute ξ(s, μ) for all samples
    # ------------------------------------------------------------------
    print("\nFirst pass: computing ξ(s, μ) for all samples...")
    all_results = []

    # Base parameters for paircounts file naming
    base_params = {
        'mag_max': mag_max,
        'min_sep_2d': min_sep_2d,
        'max_sep_2d': max_sep_2d,
        'bin_size_2d': bin_size_2d,
        'dist_bin_mode': dist_bin_mode,
    }

    #------------------
    # Full sample
    #------------------
    params_full = base_params.copy()
    params_full['dist_bin_mode'] = 'full'
    paircounts_file_full = get_paircounts_filename("full", params_full)

    xi_full, s_bins, mu_bins = compute_xi_s_mu(
        cat["x"].values, cat["y"].values, cat["z"].values,
        min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
        paircounts_file=paircounts_file_full,
        force_recompute=force_recompute_full
    )
    all_results.append( (xi_full, s_bins, mu_bins,
                         rf"$\xi(s, \mu)$ Full Sample", "xi_s_mu_full.png") )

    # Monopole for full sample
    xi0_full = compute_monopole_from_xi_s_mu(xi_full, mu_bins)
    s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])
    monopoles_list = [(s_centers, xi0_full)]
    labels_list = ['Full Sample']
    monopole_file = os.path.join(monopoles_dir, f"monopole_full_{params_full['dist_bin_mode']}.npz")
    np.savez(monopole_file, s=s_centers, xi0=xi0_full)

    # Xi(s) for full sample
    paircounts_file_full_xi_s = get_paircounts_filename("full_xi_s", params_full)
    xi_s_full, s_bins_xi_s = compute_xi_s(
        cat["x"].values, cat["y"].values, cat["z"].values,
        min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
        paircounts_file=paircounts_file_full_xi_s,
        force_recompute=force_recompute_full
    )
    s_centers_xi_s = 0.5 * (s_bins_xi_s[:-1] + s_bins_xi_s[1:])
    xi_s_list = [(s_centers_xi_s, xi_s_full)]
    xi_s_file = os.path.join(paircounts_dir, f"xi_s_full_{params_full['dist_bin_mode']}.npz")
    np.savez(xi_s_file, s=s_centers_xi_s, xi_s=xi_s_full)

    # Plot ξ(σ, π) for full sample
    sigma_bins = np.linspace(min_sep_2d, max_sep_2d, int((max_sep_2d - min_sep_2d) / bin_size_2d) + 1)
    plot_xi_sigma_pi_from_xi_s_mu(
        xi_full, s_bins, mu_bins, sigma_bins, pi_rebin,
        title=rf"$\xi(\sigma,\pi)$ Full Sample",
        output_folder=output_folder,
        plotname="xi_sigma_pi_full.png",
        min_sep=min_sep_2d,
        vmin_global=None,  # auto‑scale; or you can compute global limits as before
        vmax_global=None
    )

    # ------------------------
    # Subsamples
    # ------------------------
    for i, (gxs, lab) in enumerate(zip(bins, labels)):
        params_bin = base_params.copy()
        params_bin['dist_bin_mode'] = dist_bin_mode
        paircounts_file_bin = get_paircounts_filename(f"bin{i}", params_bin)

        dfil_bin_metadata = {
            'dfil_min': gxs["dist_fil"].min(),
            'dfil_max': gxs["dist_fil"].max(),
        }

        xi_bin, s_bins_bin, mu_bins_bin = compute_xi_s_mu(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
            paircounts_file=paircounts_file_bin,
            force_recompute=force_recompute_bin,
            dfil_bin_metadata=dfil_bin_metadata
        )
        all_results.append( (xi_bin, s_bins_bin, mu_bins_bin,
                             rf"$\xi(s, \mu)$ {lab}", f"xi_s_mu_bin{i}.png") )

        # Monopole for this bin
        xi0_bin = compute_monopole_from_xi_s_mu(xi_bin, mu_bins_bin)
        s_centers_bin = 0.5 * (s_bins_bin[:-1] + s_bins_bin[1:])
        monopoles_list.append((s_centers_bin, xi0_bin))
        labels_list.append(lab)
        monopole_file = os.path.join(monopoles_dir, f"monopole_bin{i}_{params_bin['dist_bin_mode']}.npz")
        np.savez(monopole_file, s=s_centers_bin, xi0=xi0_bin, **dfil_bin_metadata)

        # Xi(s) for this bin
        paircounts_file_bin_xi_s = get_paircounts_filename(f"bin{i}_xi_s", params_bin)
        xi_s_bin, s_bins_xi_s_bin = compute_xi_s(
            gxs["x"].values, gxs["y"].values, gxs["z"].values,
            min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
            paircounts_file=paircounts_file_bin_xi_s,
            force_recompute=force_recompute_bin,
            dfil_bin_metadata=dfil_bin_metadata
        )
        s_centers_xi_s_bin = 0.5 * (s_bins_xi_s_bin[:-1] + s_bins_xi_s_bin[1:])
        xi_s_list.append((s_centers_xi_s_bin, xi_s_bin))
        xi_s_file = os.path.join(paircounts_dir, f"xi_s_bin{i}_{params_bin['dist_bin_mode']}.npz")
        np.savez(xi_s_file, s=s_centers_xi_s_bin, xi_s=xi_s_bin, **dfil_bin_metadata)

        # Plot ξ(σ, π) for this bin
        plot_xi_sigma_pi_from_xi_s_mu(
            xi_bin, s_bins, mu_bins, sigma_bins, pi_rebin,
            title=rf"$\xi(\sigma,\pi)$ Bin ({lab})",
            output_folder=output_folder,
            plotname=f"xi_sigma_pi_bin{i}.png",
            min_sep=min_sep_2d,
            vmin_global=None,  # auto‑scale; or you can compute global limits as before
            vmax_global=None
        )
    # ------------------------------------------------------------------
    # Determine global color limits from all xi arrays
    # ------------------------------------------------------------------
    all_xi_flat = np.concatenate([xi.ravel() for xi, _, _, _, _ in all_results])
    vmin_global = np.percentile(all_xi_flat, 1)
    vmax_global = np.percentile(all_xi_flat, 99)
    if vmin_global >= 0:
        vmin_global = -vmax_global / 2
    if vmax_global <= 0:
        vmax_global = -vmin_global / 2
    print(f"Global color limits: vmin={vmin_global:.3f}, vmax={vmax_global:.3f}")

    # ------------------------------------------------------------------
    # Second pass: plot all ξ(s, μ) with fixed color scale and contours
    # ------------------------------------------------------------------
    print("\nSecond pass: plotting with fixed color scale...")
    # log array from -0.01 to 2
    contour_levels = np.concatenate(([-0.01, -0.005, -0.001], np.logspace(-3, 0, 10)))


    for xi, s_edges, mu_edges, title, plotname in all_results:
        plot_xi_s_mu(xi, s_edges, mu_edges,
                     title=title, output_folder=output_folder, plotname=plotname,
                     min_sep=min_sep_2d,
                     vmin_global=vmin_global, vmax_global=vmax_global,
                     contour_levels=contour_levels, contour_colors='black')

    # ------------------------------------------------------------------
    # Combined monopole plot
    # ------------------------------------------------------------------
    plot_monopoles_combined(monopoles_list, labels_list,
                            output_folder=output_folder,
                            filename='xi0_combined.png')
    
    plot_xi_s_combined(xi_s_list, labels_list,
                            output_folder=output_folder,
                            filename='xi_s_combined.png')
    
    # Plot monopole to xi_s ratios
    plot_monopole_to_xi_s_ratio(monopoles_list, xi_s_list, labels_list, output_folder)



    print("All done.")

if __name__ == "__main__":
    main()