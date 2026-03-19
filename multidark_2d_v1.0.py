"""
v3.0 - Full 2D correlation pipeline

Features:
- Generate galaxy + random catalogs
- Compute 2D correlation ξ(σ,π) using Corrfunc
- Compute monopole ξ0(s)
- Diagnostic plots: RA/Dec, redshift, filament distance bins
- Save pair counts and monopoles
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import gaussian_kde
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from Corrfunc.mocks import DDsmu_mocks # for 2D ξ

# ---------------------------
# PARAMETERS
# ---------------------------

lightcone_filename = '../data/lightcone_real_and_rsd_withfil.csv'
test_dilute = 0.25
h = 0.6774
zmin, zmax = 0.07, 0.2
mag_max = -20.5
gr_min = 0

# Random catalog
nrand_mult = 15
ran_method = 'poly'
deg = 1
apply_dec_weights = True

# 2D correlation function
s_bins = np.arange(1, 151, 3)  # bins in Mpc/h
mu_bins = np.linspace(0, 1, 21)

# Filament distance binning
nbins_dist = 4

# Output folders
folderName = f'XISIGMAPI_z{zmin:.2f}-{zmax:.2f}_mag{mag_max:.1f}_gr{gr_min}_nrand{nrand_mult}'
output_folder = f"../plots/{folderName}/"
paircounts_dir = "../data/pair_counts/"
monopoles_dir = "../data/monopoles/"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(paircounts_dir, exist_ok=True)
os.makedirs(monopoles_dir, exist_ok=True)

# Cosmology
cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3089)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def safe_trapz(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_figure(fig, path, dpi=300):
    ensure_dir_exists(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# RANDOM CATALOG GENERATION
# ---------------------------

def generate_random_red(redshift, nrand, method='poly', deg=1):
    if method == 'poly':
        hist, bins = np.histogram(redshift, bins=50, density=True)
        centers = 0.5*(bins[:-1]+bins[1:])
        from numpy.polynomial.polynomial import Polynomial
        poly = Polynomial.fit(centers, hist, deg=deg)
        z_vals = np.linspace(redshift.min(), redshift.max(), 10000)
        pdf = np.clip(poly(z_vals), 0, None)
        pdf /= safe_trapz(pdf, z_vals)
        cdf = np.cumsum(pdf) * (z_vals[1]-z_vals[0])
        cdf /= cdf[-1]
        interp_func = interp1d(cdf, z_vals, bounds_error=False, fill_value=(z_vals[0], z_vals[-1]))
        return interp_func(np.random.uniform(0,1,nrand))
    elif method == 'random_choice':
        return np.random.choice(redshift, nrand)
    else:
        raise ValueError(f"Unknown method {method}")

def compute_dec_weights(data_dec, rand_dec, alpha=1.0, method="auto", nbins=40, spline_s=0.5):
    epsilon = 1e-10
    if method == "auto":
        method = "kde" if len(rand_dec) < 1e6 else "spline"

    if method == "kde":
        kde_data = gaussian_kde(data_dec)
        kde_rand = gaussian_kde(rand_dec)
        grid = np.linspace(min(data_dec.min(), rand_dec.min()), max(data_dec.max(), rand_dec.max()), 300)
        ratio = (kde_data(grid)+epsilon)/(kde_rand(grid)+epsilon)
        weights = np.interp(rand_dec, grid, ratio)
    elif method == "spline":
        hist_data, edges = np.histogram(data_dec, bins=nbins)
        hist_rand, _ = np.histogram(rand_dec, bins=edges)
        centers = 0.5*(edges[:-1]+edges[1:])
        ratio = (hist_data+epsilon)/(hist_rand+epsilon)
        spline = UnivariateSpline(centers, ratio, s=spline_s)
        weights = np.clip(spline(rand_dec), 0.01, None)
    else:
        raise ValueError("Invalid method")
    weights = 1 + alpha*(weights-1)
    weights /= np.mean(weights)
    return weights

def generate_random_catalog(cat_gal, nrand_mult=15, cosmo=None, ran_method='poly', deg=1, apply_dec_weights=True):
    np.random.seed(42)
    N_rand = len(cat_gal) * nrand_mult
    ra_min, ra_max = cat_gal['ra'].min(), cat_gal['ra'].max()
    dec_min, dec_max = cat_gal['dec'].min(), cat_gal['dec'].max()

    ra_rand = np.random.uniform(ra_min, ra_max, N_rand)
    sin_dec_min, sin_dec_max = np.sin(np.radians(dec_min)), np.sin(np.radians(dec_max))
    dec_rand = np.degrees(np.arcsin(np.random.uniform(sin_dec_min, sin_dec_max, N_rand)))

    red_rand = generate_random_red(cat_gal['red'].values, N_rand, method=ran_method, deg=deg)

    rand_cat = pd.DataFrame({"ra": ra_rand, "dec": dec_rand, "red": red_rand})

    chi = cosmo.comoving_distance(rand_cat['red'].values).value * h
    theta = np.radians(90.0 - rand_cat['dec'].values)
    phi = np.radians(rand_cat['ra'].values)
    rand_cat['x'] = chi * np.sin(theta) * np.cos(phi)
    rand_cat['y'] = chi * np.sin(theta) * np.sin(phi)
    rand_cat['z'] = chi * np.cos(theta)

    hist, bins = np.histogram(cat_gal['mag_abs_r'], bins=50, density=True)
    cdf = np.cumsum(hist*np.diff(bins))
    cdf /= cdf[-1]
    interp_mag = interp1d(cdf, 0.5*(bins[:-1]+bins[1:]), bounds_error=False, fill_value=(bins[0], bins[-1]))
    rand_cat['mag_abs_r'] = interp_mag(np.random.uniform(0,1,len(rand_cat)))

    rand_cat = rand_cat[rand_cat['mag_abs_r'] < mag_max].reset_index(drop=True)

    if apply_dec_weights:
        rand_cat['weight'] = compute_dec_weights(cat_gal['dec'].values, rand_cat['dec'].values)
    else:
        rand_cat['weight'] = 1.0

    return rand_cat

# ---------------------------
# CATALOG LOADING & SELECTION
# ---------------------------

def load_catalog(lightcone_filename, dilute=1.0):
    cat = pd.read_csv(lightcone_filename)
    if dilute < 1.0:
        cat = cat.sample(frac=dilute, random_state=42).reset_index(drop=True)
    cat.rename(columns={'z_obs':'red','mag_r':'mag_abs_r','dfil':'dist_fil','ra_deg':'ra','dec_deg':'dec'}, inplace=True)
    return cat

def select_sample(cat):
    cat_z = cat[(cat['red']>=zmin) & (cat['red']<=zmax)]
    cat_z_mag = cat_z[cat_z['mag_abs_r']<mag_max].copy()
    cat_z_mag.loc[:, "r"] = cosmo.comoving_distance(cat_z_mag['red'].values).value*h
    return cat_z, cat_z_mag

def split_by_dist_fil_bins(cat_z_mag):
    values = cat_z_mag['dist_fil'].values
    percentiles = np.linspace(0,100, nbins_dist+1)
    edges = np.percentile(values, percentiles)
    bins = []
    labels = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mask = (values>=lo)&(values<hi) if i<len(edges)-2 else (values>=lo)&(values<=hi)
        bins.append(cat_z_mag.loc[mask].copy())
        labels.append(f"${lo:.1f}<r_{{fil}}<={hi:.1f}$")
    return bins, labels

# ---------------------------
# DIAGNOSTIC PLOTS
# ---------------------------

def plot_redshift_k(cat):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist2d(cat['red'], cat['mag_abs_r'], bins=40, cmap='Blues', norm=LogNorm())
    ax.axvline(zmin,color='k',linestyle=':')
    ax.axvline(zmax,color='k',linestyle=':')
    ax.axhline(mag_max,color='k',linestyle=':')
    ax.invert_yaxis()
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Mag_abs_r")
    save_figure(fig, os.path.join(output_folder,'redshift_magnitude.png'))

def plot_radec_distribution(cat, randoms, label=None):
    fig, axes = plt.subplots(2,1,figsize=(8,10))
    axes[0].hist(cat['ra'], bins=40, density=True, histtype='stepfilled', color='C0', alpha=0.8, label='Galaxies')
    axes[0].hist(randoms['ra'], bins=40, density=True, histtype='step', color='k', lw=1.5, label='Randoms')
    axes[0].set_xlabel('RA')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[1].hist(cat['dec'], bins=40, density=True, histtype='stepfilled', color='C0', alpha=0.8, label='Galaxies')
    axes[1].hist(randoms['dec'], bins=40, density=True, histtype='step', color='k', lw=1.5, label='Randoms')
    if 'weight' in randoms.columns:
        axes[1].hist(randoms['dec'], bins=40, density=True, weights=randoms['weight'], histtype='step', color='k', lw=1.5, linestyle='--', label='Weighted Randoms')
    axes[1].set_xlabel('Dec')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    title = label if label else 'Full sample'
    fig.suptitle(title)
    save_figure(fig, os.path.join(output_folder,'radec_distribution.png'))

# ---------------------------
# RESULTS PLOTS
# ---------------------------

def plot_xi_sigma_pi(ddsmu_result, s_bins, mu_bins, output_path, title="ξ(σ,π)"):
    # Extract xi values and r, mu arrays
    xi = np.array([d['xi'] for d in ddsmu_result])
    s = np.array([d['r'] for d in ddsmu_result])
    mu = np.linspace(0, 1, xi.shape[1])  # assuming nmu_bins

    # Project onto s_perp, s_parallel
    s_perp = s * np.sqrt(1 - mu[None, :]**2)
    s_par = s[:, None] * mu[None, :]
    
    fig, ax = plt.subplots(figsize=(7,6))
    pcm = ax.pcolormesh(s_perp, s_par, xi, shading='auto', cmap='RdBu_r', vmin=-0.05, vmax=0.2)
    fig.colorbar(pcm, ax=ax, label="ξ(σ,π)")
    ax.set_xlabel(r'σ [Mpc/h]')
    ax.set_ylabel(r'π [Mpc/h]')
    ax.set_title(title)
    save_figure(fig, output_path)


def plot_monopoles(monopole_files, labels, output_path):
    fig, ax = plt.subplots(figsize=(7,6))
    for f, lab in zip(monopole_files, labels):
        arr = np.load(f)
        s, xi0 = arr[0], arr[1]
        ax.plot(s, xi0*s**2, label=lab)  # s^2 xi for BAO bump visualization
    ax.set_xlabel(r's [Mpc/h]')
    ax.set_ylabel(r's² ξ₀(s) [Mpc/h]²')
    ax.legend()
    ax.grid(True)
    save_figure(fig, output_path)

# ---------------------------
# CORRELATION FUNCTION
# ---------------------------

def compute_xi_sigma_pi(cat_gal, cat_rand, s_bins, mu_bins, output_prefix):
    X = cat_gal['x'].values
    Y = cat_gal['y'].values
    Z = cat_gal['z'].values
    W = np.ones_like(X)
    
    XR = cat_rand['x'].values
    YR = cat_rand['y'].values
    ZR = cat_rand['z'].values
    WR = cat_rand['weight'].values

    results = DDsmu_mocks(autocorr=1, nthreads=4,
                           binfile=s_bins, 
                           mu_max=1.0, nmu_bins=len(mu_bins)-1,
                           X1=X, Y1=Y, Z1=Z, W1=W,
                           X2=XR, Y2=YR, Z2=ZR, W2=WR)

    # Save raw counts
    np.savez(os.path.join(paircounts_dir,f'{output_prefix}_ddsmu.npz'), results=results)
    return results

def compute_monopole(ddsmu_result, mu_bins):
    xi_mu = np.array([d['xi'] for d in ddsmu_result])
    xi0 = np.mean(xi_mu, axis=1)
    s = np.array([d['r'] for d in ddsmu_result])
    return s, xi0

# ---------------------------
# MAIN
# ---------------------------

def main():
    print(f"Loading catalog: {lightcone_filename}")
    cat_full = load_catalog(lightcone_filename, dilute=test_dilute)
    cat_z, cat_z_mag = select_sample(cat_full)

    plot_redshift_k(cat_full)

    bins, labels = split_by_dist_fil_bins(cat_z_mag)

    print("Generating random catalog...")
    rand_cat = generate_random_catalog(cat_z_mag, nrand_mult=nrand_mult, cosmo=cosmo,
                                       ran_method=ran_method, deg=deg, apply_dec_weights=apply_dec_weights)

    plot_radec_distribution(cat_z_mag, rand_cat)

    print(f"Random catalog generated with {len(rand_cat)} points")

    # Full sample correlation
    print("Computing 2D correlation for full sample...")
    xi2d_full = compute_xi_sigma_pi(cat_z_mag, rand_cat, s_bins, mu_bins, 'full_sample')
    s, xi0 = compute_monopole(xi2d_full, mu_bins)
    np.save(os.path.join(monopoles_dir, 'xi0_full_sample.npy'), np.vstack([s, xi0]))

    # Plot 2D xi
    plot_xi_sigma_pi(xi2d_full, s_bins, mu_bins, os.path.join(output_folder,'xi_sigma_pi_full.png'),
                     title='Full Sample ξ(σ,π)')

    # Prepare for monopole plotting
    monopole_files = [os.path.join(monopoles_dir, 'xi0_full_sample.npy')]
    monopole_labels = ['Full Sample']

    # Correlation per filament bin
    for ibin, (cat_bin, label) in enumerate(zip(bins, labels)):
        print(f"Computing correlation for filament bin {ibin+1}: {label}")
        xi2d_bin = compute_xi_sigma_pi(cat_bin, rand_cat, s_bins, mu_bins, f'bin_{ibin+1}')
        s_bin, xi0_bin = compute_monopole(xi2d_bin, mu_bins)
        fname = os.path.join(monopoles_dir, f'xi0_bin{ibin+1}.npy')
        np.save(fname, np.vstack([s_bin, xi0_bin]))

        # Plot 2D xi for each bin
        plot_xi_sigma_pi(xi2d_bin, s_bins, mu_bins,
                         os.path.join(output_folder,f'xi_sigma_pi_bin{ibin+1}.png'),
                         title=f'Bin {ibin+1}: {label}')

        monopole_files.append(fname)
        monopole_labels.append(f'Bin {ibin+1}')

    # Plot monopoles all together
    plot_monopoles(monopole_files, monopole_labels, os.path.join(output_folder,'monopoles_all.png'))
    print("All monopole and 2D correlation plots saved.")

if __name__=="__main__":
    main()