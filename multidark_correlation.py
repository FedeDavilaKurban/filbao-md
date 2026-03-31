import numpy as np
import multiprocessing
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.DD import DD
import os

def get_paircounts_filename(bin_name, params, base_dir):
    """Build unique filename for saved pair counts."""
    parts = []
    if 'mag_max' in params:
        parts.append(f"mag={params['mag_max']:.1f}")
    if 'min_sep' in params and 'max_sep' in params:
        parts.append(f"sep={params['min_sep']:.1f}-{params['max_sep']:.1f}")
    if 'bin_size' in params:
        parts.append(f"binsep={params['bin_size']:.1f}")
    parts.append(f"sample={bin_name}")
    if 'dist_bin_mode' in params:
        parts.append(f"distbinmode={params['dist_bin_mode']}")
    fname = "_".join(parts)
    return os.path.join(base_dir, fname + ".npz")

def compute_xi_s_mu(x, y, z, min_sep, max_sep, bin_size,
                    boxsize=1000.0, paircounts_file=None, force_recompute=False,
                    dfil_bin_metadata=None, nthreads=None):
    """
    Compute ξ(s, μ) using analytic random pairs.
    Returns (xi, s_bins, mu_bins).
    """
    if nthreads is None:
        nthreads = min(multiprocessing.cpu_count() - 4, 16)

    nbins_s = int((max_sep - min_sep) / bin_size)
    s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)
    nbins_mu = nbins_s * 2
    mu_max = 1.0
    mu_bins = np.linspace(0.0, mu_max, nbins_mu + 1)

    # Load or compute DD
    if paircounts_file and os.path.exists(paircounts_file) and not force_recompute:
        data = np.load(paircounts_file)
        H_dd = data['H_dd']
        s_bins = data['s_bins']
        mu_bins = data['mu_bins']
        nbins_s = len(s_bins) - 1
        nbins_mu = len(mu_bins) - 1
    else:
        dd_counts = DDsmu(autocorr=1, nthreads=nthreads, binfile=s_bins,
                          mu_max=mu_max, nmu_bins=nbins_mu,
                          X1=x, Y1=y, Z1=z,
                          periodic=True, boxsize=boxsize, verbose=False)
        H_dd = dd_counts['npairs'].reshape(nbins_s, nbins_mu).astype(np.float64)
        if paircounts_file:
            os.makedirs(os.path.dirname(paircounts_file), exist_ok=True)
            save_dict = {'s_bins': s_bins, 'mu_bins': mu_bins, 'H_dd': H_dd}
            if dfil_bin_metadata:
                save_dict.update(dfil_bin_metadata)
            np.savez(paircounts_file, **save_dict)

    # Analytic RR
    N = len(x)
    V = boxsize**3
    RR = np.zeros((nbins_s, nbins_mu))
    for i in range(nbins_s):
        s_lo, s_hi = s_bins[i], s_bins[i+1]
        vol_shell = (4 * np.pi / 3.0) * (s_hi**3 - s_lo**3)
        for j in range(nbins_mu):
            mu_lo, mu_hi = mu_bins[j], mu_bins[j+1]
            dmu = mu_hi - mu_lo
            dV = vol_shell * dmu
            RR[i, j] = (N * (N - 1) / V) * dV

    with np.errstate(divide='ignore', invalid='ignore'):
        xi = H_dd / RR - 1.0
        xi[RR == 0] = np.nan

    print(f"DD pairs: {np.sum(H_dd):.3e}, RR pairs: {np.sum(RR):.3e}, DD/RR: {np.sum(H_dd)/np.sum(RR):.6f}")
    return xi, s_bins, mu_bins

def compute_xi_s(x, y, z, min_sep, max_sep, bin_size,
                 boxsize=1000.0, paircounts_file=None, force_recompute=False,
                 dfil_bin_metadata=None, nthreads=None):
    """Compute ξ(s) using analytic random pairs."""
    if nthreads is None:
        nthreads = min(multiprocessing.cpu_count() - 4, 16)

    nbins_s = int((max_sep - min_sep) / bin_size)
    s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)

    if paircounts_file and os.path.exists(paircounts_file) and not force_recompute:
        data = np.load(paircounts_file)
        H_dd = data['H_dd']
        s_bins = data['s_bins']
        nbins_s = len(s_bins) - 1
    else:
        dd_counts = DD(autocorr=1, nthreads=nthreads, binfile=s_bins,
                       X1=x, Y1=y, Z1=z,
                       periodic=True, boxsize=boxsize, verbose=False)
        H_dd = dd_counts['npairs'].astype(np.float64)
        if paircounts_file:
            os.makedirs(os.path.dirname(paircounts_file), exist_ok=True)
            save_dict = {'s_bins': s_bins, 'H_dd': H_dd}
            if dfil_bin_metadata:
                save_dict.update(dfil_bin_metadata)
            np.savez(paircounts_file, **save_dict)

    N = len(x)
    V = boxsize**3
    smin = s_bins[:-1]
    smax = s_bins[1:]
    shell_vol = (4.0/3.0) * np.pi * (smax**3 - smin**3)
    RR_s = shell_vol * (N * (N - 1) / V)

    with np.errstate(divide='ignore', invalid='ignore'):
        xi = H_dd / RR_s - 1.0
        xi[RR_s == 0] = np.nan

    print(f"DD pairs: {np.sum(H_dd):.3e}, RR pairs: {np.sum(RR_s):.3e}, DD/RR: {np.sum(H_dd)/np.sum(RR_s):.6f}")
    return xi, s_bins

def compute_monopole_from_xi_s_mu(xi, mu_edges):
    """Integrate ξ(s, μ) over μ to get monopole ξ₀(s)."""
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = mu_centers[1] - mu_centers[0] if len(mu_centers) > 1 else 1.0
    xi0 = np.trapz(xi, dx=dmu, axis=1)
    return xi0