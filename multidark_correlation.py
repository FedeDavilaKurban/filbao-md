import numpy as np
import multiprocessing
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.DD import DD
import os

def get_save_filename(bin_desc, params, out_folder, filetype='monopole'):
    """
    Generate a filename for monopoles / xi_s / paircounts in the new style.

    Example outputs:
    monopole: mag=-21.2_sep=1.0-150.0_binsep=2.0_dilute=0.5_sample=dfil_in_0.0-4.0,_log10rho3_in_-2.2-2.5.npz
    xi_s:     mag=-21.2_sep=1.0-150.0_binsep=2.0_dilute=0.5_sample=dfil_in_0.0-4.0,_log10rho3_in_-2.2-2.5_xi_s.npz
    paircounts: mag=-21.2_sep=1.0-150.0_binsep=2.0_dilute=0.5_sample=dfil_in_0.0-4.0,_log10rho3_in_-2.2-2.5_paircounts.npz
    """
    mag = params.get('mag_max')
    min_sep = params.get('min_sep')
    max_sep = params.get('max_sep')
    bin_size = params.get('bin_size')
    dilute = params.get('test_dilute')

    filename = f"mag={mag:.1f}_sep={min_sep}-{max_sep}_binsep={bin_size:.1f}"

    if dilute is not None and dilute < 1.0:
        filename += f"_dilute={dilute}"

    filename += f"_{bin_desc}"

    # Add filetype suffix except for monopoles (default)
    suffix_map = {
        'monopole': '',
        'xi_s': '_xi_s',
        'paircounts': '_paircounts'
    }
    suffix = suffix_map.get(filetype, '')
    filename += f"{suffix}.npz"

    return os.path.join(out_folder, filename)

def compute_xi_s_mu(x, y, z, min_sep, max_sep, bin_size,
                    boxsize=1000.0, paircounts_filename=None, force_recompute=False,
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
    if paircounts_filename and os.path.exists(paircounts_filename) and not force_recompute:
        data = np.load(paircounts_filename)
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
        if paircounts_filename:
            os.makedirs(os.path.dirname(paircounts_filename), exist_ok=True)
            save_dict = {'s_bins': s_bins, 'mu_bins': mu_bins, 'H_dd': H_dd}
            if dfil_bin_metadata:
                save_dict.update(dfil_bin_metadata)
            np.savez(paircounts_filename, **save_dict)

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
                 boxsize=1000.0, paircounts_filename=None,
                 force_recompute=False, dfil_bin_metadata=None,
                 nthreads=None):
    """
    Compute ξ(s) independently from scratch, using analytic randoms.
    Saves the result to paircounts_filename if provided.
    """
    if nthreads is None:
        nthreads = min(multiprocessing.cpu_count() - 4, 16)

    # s bins
    nbins_s = int((max_sep - min_sep) / bin_size)
    s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)

    # Check if xi_s already exists
    if paircounts_filename and os.path.exists(paircounts_filename) and not force_recompute:
        data = np.load(paircounts_filename)
        if 'xi_s' in data:
            xi_s = data['xi_s']
            s_bins_loaded = data['s']
            print(f"Loaded xi_s from {paircounts_filename}")
            return xi_s, s_bins_loaded

    # Compute 1D DD counts
    dd_counts = DD(autocorr=1, nthreads=nthreads, binfile=s_bins,
                   X1=x, Y1=y, Z1=z,
                   periodic=True, boxsize=boxsize, verbose=False)
    H_dd = dd_counts['npairs'].astype(np.float64)

    # Analytic RR counts
    N = len(x)
    V = boxsize**3
    smin = s_bins[:-1]
    smax = s_bins[1:]
    shell_vol = (4.0/3.0) * np.pi * (smax**3 - smin**3)
    RR_s = shell_vol * (N * (N - 1) / V)

    # xi(s)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi_s = H_dd / RR_s - 1.0
        xi_s[RR_s == 0] = np.nan

    # Save
    if paircounts_filename:
        os.makedirs(os.path.dirname(paircounts_filename), exist_ok=True)
        save_dict = {'s': 0.5*(smin+smax), 'xi_s': xi_s}
        if dfil_bin_metadata:
            save_dict.update(dfil_bin_metadata)
        np.savez(paircounts_filename, **save_dict)

    print(f"DD pairs: {np.sum(H_dd):.3e}, RR pairs: {np.sum(RR_s):.3e}, DD/RR: {np.sum(H_dd)/np.sum(RR_s):.6f}")
    return xi_s, 0.5*(smin+smax)

def compute_monopole_from_xi_s_mu(xi, mu_edges):
    """Integrate ξ(s, μ) over μ to get monopole ξ₀(s)."""
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = mu_centers[1] - mu_centers[0] if len(mu_centers) > 1 else 1.0
    xi0 = np.trapz(xi, dx=dmu, axis=1)
    return xi0