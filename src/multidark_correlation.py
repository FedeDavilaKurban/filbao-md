"""
Two-point correlation functions in the periodic box, with Corrfunc pair counts and
analytic randoms (RR = N (N - 1) / V x shell volume, exact for a uniform periodic sample).

    get_save_filename              file names of pair counts, monopoles and xi(s)
    compute_xi_s_mu                xi(s, mu) from DD(s, mu)   -> the monopole used in the fit
    compute_monopole_from_xi_s_mu  xi_0(s) = average of xi(s, mu) over mu
    compute_xi_s                   1D xi(s) from DD(s)       (only for the diagnostic plots)
"""
import os
import multiprocessing
import numpy as np
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.DD import DD


def get_save_filename(bin_desc, params, out_folder, filetype='monopole'):
    """
    mag=<mag>_sep=<min>-<max>_binsep=<bin>_<bin_desc>[_xi_s|_paircounts].npz
    e.g. monopole for the full sample:  mag=-21.2_sep=5.0-150.0_binsep=5.0_full.npz
    """
    filename = (f"mag={params['mag_max']:.1f}_sep={params['min_sep']}-{params['max_sep']}"
                f"_binsep={params['bin_size']:.1f}_{bin_desc}")
    suffix = {'monopole': '', 'xi_s': '_xi_s', 'paircounts': '_paircounts'}.get(filetype, '')
    return os.path.join(out_folder, filename + suffix + ".npz")


def compute_xi_s_mu(x, y, z, min_sep, max_sep, bin_size, boxsize=1000.0,
                    paircounts_filename=None, force_recompute=False, nthreads=None):
    """
    xi(s, mu) = DD / RR - 1 with analytic RR.  s bins of width bin_size, as many mu bins in
    [0, 1] as s bins.  DD is read from paircounts_filename if it exists (unless
    force_recompute), otherwise computed and saved there.
    Returns xi (n_s, n_mu), s_edges, mu_edges.
    """
    if nthreads is None:
        nthreads = min(multiprocessing.cpu_count() - 4, 16)

    nbins_s = int((max_sep - min_sep) / bin_size)
    s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)
    nbins_mu = nbins_s
    mu_max = 1.0
    mu_bins = np.linspace(0.0, mu_max, nbins_mu + 1)

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
            np.savez(paircounts_filename, s_bins=s_bins, mu_bins=mu_bins, H_dd=H_dd)

    # Analytic RR: expected pairs of a uniform sample in each (s, mu) cell
    N = len(x)
    V = boxsize**3
    RR = np.zeros((nbins_s, nbins_mu))
    for i in range(nbins_s):
        vol_shell = (4 * np.pi / 3.0) * (s_bins[i+1]**3 - s_bins[i]**3)
        for j in range(nbins_mu):
            dV = vol_shell * (mu_bins[j+1] - mu_bins[j])
            RR[i, j] = (N * (N - 1) / V) * dV

    with np.errstate(divide='ignore', invalid='ignore'):
        xi = H_dd / RR - 1.0
        xi[RR == 0] = np.nan
    return xi, s_bins, mu_bins


def compute_monopole_from_xi_s_mu(xi):
    """xi_0(s): the average of xi(s, mu) over the (equal-width) mu bins."""
    return xi.mean(axis=1)


def compute_xi_s(x, y, z, min_sep, max_sep, bin_size, boxsize=1000.0,
                 paircounts_filename=None, force_recompute=False, nthreads=None):
    """
    1D xi(s) = DD(s) / RR(s) - 1 with analytic RR, computed directly (not from xi(s, mu)).
    In real space it should equal the monopole; the ratio plot checks this.
    Read from / saved to paircounts_filename.  Returns xi_s, s_centres.
    """
    if nthreads is None:
        nthreads = min(multiprocessing.cpu_count() - 4, 16)

    nbins_s = int((max_sep - min_sep) / bin_size)
    s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)

    if paircounts_filename and os.path.exists(paircounts_filename) and not force_recompute:
        data = np.load(paircounts_filename)
        if 'xi_s' in data:
            print(f"Loaded xi_s from {paircounts_filename}")
            return data['xi_s'], data['s']

    dd_counts = DD(autocorr=1, nthreads=nthreads, binfile=s_bins,
                   X1=x, Y1=y, Z1=z,
                   periodic=True, boxsize=boxsize, verbose=False)
    H_dd = dd_counts['npairs'].astype(np.float64)

    N = len(x)
    V = boxsize**3
    smin, smax = s_bins[:-1], s_bins[1:]
    RR_s = (4.0 / 3.0) * np.pi * (smax**3 - smin**3) * (N * (N - 1) / V)

    with np.errstate(divide='ignore', invalid='ignore'):
        xi_s = H_dd / RR_s - 1.0
        xi_s[RR_s == 0] = np.nan

    s_centres = 0.5 * (smin + smax)
    if paircounts_filename:
        os.makedirs(os.path.dirname(paircounts_filename), exist_ok=True)
        np.savez(paircounts_filename, s=s_centres, xi_s=xi_s)

    print(f"DD pairs: {np.sum(H_dd):.3e}, RR pairs: {np.sum(RR_s):.3e}, DD/RR: {np.sum(H_dd)/np.sum(RR_s):.6f}")
    return xi_s, s_centres
