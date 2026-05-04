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
                    dfil_bin_metadata=None, nthreads=None, volume=None):
    """
    Compute ξ(s, μ) using analytic random pairs.

    Parameters
    ----------
    x, y, z : array_like
        Coordinates of tracers (in Mpc/h).
    ...
    volume : float, optional
        Effective survey volume for the analytic random pairs.
        If None, the full cubic volume boxsize**3 is used.
        For jackknife subsets this should be the volume of the remaining
        region (boxsize**3 minus the removed sub‑volume).
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
    V = volume if volume is not None else boxsize**3   # <-- use effective volume
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

    #print(f"DD pairs: {np.sum(H_dd):.3e}, RR pairs: {np.sum(RR):.3e}, DD/RR: {np.sum(H_dd)/np.sum(RR):.6f}")
    return xi, s_bins, mu_bins

def compute_xi_s(x, y, z, min_sep, max_sep, bin_size,
                 boxsize=1000.0, paircounts_filename=None,
                 force_recompute=False, dfil_bin_metadata=None,
                 nthreads=None, volume=None):
    """
    Compute ξ(s) independently from scratch, using analytic randoms.
    Saves the result to paircounts_filename if provided.

    Parameters
    ----------
    ...
    volume : float, optional
        Effective survey volume. If None, uses boxsize**3.
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
    V = volume if volume is not None else boxsize**3
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
    xi0 = np.trapezoid(xi, dx=dmu, axis=1)
    return xi0


def compute_jackknife_monopole_covariance(
    x, y, z,
    min_sep, max_sep, bin_size,
    boxsize=1000.0,
    n_sub_per_side=5,
    nthreads=None,
):
    """
    Compute the jackknife covariance matrix of the monopole ξ₀(s)
    using spatial sub‑volumes of a periodic simulation box.

    Returns
    -------
    s_centres : 1D array
        Central separations of the monopole bins.
    xi0_full : 1D array
        Monopole of the full sample (the unbiased estimate).
    cov : 2D array (n_bins × n_bins)
        Jackknife covariance matrix of xi0_full.
    """
    if nthreads is None:
        nthreads = min(multiprocessing.cpu_count() - 4, 16)

    # 1. Full sample monopole (use full volume)
    xi_full, s_bins, mu_bins = compute_xi_s_mu(
        x, y, z,
        min_sep, max_sep, bin_size,
        boxsize=boxsize, nthreads=nthreads,
        volume=boxsize**3
    )
    xi0_full = compute_monopole_from_xi_s_mu(xi_full, mu_bins)
    s_centres = 0.5 * (s_bins[:-1] + s_bins[1:])
    n_bins = len(s_centres)

    # 2. Assign particles to sub‑volumes
    sub_edges = np.linspace(0, boxsize, n_sub_per_side + 1)
    i_idx = np.clip(np.searchsorted(sub_edges, x, side='right') - 1, 0, n_sub_per_side - 1)
    j_idx = np.clip(np.searchsorted(sub_edges, y, side='right') - 1, 0, n_sub_per_side - 1)
    k_idx = np.clip(np.searchsorted(sub_edges, z, side='right') - 1, 0, n_sub_per_side - 1)
    particle_sub = i_idx * n_sub_per_side**2 + j_idx * n_sub_per_side + k_idx

    n_sub_total = n_sub_per_side**3
    sub_vol = (boxsize / n_sub_per_side)**3
    volume_rem = boxsize**3 - sub_vol

    # 3. Compute leave‑one‑out monopoles
    xi_sub_all = np.empty((n_sub_total, n_bins))
    for sub_id in range(n_sub_total):
        mask = (particle_sub != sub_id)
        if np.sum(mask) < 10:
            print(f"Warning: sub‑volume {sub_id} contains almost all particles; using full sample as fallback")
            xi_sub_all[sub_id] = xi0_full
            continue

        print(f"Processing sub‑volume {sub_id+1}/{n_sub_total}")

        x_sub, y_sub, z_sub = x[mask], y[mask], z[mask]
        xi_sub, _, mu_bins_sub = compute_xi_s_mu(
            x_sub, y_sub, z_sub,
            min_sep, max_sep, bin_size,
            boxsize=boxsize, nthreads=nthreads,
            volume=volume_rem,             # corrected volume
            force_recompute=True,
            paircounts_filename=None
        )
        xi_sub_all[sub_id] = compute_monopole_from_xi_s_mu(xi_sub, mu_bins_sub)

    # 4. Jackknife covariance (delete‑one formula)
    xi_bar = np.mean(xi_sub_all, axis=0)
    diff = xi_sub_all - xi_bar   # (n_sub, n_bins)
    # Standard jackknife covariance: (N-1)/N * sum (T_i - 𝔼[T])²
    cov = (n_sub_total - 1) / n_sub_total * np.einsum('ij,ik->jk', diff, diff)

    return s_centres, xi0_full, cov