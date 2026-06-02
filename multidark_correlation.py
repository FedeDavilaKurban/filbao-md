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


# ---------------------------------------------------------------------------
# Jackknife helpers
# ---------------------------------------------------------------------------

def _analytic_rr_2d(s_bins, mu_bins, N, V):
    """
    Analytic RR(s, μ) for N uniform random points in volume V.
    Shape: (nbins_s, nbins_mu).
    """
    nbins_s = len(s_bins) - 1
    nbins_mu = len(mu_bins) - 1
    dmu = (mu_bins[-1] - mu_bins[0]) / nbins_mu          # uniform μ bins
    RR = np.empty((nbins_s, nbins_mu))
    for i in range(nbins_s):
        vol_shell = (4 * np.pi / 3.0) * (s_bins[i + 1]**3 - s_bins[i]**3)
        RR[i, :] = (N * (N - 1) / V) * (vol_shell * dmu)
    return RR


# ---------------------------------------------------------------------------
# Jackknife helpers – shared memory version
# ---------------------------------------------------------------------------

# Global variables for workers (set by initializer)
_jk_x = None
_jk_y = None
_jk_z = None
_jk_particle_sub = None
_jk_s_bins = None
_jk_mu_max = None
_jk_nbins_mu = None
_jk_boxsize = None
_jk_threads = None

def _init_worker(x, y, z, particle_sub, s_bins, mu_max, nbins_mu, boxsize, threads):
    global _jk_x, _jk_y, _jk_z, _jk_particle_sub, _jk_s_bins
    global _jk_mu_max, _jk_nbins_mu, _jk_boxsize, _jk_threads
    _jk_x = x
    _jk_y = y
    _jk_z = z
    _jk_particle_sub = particle_sub
    _jk_s_bins = s_bins
    _jk_mu_max = mu_max
    _jk_nbins_mu = nbins_mu
    _jk_boxsize = boxsize
    _jk_threads = threads

def _jk_worker_shared(sub_id):
    """Worker that accesses global arrays – no data duplication."""
    mask_in = (_jk_particle_sub == sub_id)
    mask_out = ~mask_in

    x_in = _jk_x[mask_in]
    y_in = _jk_y[mask_in]
    z_in = _jk_z[mask_in]
    x_out = _jk_x[mask_out]
    y_out = _jk_y[mask_out]
    z_out = _jk_z[mask_out]

    nbins_s = len(_jk_s_bins) - 1

    if len(x_in) >= 2:
        dd_in = DDsmu(autocorr=1, nthreads=_jk_threads, binfile=_jk_s_bins,
                      mu_max=_jk_mu_max, nmu_bins=_jk_nbins_mu,
                      X1=x_in, Y1=y_in, Z1=z_in,
                      periodic=True, boxsize=_jk_boxsize, verbose=False)
        H_in = dd_in['npairs'].reshape(nbins_s, _jk_nbins_mu).astype(np.float64)
    else:
        H_in = np.zeros((nbins_s, _jk_nbins_mu))

    dd_cross = DDsmu(autocorr=0, nthreads=_jk_threads, binfile=_jk_s_bins,
                     mu_max=_jk_mu_max, nmu_bins=_jk_nbins_mu,
                     X1=x_in, Y1=y_in, Z1=z_in,
                     X2=x_out, Y2=y_out, Z2=z_out,
                     periodic=True, boxsize=_jk_boxsize, verbose=False)
    H_cross = dd_cross['npairs'].reshape(nbins_s, _jk_nbins_mu).astype(np.float64)

    print(f"  Jackknife sub-volume {sub_id+1} done", flush=True)
    return sub_id, H_in, H_cross

def compute_jackknife_monopole_covariance(
    x, y, z,
    min_sep, max_sep, bin_size,
    boxsize=1000.0,
    n_sub_per_side=5,
    nthreads=None,
    n_workers=None,
):
    """
    Compute the jackknife covariance matrix of the monopole ξ₀(s)
    using spatial sub-volumes of a periodic simulation box.

    Optimisation
    ------------
    Instead of re-running DDsmu on the full N − N_k leave-one-out catalog
    for every realisation (O(N²) each), we use the **pair-subtraction trick**:

        DD_loo_k = DD_full − DD_within_k − DD_cross_k

    The two small counts scale as O((N/K)²) and O(N/K · (K−1)N/K) —
    roughly K times cheaper than the naïve approach.  The loop over K
    sub-volumes is then run in parallel across ``n_workers`` processes,
    each using a single Corrfunc thread to avoid CPU over-subscription.

    Parameters
    ----------
    n_workers : int, optional
        Number of parallel worker processes.  Defaults to
        ``min(cpu_count − 1, n_sub_total)``.  Set to 1 to disable
        parallelism (useful for debugging).

    Returns
    -------
    s_centres : 1D array
    xi0_full  : 1D array
    cov       : 2D array (n_bins × n_bins)
    """
    ncpu = multiprocessing.cpu_count()
    if nthreads is None:
        nthreads = max(1, min(ncpu - 4, 16))

    # ------------------------------------------------------------------ #
    # 1. Bin definitions
    # ------------------------------------------------------------------ #
    nbins_s  = int((max_sep - min_sep) / bin_size)
    s_bins   = np.linspace(min_sep, max_sep, nbins_s + 1)
    nbins_mu = nbins_s * 2
    mu_max   = 1.0
    mu_bins  = np.linspace(0.0, mu_max, nbins_mu + 1)

    # ------------------------------------------------------------------ #
    # 2. Full-sample DD (computed once)
    # ------------------------------------------------------------------ #
    print("Computing DD for full sample …")
    dd_full_result = DDsmu(
        autocorr=1, nthreads=nthreads, binfile=s_bins,
        mu_max=mu_max, nmu_bins=nbins_mu,
        X1=x, Y1=y, Z1=z,
        periodic=True, boxsize=boxsize, verbose=False,
    )
    H_dd_full = dd_full_result['npairs'].reshape(nbins_s, nbins_mu).astype(np.float64)

    # Full-sample monopole
    N = len(x)
    V = boxsize**3
    RR_full = _analytic_rr_2d(s_bins, mu_bins, N, V)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi_full = np.where(RR_full > 0, H_dd_full / RR_full - 1.0, np.nan)
    xi0_full  = compute_monopole_from_xi_s_mu(xi_full, mu_bins)
    s_centres = 0.5 * (s_bins[:-1] + s_bins[1:])
    n_bins    = len(s_centres)

    # ------------------------------------------------------------------ #
    # 3. Assign particles to sub-volumes
    # ------------------------------------------------------------------ #
    sub_edges   = np.linspace(0, boxsize, n_sub_per_side + 1)
    i_idx       = np.clip(np.searchsorted(sub_edges, x, side='right') - 1, 0, n_sub_per_side - 1)
    j_idx       = np.clip(np.searchsorted(sub_edges, y, side='right') - 1, 0, n_sub_per_side - 1)
    k_idx       = np.clip(np.searchsorted(sub_edges, z, side='right') - 1, 0, n_sub_per_side - 1)
    particle_sub = i_idx * n_sub_per_side**2 + j_idx * n_sub_per_side + k_idx

    n_sub_total = n_sub_per_side**3
    sub_vol     = (boxsize / n_sub_per_side)**3
    V_rem       = V - sub_vol

    # ------------------------------------------------------------------ #
    # 4. Parallel pair-subtraction over sub-volumes
    # ------------------------------------------------------------------ #
    # When parallelising, each worker uses 1 Corrfunc thread so the total
    # thread count stays ≤ n_workers (avoids OpenMP over-subscription).
    if n_workers is None:
        n_workers = min(max(1, ncpu - 1), n_sub_total)
    worker_threads = 1 if n_workers > 1 else nthreads

    work_items = []
    for sub_id in range(n_sub_total):
        mask_in  = (particle_sub == sub_id)
        mask_out = ~mask_in
        work_items.append((
            sub_id, n_sub_total,
            x[mask_in],  y[mask_in],  z[mask_in],
            x[mask_out], y[mask_out], z[mask_out],
            s_bins, mu_max, nbins_mu,
            boxsize, worker_threads,
        ))

    print(f"Running {n_sub_total} jackknife realisations "
          f"across {n_workers} worker(s) …")

    xi_sub_all = np.empty((n_sub_total, n_bins))

    if n_workers > 1:
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(x, y, z, particle_sub, s_bins, mu_max, nbins_mu, boxsize, worker_threads)
        ) as pool:
            results = pool.map(_jk_worker_shared, range(n_sub_total))
    else:
        results = [_jk_worker_shared(i) for i in range(n_sub_total)]

    # ------------------------------------------------------------------ #
    # 5. Reconstruct leave-one-out ξ₀ from subtracted pair counts
    # ------------------------------------------------------------------ #
    for sub_id, H_in, H_cross in results:
        mask_in = (particle_sub == sub_id)
        N_in  = int(np.sum(mask_in))
        N_out = N - N_in

        if N_in < 2:
            print(f"Warning: sub-volume {sub_id} nearly empty; using full-sample fallback")
            xi_sub_all[sub_id] = xi0_full
            continue

        H_loo  = H_dd_full - H_in - H_cross
        RR_rem = _analytic_rr_2d(s_bins, mu_bins, N_out, V_rem)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi_loo = np.where(RR_rem > 0, H_loo / RR_rem - 1.0, np.nan)
        xi_sub_all[sub_id] = compute_monopole_from_xi_s_mu(xi_loo, mu_bins)

    # ------------------------------------------------------------------ #
    # 6. Jackknife covariance  Cov = (K−1)/K · Σ (T_i − T̄)(T_i − T̄)ᵀ
    # ------------------------------------------------------------------ #
    xi_bar = np.mean(xi_sub_all, axis=0)
    diff   = xi_sub_all - xi_bar
    cov    = (n_sub_total - 1) / n_sub_total * np.einsum('ij,ik->jk', diff, diff)

    return s_centres, xi0_full, cov