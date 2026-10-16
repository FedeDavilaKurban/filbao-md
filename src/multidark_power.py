"""
Auto- and cross-power spectra of galaxy samples in a periodic box.

    measure_spectra             auto/cross P(k) of all samples of a run -> data/processed/pk/
                                (read by src/bao_fitting.py, gaussian_cov_pk, for the fit)
    xi_covariance_from_spectra  Gaussian covariance of the xi monopoles from those spectra,
                                used only for the error bands of the diagnostic plots

Both build the Gaussian covariance of the xi monopole
from the MEASURED spectra instead of b^2 P_lin + 1/nbar.  That captures any non-Poisson
(excess) shot noise and small-scale clustering of a selected sample, e.g. galaxies selected
by distance to DisPerSE filaments, and the cross-spectra give the correlation between
samples of the same box.

Estimator: CIC assignment, two interlaced grids (Sefusatti et al. 2016) to cancel the
leading aliasing, deconvolution of the CIC window.  Spectra are NOT shot-noise subtracted:
the covariance needs the total power  P_AB(k) + N_AB.
Memory: ~ 3 x ng^3 x 4 bytes per sample (ng = 384 -> ~0.7 GB per sample).
"""
import numpy as np
from scipy import fft


def _cic_counts(pos, L, ng, shift=0.0):
    """CIC-assigned counts on an ng^3 grid; `shift` in units of the cell (0.5 = interlaced grid)."""
    x = np.asarray(pos, np.float64) / L * ng + shift
    i0 = np.floor(x).astype(np.int64)
    d = x - i0
    i0 %= ng
    i1 = (i0 + 1) % ng
    idx, w = [], []
    for cx in (0, 1):
        ix = i1[:, 0] if cx else i0[:, 0]; wx = d[:, 0] if cx else 1 - d[:, 0]
        for cy in (0, 1):
            iy = i1[:, 1] if cy else i0[:, 1]; wy = d[:, 1] if cy else 1 - d[:, 1]
            for cz in (0, 1):
                iz = i1[:, 2] if cz else i0[:, 2]; wz = d[:, 2] if cz else 1 - d[:, 2]
                idx.append((ix * ng + iy) * ng + iz); w.append(wx * wy * wz)
    grid = np.bincount(np.concatenate(idx), weights=np.concatenate(w), minlength=ng**3)
    return grid.reshape(ng, ng, ng).astype(np.float32)


def _kgrid(L, ng):
    kf = 2 * np.pi / L
    kx = fft.fftfreq(ng, 1.0 / ng) * kf
    kz = fft.rfftfreq(ng, 1.0 / ng) * kf
    return kx[:, None, None], kx[None, :, None], kz[None, None, :]


def density_fourier(pos, L, ng=384, interlace=True):
    """Fourier transform of the overdensity, interlaced and CIC-deconvolved (complex64, rfft layout)."""
    N = len(pos)
    nbar_cell = N / ng**3
    kx, ky, kz = _kgrid(L, ng)
    h = L / ng
    D = fft.rfftn(_cic_counts(pos, L, ng) / nbar_cell - 1.0, workers=-1)
    if interlace:
        D2 = fft.rfftn(_cic_counts(pos, L, ng, 0.5) / nbar_cell - 1.0, workers=-1)
        D = 0.5 * (D + D2 * np.exp(1j * (kx + ky + kz) * h / 2).astype(np.complex64))
        del D2
    W = (np.sinc(kx * h / (2 * np.pi)) * np.sinc(ky * h / (2 * np.pi)) * np.sinc(kz * h / (2 * np.pi)))**2
    return (D / W).astype(np.complex64)


def measure_spectra(pos_list, labels, L, ng=384, interlace=True, save=None, kmax_frac=1.0):
    """
    pos_list : list of (N_i, 3) arrays in [0, L) (h^-1 Mpc); labels: matching names
    Returns dict(k, P[i, j, k], labels, n_gal, L, ng, kNyq, nmodes); P includes shot noise.
    Bins: width k_f = 2 pi / L from k_f to kmax_frac * k_Nyquist.
    """
    kx, ky, kz = _kgrid(L, ng)
    K = np.sqrt(kx**2 + ky**2 + kz**2)
    kf = 2 * np.pi / L; kN = np.pi * ng / L
    edges = np.arange(0.5 * kf, kmax_frac * kN + kf, kf)
    ib = np.digitize(K, edges) - 1
    w = np.full(K.shape, 2.0, np.float32); w[..., 0] = 1.0
    if ng % 2 == 0:
        w[..., -1] = 1.0                                    # rfft: count each +-k pair once
    nb = len(edges) - 1
    ok = (ib >= 0) & (ib < nb)
    ibf, wf = ib[ok], w[ok]
    nmodes = np.bincount(ibf, weights=wf, minlength=nb)
    kmean = np.bincount(ibf, weights=(K[ok] * wf), minlength=nb) / np.maximum(nmodes, 1)
    Ds = [density_fourier(p, L, ng, interlace) for p in pos_list]
    S = len(Ds)
    P = np.zeros((S, S, nb))
    for a in range(S):
        for b in range(a, S):
            cross = (Ds[a] * np.conj(Ds[b])).real[ok] * L**3 / ng**6
            P[a, b] = P[b, a] = np.bincount(ibf, weights=cross * wf, minlength=nb) / np.maximum(nmodes, 1)
    good = nmodes > 0
    out = dict(k=kmean[good], P=P[:, :, good], labels=np.array(labels), n_gal=np.array([len(p) for p in pos_list]),
               L=float(L), ng=int(ng), kNyq=kN, nmodes=nmodes[good])
    if save:
        import os
        os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
        np.savez(save, **out)
        print(f"Saved spectra of {list(labels)} -> {save}")
    return out


# ---------------------------------------------------------------------------------------------
# Standalone Gaussian covariance of the xi monopole from the measured spectra
# (no CAMB, no fitting: usable directly in multidark_analysis.py for the error bands of plots)
# ---------------------------------------------------------------------------------------------

def shell_avg_j0(k, s_lo, s_hi):
    """Volume average of j0(k s) over each shell s_lo < s < s_hi (what DD/RR - 1 measures in a
    bin), using  int s^2 j0(ks) ds = [sin(ks) - ks cos(ks)] / k^3.  Shape (n_bins, n_k)."""
    k = np.asarray(k, float)[None, :]
    a = np.asarray(s_lo, float)[:, None]; b = np.asarray(s_hi, float)[:, None]
    F = lambda s: (np.sin(k * s) - k * s * np.cos(k * s)) / k**3
    return 3.0 * (F(b) - F(a)) / (b**3 - a**3)


def _lattice_modes(L, kmax):
    """Exact number of Fourier modes n != 0 of the periodic box at each |k| <= kmax."""
    kf = 2 * np.pi / L
    nmax = int(np.floor(kmax / kf))
    n = np.arange(-nmax, nmax + 1)
    n2 = (n[:, None, None]**2 + n[None, :, None]**2 + n[None, None, :]**2).ravel()
    cnt = np.bincount(n2[n2 <= nmax**2], minlength=nmax**2 + 1).astype(float)
    cnt[0] = 0.0
    m = np.nonzero(cnt)[0]
    return kf * np.sqrt(m), cnt[m]


def power_for_cov(spec, a, b, k, k_smooth=(0.08, 0.15), kfit_max=0.25, deg=4, kuse_frac=0.6,
                  return_tail=False):
    """
    Total power P_ab(k) (clustering + shot noise, as measured) to put in the covariance.
      k >= k_smooth[1]       : the measured spectrum (millions of modes: precise)
      k <= k_smooth[0]       : a smooth fit to the measured spectrum (polynomial in log k over
                               k <= kfit_max, weighted by the Gaussian error of each shell).
                               The few low-k modes are noisy (e.g. +-25 % at k = 0.01), and
                               plugging that noise into the covariance would make it noisy too.
      in between             : linear blend
      k > kuse_frac * k_Nyq  : extrapolated with N + A k^-n fitted on [0.5, 1] x that k (grid
                               effects grow near Nyquist); contributes little on > 5 h^-1 Mpc
    """
    km, P, nm = spec['k'], spec['P'][a, b], spec['nmodes']
    Paa, Pbb = spec['P'][a, a], spec['P'][b, b]
    sig = np.sqrt((Paa * Pbb + P**2) / np.maximum(nm, 1))          # Gaussian error of each shell
    mf = km <= kfit_max
    coef = np.polyfit(np.log(km[mf]), P[mf], deg, w=1.0 / sig[mf])
    smooth = np.polyval(coef, np.log(np.maximum(k, km[0])))
    kuse = kuse_frac * float(spec['kNyq'])
    ok = km <= kuse
    meas = np.interp(k, km[ok], P[ok])
    # beyond kuse: P = N + A k^-n fitted on [0.5, 1] kuse (a white plateau plus a decaying
    # clustering term), so neither the plateau nor the clustering is frozen at its kuse value
    mt = (km >= 0.5 * kuse) & ok
    tail = np.full_like(np.asarray(k, float), P[ok][-1])
    N_tail = float(P[ok][-1])                                        # constant plateau
    try:
        from scipy.optimize import curve_fit
        f = lambda x, N, A, n: N + A * x**(-n)
        p0 = (P[mt][-1] * 0.8, P[mt][-1] * 0.2 * kuse**1.5, 1.5)
        popt, _ = curve_fit(f, km[mt], P[mt], p0=p0, sigma=sig[mt], maxfev=20000,
                            bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, 4.0]))
        tail = f(np.maximum(k, kuse), *popt)
        N_tail = float(popt[0])
    except Exception:
        pass
    t = np.clip((k - k_smooth[0]) / (k_smooth[1] - k_smooth[0]), 0, 1)
    out = (1 - t) * smooth + t * meas
    out = np.where(k > kuse, tail, out)
    return (out, N_tail) if return_tail else out


def xi_covariance_from_spectra(spec, s_edges, labels=None, kmax=3.0, k_exact=0.3, dk=5e-4,
                               analytic_shot=True, **kw):
    """
    Joint Gaussian covariance of the bin-averaged xi monopoles of the samples `labels`
    (default: all) in a periodic box, from their MEASURED auto- and cross-spectra:

        Cov[xi_A(i), xi_B(j)] = (2 / V^2) sum_k P_AB(k)^2 <j0>_i(k) <j0>_j(k),     V = L^3

    (same formula as gaussian_cov_pk in src/bao_fitting.py, which uses linear theory
    instead of a smooth fit at k < 0.08).  Modes: exact lattice count for k < k_exact, smooth
    4 pi k^2 dk / k_f^3 above.  spec: dict from measure_spectra or the saved .npz path.
    Returns (C, labels); C has shape (n_samples * n_bins, n_samples * n_bins), blocks ordered
    as `labels`.

    analytic_shot=True: the constant plateau N_AB (shot noise incl. any excess) is squared and
    summed over ALL modes analytically (sum_k <j0>_i <j0>_j = V delta_ij / V_shell), giving the
    exact 2 N^2 / (V V_shell) diagonal term; the rest is summed to kmax.  Makes the result
    independent of kmax for narrow bins (same as gaussian_cov_pk).
    """
    if isinstance(spec, str):
        d = np.load(spec, allow_pickle=True); spec = {key: d[key] for key in d.files}
    all_labels = [str(x) for x in spec['labels']]
    labels = all_labels if labels is None else list(labels)
    idx = [all_labels.index(l) for l in labels]
    L = float(spec['L']); V = L**3; kf = 2 * np.pi / L
    k1, c1 = _lattice_modes(L, k_exact)
    k2 = np.arange(k1[-1] + 0.5 * dk, kmax, dk)
    kk = np.concatenate([k1, k2]); cnt = np.concatenate([c1, 4 * np.pi * k2**2 * dk / kf**3])
    s_edges = np.asarray(s_edges, float)
    J = shell_avg_j0(kk, s_edges[:-1], s_edges[1:])
    nb = J.shape[0]; T = len(idx)
    V_shell = 4.0 * np.pi / 3.0 * (s_edges[1:]**3 - s_edges[:-1]**3)
    C = np.zeros((T * nb, T * nb))
    for A in range(T):
        for B in range(A, T):
            PAB, N_AB = power_for_cov(spec, idx[A], idx[B], kk, return_tail=True, **kw)
            if analytic_shot:
                blk = (2.0 / V**2) * (J * (cnt * (PAB**2 - N_AB**2))) @ J.T
                blk = blk + np.diag(2.0 * N_AB**2 / (V * V_shell))
            else:
                blk = (2.0 / V**2) * (J * (cnt * PAB**2)) @ J.T
            C[A*nb:(A+1)*nb, B*nb:(B+1)*nb] = blk
            C[B*nb:(B+1)*nb, A*nb:(A+1)*nb] = blk.T
    return C, labels

