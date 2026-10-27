"""
Auto- and cross-power spectra of galaxy samples in a periodic box.

    measure_spectra   auto/cross P(k) of all samples of a run -> data/processed/pk/
                      (read by src/bao_fitting.py: gaussian_cov_pk builds the covariance of the
                      xi monopoles from them, for the fits AND for the error bands of the
                      diagnostic plots of scripts/multidark_analysis.py)

Estimator: CIC assignment, two interlaced grids (Sefusatti et al. 2016) to cancel the
leading aliasing, deconvolution of the CIC window.  Spectra are NOT shot-noise subtracted:
the covariance needs the total power  P_AB(k) + N_AB.

High k by folding (Jenkins et al. 1998): the positions are wrapped into a box of side
L / 2^f (x -> x mod L/2^f) and measured on the same grid.  A mode k = 2 pi n / (L/2^f), n integer,
takes the same value e^{i k.x} for every galaxy before and after wrapping, so the folded box
measures exactly those modes of the full box, and P(k) = 8^f P_folded(k) (same galaxies in
1/8^f of the volume).  Each folding doubles the reach in k at the same memory; it keeps 1 mode
in 8^f, harmless at high k where there are 10^5-10^6 modes per bin.  Fold level f is used from
where level f-1 stops up to kuse_frac x its own Nyquist frequency (grid effects grow near
Nyquist), so the saved spectrum is measured, never extrapolated, up to its last k.
Memory: ~ 3 x ng^3 x 4 bytes per sample (ng = 384 -> ~0.7 GB per sample), the same for every
fold level (levels are measured one after the other).
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


def _spectra_in_box(pos_list, L, ng, interlace, edges):
    """Auto/cross P(k) of all samples in a periodic box of side L, in the k bins `edges`.
    Returns (k mean, P[i, j, bin], number of modes) for the bins that contain modes."""
    kx, ky, kz = _kgrid(L, ng)
    K = np.sqrt(kx**2 + ky**2 + kz**2)
    ib = np.digitize(K, edges) - 1
    w = np.full(K.shape, 2.0, np.float32); w[..., 0] = 1.0
    if ng % 2 == 0:
        w[..., -1] = 1.0                                    # rfft: count each +-k pair once
    nb = len(edges) - 1
    ok = (ib >= 0) & (ib < nb)
    ibf, wf = ib[ok], w[ok]
    nmodes = np.bincount(ibf, weights=wf, minlength=nb)
    kmean = np.bincount(ibf, weights=(K[ok] * wf), minlength=nb) / np.maximum(nmodes, 1)
    del K, ib, w
    Ds = [density_fourier(p, L, ng, interlace) for p in pos_list]
    S = len(Ds)
    P = np.zeros((S, S, nb))
    for a in range(S):
        for b in range(a, S):
            cross = (Ds[a] * np.conj(Ds[b])).real[ok] * L**3 / ng**6
            P[a, b] = P[b, a] = np.bincount(ibf, weights=cross * wf, minlength=nb) / np.maximum(nmodes, 1)
    good = nmodes > 0
    return kmean[good], P[:, :, good], nmodes[good]


def measure_spectra(pos_list, labels, L, ng=384, interlace=True, save=None, n_fold=2, kuse_frac=0.6):
    """
    pos_list : list of (N_i, 3) arrays in [0, L) (h^-1 Mpc); labels: matching names
    n_fold   : number of foldings (0 = no folding).  Level f measures the box folded to L / 2^f.
    kuse_frac: each level is trusted up to kuse_frac x its own Nyquist frequency pi ng 2^f / L;
               the next level takes over from there.  With ng = 384, L = 1000, n_fold = 2:
               level 0 to 0.72, level 1 to 1.45, level 2 to 2.9 h/Mpc.
    Bins: width 2 pi 2^f / L (the fundamental frequency of the box of that level).
    Returns dict(k, P[i, j, k], labels, n_gal, L, ng, kNyq, nmodes, level, n_fold, kuse_frac, kmax);
    P includes shot noise.  kNyq = Nyquist frequency of the unfolded grid; level = fold level of
    each k bin; kmax = upper edge of the last bin (the spectrum is measured up to there).
    """
    kN = np.pi * ng / L
    ks, Ps, nms, lev = [], [], [], []
    k_start = 0.5 * 2 * np.pi / L                           # level 0: bins centred on n k_f
    for f in range(n_fold + 1):
        Lf = L / 2**f
        kf = 2 * np.pi / Lf
        k_top = kuse_frac * np.pi * ng / Lf
        edges = k_start + kf * np.arange(int(np.floor((k_top - k_start) / kf)) + 1)
        pos_f = [np.mod(np.asarray(p, np.float64), Lf) for p in pos_list]
        k, P, nm = _spectra_in_box(pos_f, Lf, ng, interlace, edges)
        ks.append(k); Ps.append(8.0**f * P); nms.append(nm); lev.append(np.full(len(k), f))
        print(f"  fold level {f}: box {Lf:.0f} h^-1 Mpc, k = {edges[0]:.3f}-{edges[-1]:.3f} h/Mpc, {len(k)} bins")
        k_start = edges[-1]
    out = dict(k=np.concatenate(ks), P=np.concatenate(Ps, axis=2), labels=np.array(labels),
               n_gal=np.array([len(p) for p in pos_list]), L=float(L), ng=int(ng), kNyq=kN,
               nmodes=np.concatenate(nms), level=np.concatenate(lev), n_fold=int(n_fold),
               kuse_frac=float(kuse_frac), kmax=float(k_start))
    if save:
        import os
        os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
        np.savez(save, **out)
        print(f"Saved spectra of {list(labels)} -> {save}")
    return out
