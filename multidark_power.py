"""
Auto- and cross-power spectra of galaxy samples in a periodic box.

Used by filbao_fit.gaussian_cov_pk to build the Gaussian covariance of the xi monopole
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
