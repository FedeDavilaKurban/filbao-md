"""
FilBAO, plan step 3: BAO fit of the xi_0(s) monopole of every subsample.

    run_all(samples, pk_file, pairs)  ->  per sample: alpha, Sigma, Delta chi2 of the BAO
    detection and its null-mock p-value; per pair: Delta alpha, Delta Sigma with errors from
    correlated mocks; figures: each monopole with its fit, (alpha, Sigma) contours.

Model (Xu+12 eq. 25-26, Chen+24 eq. 26 in configuration space):

    xi(s) = B^2 [ xi_nw(s) + xi_wig(alpha s; Sigma) ] + a0 + a1 (100/s) + a2 (100/s)^2

    xi_wig(s; Sigma) = int k^2 dk / 2 pi^2  [P_lin - P_nw](k) exp(-k^2 Sigma^2 / 2) j0(k s)

Sections
    1. Settings
    2. Linear theory and the wiggle / no-wiggle split
    3. The template xi(s; alpha, Sigma)
    4. Theory Sigma_nl (centre of the Sigma prior)
    5. Gaussian covariance from the measured P(k)
    6. Bias from scales below the BAO (used only in the covariance)
    7. Likelihood on the (alpha, Sigma) grid
    8. Gaussian mocks: detection p-value, errors on differences
    9. Plots
   10. Drivers: analyse_sample, run_all

The design choices and the tests behind them (V0-V6) are in
archive/bao_fitting_gaussian_v3.ipynb.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, 'data', 'processed', 'cache')
FIG_DIR = os.path.join(REPO, 'figures')

# =============================================================================================
# 1. Settings.  CHECK THESE against the simulation you are analysing.
# =============================================================================================
# MultiDark-Planck (MDPL2): Omega_m = 0.307115, Omega_b = 0.048206, h = 0.6777, n_s = 0.96,
# sigma_8 = 0.8228, no massive neutrinos.  ombh2 = Omega_b h^2, omch2 = (Omega_m - Omega_b) h^2;
# As chosen so that CAMB gives sigma_8 = 0.8228 (sets the amplitude of P_lin, hence Sigma_nl theory).
COSMO = dict(H0=67.77, ombh2=0.022140, omch2=0.118911, mnu=0.0, As=2.1186e-9, ns=0.96)
Z_SNAP = 0.0          # redshift of the MultiDark snapshot; positions are in REAL space
L_BOX = 1000.0        # h^-1 Mpc

# Broadband: a0 + a1 (100/s) + a2 (100/s)^2 = Xu+12 eq. 26 ("poly3"); powers of (s / 100)
BROADBAND_POWERS = (0, -1, -2)


# =============================================================================================
# 2. Linear theory and the wiggle / no-wiggle split
# ---------------------------------------------------------------------------------------------
# CAMB linear P(k) once (cached), plus r_d, f, sigma_8.  The no-wiggle P_nw is the Wallisch
# (2018) / Baumann et al. (2018) split, the DESI baseline (Chen+24 Sec. 7.4): it removes the
# WHOLE r_d feature, including the lowest-k BAO harmonic, so xi_nw has no bump.
# =============================================================================================
def linear_theory(cosmo=COSMO, z=Z_SNAP, kmax=20.0, cache=None, force=False):
    """CAMB linear matter P(k) to k = 20 h/Mpc plus the derived numbers the pipeline needs:
    r_d (sound horizon at the drag epoch) in h^-1 Mpc, growth rate f, sigma8(z).
    Read from data/processed/cache if there AND computed for this cosmology (CAMB is only needed
    to recompute)."""
    cache = cache or os.path.join(CACHE_DIR, f"camb_linear_z{z:.2f}.npz")
    tag = repr(sorted(cosmo.items()))                     # the cosmology the cache was made for
    if (not force) and os.path.exists(cache):
        d = np.load(cache)
        if 'cosmo' in d.files and str(d['cosmo']) == tag:
            return {key: (d[key] if d[key].ndim else float(d[key])) for key in d.files if key != 'cosmo'}
        print(f"{cache} was computed for another cosmology: recomputing")
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo['H0'], ombh2=cosmo['ombh2'], omch2=cosmo['omch2'],
                       mnu=cosmo.get('mnu', 0.06))        # CAMB's default is mnu = 0.06 eV
    pars.InitPower.set_params(As=cosmo['As'], ns=cosmo['ns'])
    pars.set_matter_power(redshifts=[z], kmax=kmax)
    res = camb.get_results(pars)
    kh, _, pk = res.get_matter_power_spectrum(minkh=1e-5, maxkh=kmax, npoints=40000)
    h = cosmo['H0'] / 100.0
    s8 = float(res.get_sigma8()[0])
    out = dict(kh=kh, pk=pk[0], rd=res.get_derived_params()['rdrag'] * h,
               f=float(res.get_fsigma8()[0]) / s8, sigma8=s8, z=z)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, cosmo=tag, **out)
    return out


def dewiggle_wallisch(kh, pk, rd, kmin=1e-4, kmax=7.0, n=2**16, hole=(0.73, 1.46)):
    """Wallisch (2018) / Baumann et al. (2018) no-wiggle, the DESI baseline (Chen+24 Sec. 7.4):
    discrete sine transform of ln(kP) on a uniform k grid; the BAO is a bump in the even and
    odd DST coefficients at 'frequency' r ~ r_d.  Cut the coefficients for r in hole*r_d,
    re-spline (coefficients x (i+1)^2), transform back."""
    from scipy.fft import dst, idst
    k = np.linspace(kmin, kmax, n)
    P = np.exp(np.interp(np.log(k), np.log(kh), np.log(pk)))
    c = dst(np.log(k * P), type=2, norm='ortho')
    K = kmax - kmin
    jlo, jhi = int(hole[0] * rd * K / (2 * np.pi)), int(hole[1] * rd * K / (2 * np.pi))
    c2 = np.empty_like(c)
    for par in (0, 1):                                    # even and odd coefficients
        a = c[par::2]; j = np.arange(len(a)); keep = np.ones(len(a), bool); keep[jlo:jhi] = False
        c2[par::2] = CubicSpline(j[keep], a[keep] * (j[keep] + 1)**2)(j) / (j + 1)**2
    return k, np.exp(idst(c2, type=2, norm='ortho')) / k


def generate_templates(force_recompute=False, save_path=None, z=Z_SNAP, cosmo=COSMO, lin=None):
    """(k, P_w, P_nw) on a uniform grid 0.001-0.8 h/Mpc (2048 points); P_w = linear P(k),
    P_nw = Wallisch no-wiggle.  The smooth tail beyond k = 0.8 is added by BAOTemplate."""
    save_path = save_path or os.path.join(CACHE_DIR, f'templates_wallisch_z{z:.2f}.npz')
    tag = repr(sorted(cosmo.items()))
    if not force_recompute and os.path.exists(save_path):
        d = np.load(save_path)
        if 'cosmo' in d.files and str(d['cosmo']) == tag:
            return d['k'], d['P_w'], d['P_nw']
        print(f"{save_path} was computed for another cosmology: recomputing")
    lin = lin or linear_theory(cosmo, z)
    k = np.linspace(1e-3, 0.8, 2048)
    P_w = np.exp(np.interp(np.log(k), np.log(lin['kh']), np.log(lin['pk'])))
    kk, Pn = dewiggle_wallisch(lin['kh'], lin['pk'], lin['rd'])
    P_nw = np.interp(k, kk, Pn)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, k=k, P_w=P_w, P_nw=P_nw, cosmo=tag)
    print(f"Templates (wallisch) computed and saved to {save_path}")
    return k, P_w, P_nw


def load_monopole(filename):
    """(s, xi0, n_gal) from a monopole file written by scripts/multidark_analysis.py."""
    data = np.load(filename, allow_pickle=True)
    n_gal = int(data['n_gal']) if 'n_gal' in data else None
    return data['s'], data['xi0'], n_gal


def edges_from_centres(s):
    """Bin edges from equally spaced bin centres."""
    ds = s[1] - s[0]
    return np.concatenate([s - ds / 2, [s[-1] + ds / 2]])


# =============================================================================================
# 3. The template xi(s; alpha, Sigma) = xi_nw(s) + xi_wig(alpha s; Sigma)
# ---------------------------------------------------------------------------------------------
# * Damping only on the wiggle (Xu+12 eq. 14, Chen+24 eq. 26): non-linear pairwise motions
#   convolve xi with the displacement distribution (ESW07 Sec. 4) -> exp(-k^2 Sigma^2 / 2) in k.
#   A smoothing, not a shift: that is why alpha is robust.
# * Only the wiggle is dilated (Chen+24 Sec. 7.5), so alpha can only come from the BAO.
# * No ringing: the smooth part runs to k = 6 with a 1 h^-1 Mpc Gaussian taper (Xu+12 eq. 27);
#   a hard cut at k = 0.8 rings as much as the BAO itself.  The taper is not on the wiggle,
#   so it never adds to Sigma.
# * Everything is averaged over the measurement bins, as DD/RR averages the data.
# =============================================================================================
def shell_avg_j0(k, s_lo, s_hi):
    """Volume-weighted average of j0(k s) over each shell s_lo < s < s_hi.
    Uses  int s^2 j0(ks) ds = [sin(ks) - ks cos(ks)] / k^3.   Shape (n_bins, n_k).
    This is what DD/RR - 1 measures in a bin, so model and data are averaged the same way."""
    k = np.asarray(k, float)[None, :]
    a = np.asarray(s_lo, float)[:, None]
    b = np.asarray(s_hi, float)[:, None]
    F = lambda s: (np.sin(k * s) - k * s * np.cos(k * s)) / k**3
    return 3.0 * (F(b) - F(a)) / (b**3 - a**3)


def _j0_kernel(k, edges=None, s=None):
    """Bin-averaged j0 if `edges` is given, point j0(k s) if `s` is given."""
    if edges is not None:
        edges = np.asarray(edges, float)
        return shell_avg_j0(k, edges[:-1], edges[1:])
    return np.sinc(np.outer(np.asarray(s, float), k) / np.pi)


def _trap_w(x):
    """Trapezoid-rule weights on the grid x."""
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2]); w[0] = 0.5 * (x[1] - x[0]); w[-1] = 0.5 * (x[-1] - x[-2])
    return w


class BAOTemplate:
    """
    xi_nw (s)        = int k^2 dk/2pi^2  P_nw(k) e^{-k^2 a^2/2}      j0(k s)
    xi_wig(s; Sigma) = int k^2 dk/2pi^2 [P_w - P_nw](k) e^{-k^2 Sigma^2/2} j0(k s)
    """

    def __init__(self, k, P_w, P_nw, lin, a_taper=1.0, k_ext_max=6.0, dk_ext=5e-4):
        dW = P_w - P_nw
        nz = np.nonzero(dW)[0]
        sl = slice(max(nz[0] - 1, 0), min(nz[-1] + 2, len(k)))
        self.kw, self.dW = k[sl], dW[sl]                       # wiggle support only
        self.ww = _trap_w(self.kw) * self.kw**2 * self.dW / (2 * np.pi**2)
        kx = np.arange(dk_ext, k_ext_max + 0.5 * dk_ext, dk_ext)
        P_camb = np.exp(np.interp(np.log(kx), np.log(lin['kh']), np.log(lin['pk'])))
        Pnw = np.where(kx <= k.max(), np.interp(kx, k, P_nw), P_camb)  # split below 0.8, CAMB above
        self.kx = kx
        self.wx = _trap_w(kx) * kx**2 * Pnw * np.exp(-(kx * a_taper)**2 / 2) / (2 * np.pi**2)
        self.k, self.P_w, self.P_nw, self.lin = k, P_w, P_nw, lin

    def xi_nw(self, edges=None, s=None):
        """Smooth part, bin-averaged (edges) or at points (s)."""
        return _j0_kernel(self.kx, edges, s) @ self.wx

    def wiggle_kernel(self, alphas, edges=None, s=None):
        """K[a, i, k] such that xi_wig(alpha_a s_i; Sigma) = K[a, i, :] @ exp(-k^2 Sigma^2/2).
        Computing K once and then contracting with many Sigma values makes the (alpha, Sigma)
        grid cheap."""
        al = np.atleast_1d(np.asarray(alphas, float))
        return np.stack([_j0_kernel(a * self.kw, edges, s) * self.ww[None, :] for a in al])

    def damping(self, Sigmas):
        Sig = np.atleast_1d(np.asarray(Sigmas, float))
        return np.exp(-0.5 * np.outer(self.kw**2, Sig**2))          # (n_k, S)

    def xi_wig(self, Sigma, alpha=1.0, edges=None, s=None):
        K = self.wiggle_kernel([alpha], edges, s)[0]
        return K @ np.exp(-0.5 * self.kw**2 * Sigma**2)

    def xi(self, Sigma, alpha=1.0, edges=None, s=None):
        return self.xi_nw(edges, s) + self.xi_wig(Sigma, alpha, edges, s)


# =============================================================================================
# 4. Theory Sigma_nl: centre of the Gaussian Sigma prior used for the alpha fit
# =============================================================================================
def sigma_nl_theory(lin):
    """
    Pairwise Zel'dovich damping of the BAO in real space (ESW07 Sec. 4-5; Chen+24 eq. 9):

        Sigma^2 = (2/3) int dk/(2 pi^2) P_lin(k) [1 - j0(k r_d)]

    = per-axis rms of the displacement of one galaxy RELATIVE to another one r_d away.
    Modes with k << 1/r_d move both galaxies together and drop out.  7.67 h^-1 Mpc at z = 0.
    """
    kh, P, rd = lin['kh'], lin['pk'], lin['rd']
    j0 = np.sinc(kh * rd / np.pi)
    S = np.sqrt((2.0 / 3.0) * np.trapezoid(P * (1 - j0), kh) / (2 * np.pi**2))
    return dict(Sigma=S, Sigma_mono=S, rd=rd, f=lin['f'], z=lin['z'])


# =============================================================================================
# 5. Gaussian covariance
# ---------------------------------------------------------------------------------------------
#   Cov[xi_A(i), xi_B(j)] = (2/V^2) sum_{n != 0} P_AB(k_n)^2 <j0>_i(k_n) <j0>_j(k_n),  V = L^3
# (Xu+12 eq. 4 and 9, as a sum over the box's discrete modes).  P_AB is each pair of samples'
# MEASURED auto/cross power (signal + shot noise, incl. any non-Poisson excess), from
# multidark_power.measure_spectra, with b_A b_B P_lin + N_AB below k = 0.08 so the few noisy
# low-k modes of the box are not fed back.  The cross terms make filament / non-filament
# errors correlated, which is what makes the errors on differences right.
# The Poisson version (b^2 P_lin + 1/nbar) is kept only for the bias estimate and to print
# how much larger the measured-P(k) errors are.
# =============================================================================================
_LATTICE_CACHE = {}


def lattice_mode_counts(L, kmax):
    """Fourier modes n in Z^3\\{0} grouped by |n|^2, with k = (2 pi/L)|n| <= kmax."""
    key = (float(L), float(kmax))
    if key in _LATTICE_CACHE:
        return _LATTICE_CACHE[key]
    kf = 2 * np.pi / L
    nmax = int(np.floor(kmax / kf))
    n = np.arange(-nmax, nmax + 1)
    n2_2d = n[:, None]**2 + n[None, :]**2
    cnt = np.zeros(nmax**2 + 1)
    for nx in n:
        v = (nx**2 + n2_2d).ravel()
        cnt += np.bincount(v[v <= nmax**2], minlength=nmax**2 + 1)
    cnt[0] = 0.0
    m = np.nonzero(cnt)[0]
    _LATTICE_CACHE[key] = (kf * np.sqrt(m), cnt[m])
    return _LATTICE_CACHE[key]


def gaussian_cov_box(s_edges, nbar, bias, k, P_lin, L=L_BOX, kmax=3.0):
    """Poisson-Gaussian covariance of one sample: P = b^2 P_lin + 1/nbar."""
    return gaussian_cov_multi(s_edges, [dict(b=bias, n_gal=nbar * L**3)], k, P_lin, L, kmax=kmax)


def gaussian_cov_multi(s_edges, tracers, k, P_lin, L=L_BOX, n_shared=None, kmax=3.0):
    """Poisson-Gaussian joint covariance of several tracers (dict(b=..., n_gal=...)) of the same
    box: P_AB = b_A b_B P_lin + N_AB V / (N_A N_B), N_AB = shared galaxies (default: disjoint)."""
    s_edges = np.asarray(s_edges, float)
    T = len(tracers); V = L**3
    kk, cnt = lattice_mode_counts(L, min(kmax, float(k.max())))
    J = shell_avg_j0(kk, s_edges[:-1], s_edges[1:])
    P = np.interp(kk, k, P_lin)
    N = np.array([t['n_gal'] for t in tracers], float)
    b = np.array([t['b'] for t in tracers], float)
    Ns = np.diag(N) if n_shared is None else np.asarray(n_shared, float)
    nb = J.shape[0]
    C = np.zeros((T * nb, T * nb))
    for A in range(T):
        for B in range(A, T):
            PAB = b[A] * b[B] * P + Ns[A, B] * V / (N[A] * N[B])
            blk = (2.0 / V**2) * (J * (cnt * PAB**2)) @ J.T
            C[A*nb:(A+1)*nb, B*nb:(B+1)*nb] = blk
            C[B*nb:(B+1)*nb, A*nb:(A+1)*nb] = blk.T
    return C


def load_spectra(pk_file):
    """Spectra saved by multidark_power.measure_spectra (k, P[i, j, k], labels, n_gal, L, ng, kNyq)."""
    d = np.load(pk_file, allow_pickle=True)
    out = {key: d[key] for key in d.files}
    out['labels'] = [str(x) for x in out['labels']]
    out['L'] = float(out['L']); out['kNyq'] = float(out['kNyq'])
    return out


def spectrum_for_cov(spec, iA, iB, bA, bB, lin, kk, k_lo=0.08, k_hi=0.15, n_win=(0.10, 0.20),
                     kuse_frac=0.6):
    """
    Total power P_AB(k) (signal + shot noise) for the covariance, at wavenumbers kk, and the
    constant high-k excess (shot-noise plateau) ex_tail.
      k >= k_hi           : the MEASURED spectrum
      k <= k_lo           : b_A b_B P_lin(k) + N_AB, with N_AB = mean of (P_meas - b_A b_B P_lin)
                            over n_win
      k_lo < k < k_hi     : linear blend
      k > kuse_frac*k_Nyq : b_A b_B P_lin(k) + the excess measured near the last usable k
    """
    k, P = spec['k'], spec['P'][iA, iB]
    Pl = lambda x: np.exp(np.interp(np.log(x), np.log(lin['kh']), np.log(lin['pk'])))
    ex = P - bA * bB * Pl(k)
    w = spec['nmodes'] if 'nmodes' in spec else np.ones_like(k)
    mN = (k >= n_win[0]) & (k <= n_win[1])
    N_AB = np.sum(ex[mN] * w[mN]) / np.sum(w[mN])
    kuse = kuse_frac * spec['kNyq']
    mt = (k > 0.9 * kuse) & (k <= kuse)
    ex_tail = np.sum(ex[mt] * w[mt]) / np.sum(w[mt])
    meas = np.interp(kk, k[k <= kuse], P[k <= kuse])
    model_lo = bA * bB * Pl(kk) + N_AB
    model_hi = bA * bB * Pl(kk) + ex_tail
    t = np.clip((kk - k_lo) / (k_hi - k_lo), 0, 1)
    out = (1 - t) * model_lo + t * meas
    out = np.where(kk > kuse, model_hi, out)
    return out, ex_tail


def gaussian_cov_pk(s_edges, spec, idx, biases, lin, L=None, kmax=3.0, **kw):
    """
    Joint Gaussian covariance of the monopoles of samples spec['labels'][idx], from their
    measured auto- and cross-spectra.  Shared galaxies and excess noise are inside the spectra.
    The constant plateau N_AB is summed over ALL modes analytically,
        sum_n <j0>_i <j0>_j = V delta_ij / V_shell_i   (Parseval, non-overlapping shells),
    giving the exact 2 N^2 / (V V_shell) diagonal term; the rest is summed to kmax.  So narrow
    bins do not lose shot noise at the kmax cutoff.
    """
    L = spec['L'] if L is None else L
    s_edges = np.asarray(s_edges, float); V = L**3
    kk, cnt = lattice_mode_counts(L, kmax)
    J = shell_avg_j0(kk, s_edges[:-1], s_edges[1:])
    nb = J.shape[0]; T = len(idx)
    V_shell = 4.0 * np.pi / 3.0 * (s_edges[1:]**3 - s_edges[:-1]**3)
    C = np.zeros((T * nb, T * nb))
    for A in range(T):
        for B in range(A, T):
            PAB, N_AB = spectrum_for_cov(spec, idx[A], idx[B], biases[A], biases[B], lin, kk, **kw)
            blk = (2.0 / V**2) * (J * (cnt * (PAB**2 - N_AB**2))) @ J.T
            blk = blk + np.diag(2.0 * N_AB**2 / (V * V_shell))
            C[A*nb:(A+1)*nb, B*nb:(B+1)*nb] = blk
            C[B*nb:(B+1)*nb, A*nb:(A+1)*nb] = blk.T
    return C


def spectrum_key_from_file(monopole_file):
    """'mag=-21.2_sep=5.0-150.0_binsep=5.0_<bin_desc>.npz' -> '<bin_desc>' (the label used when
    the spectra were measured in multidark_analysis.py)."""
    return os.path.basename(monopole_file).split('_', 3)[3][:-4]


# =============================================================================================
# 6. Bias from scales below the BAO
# ---------------------------------------------------------------------------------------------
# b^2 from xi_gal = b^2 xi_lin on 40-70 h^-1 Mpc (spread over three ranges printed as a
# warning).  Used ONLY inside the covariance (its k < 0.08 part scales as b^4), so the
# covariance is never tuned on the BAO region.
# =============================================================================================
def estimate_bias_independent(s, xi, s_edges, nbar, T, L=L_BOX,
                              ranges=((30, 60), (40, 70), (35, 75)), primary=1, n_iter=4):
    """Weighted least squares with the DIAGONAL of the Poisson-Gaussian covariance (robust
    against the bin-to-bin structure of the data; the full-GLS version returned b -> 0 with
    2.5 h^-1 Mpc bins).  b is relative to the template amplitude."""
    k, P_w = T.k, T.P_w
    xi_lin = T.xi(0.0, edges=s_edges)            # bin-averaged, undamped
    out = []
    for (lo, hi) in ranges:
        m = (s >= lo) & (s <= hi)
        b = 1.0
        x = xi_lin[m]
        for _ in range(n_iter):
            C = gaussian_cov_box(s_edges, nbar, b, k, P_w, L)[np.ix_(m, m)]
            Ci = np.diag(1.0 / np.diag(C))
            b2 = (x @ Ci @ xi[m]) / (x @ Ci @ x)
            b = np.sqrt(max(b2, 1e-6))
        sig_b2 = np.sqrt(x @ Ci @ C @ Ci @ x) / (x @ Ci @ x)
        r = xi[m] - b2 * x
        chi2 = r @ np.linalg.solve(C, r)
        out.append(dict(range=(lo, hi), b=b, b2=b2, sigma_b=0.5 * sig_b2 / b, chi2=chi2,
                        dof=int(m.sum()) - 1))
    bs = np.array([o['b'] for o in out])
    if bs.min() < 0.3 or bs.max() / bs.min() > 1.15:
        print(f"WARNING bias estimate unstable across ranges: "
              + ", ".join(f"{o['range']}: {o['b']:.3f}" for o in out)
              + "  -> check the monopole at 30-75 h^-1 Mpc; the covariance scales as b^4 at k < 0.08")
    return dict(b=out[primary]['b'], b_min=bs.min(), b_max=bs.max(), detail=out)


# =============================================================================================
# 7. Likelihood on the (alpha, Sigma) grid
# ---------------------------------------------------------------------------------------------
# For fixed (alpha, Sigma) the model is linear in theta = (B^2, a0, a1, a2): solved exactly
# (_gls, B^2 >= 0), giving chi2(alpha, Sigma) and log det F (Chen+24 App. F).  From the grid:
#   profile  : chi2_prof(alpha) = min_Sigma [chi2 - 2 ln prior]  -> best alpha, Delta chi2 = 1
#   marginal : p(alpha) ~ int dSigma prior exp(-(chi2 + ln det F)/2) -> mean, std, 16/50/84 %
#   detection: Delta chi2 = chi2(no-wiggle model) - chi2(BAO model)   (Xu+12 Sec. 4.2)
# Sigma priors: ('gauss', mean, width), mean may be 'theory' (Chen+24 Sec. 7.2), or
# ('flat', lo, hi) to measure Sigma.  alpha grid 0.80-1.20 is a hard prior.
# =============================================================================================
def broadband_columns(s):
    return np.column_stack([(np.asarray(s) / 100.0)**p for p in BROADBAND_POWERS])


def parse_sigma_prior(prior, Sigma_theory):
    """Returns (Sigma grid, -2 ln prior on that grid)."""
    kind = prior[0]
    v = lambda x: Sigma_theory if (isinstance(x, str) and x == 'theory') else float(x)
    if kind == 'gauss':
        mu, sd = v(prior[1]), float(prior[2])
        grid = np.arange(max(0.0, mu - 4 * sd), mu + 4 * sd + 1e-9, min(0.25, sd / 4))
        return grid, ((grid - mu) / sd)**2
    if kind == 'flat':
        lo, hi = float(prior[1]), float(prior[2])
        grid = np.arange(lo, hi + 1e-9, 0.25)
        return grid, np.zeros_like(grid)
    raise ValueError(prior)


def build_design(T, s_edges_fit, alphas, Sigmas):
    """Design matrix X[a, j, i, p] for alpha_a, Sigma_j, bin i.
    Column 0 = BAO template (multiplied by B^2), columns 1.. = broadband."""
    sc = 0.5 * (s_edges_fit[:-1] + s_edges_fit[1:])
    K = T.wiggle_kernel(alphas, edges=s_edges_fit)                     # (A, N, nk)
    xw = np.einsum('ank,ks->asn', K, T.damping(Sigmas))                # (A, S, N)
    nw = T.xi_nw(edges=s_edges_fit)[None, None, :]
    bb = broadband_columns(sc)
    A_, S_, N_ = xw.shape
    X = np.empty((A_, S_, N_, 1 + bb.shape[1]))
    X[..., 0] = nw + xw
    X[..., 1:] = bb
    return X


def design_nowiggle(T, s_edges_fit):
    sc = 0.5 * (s_edges_fit[:-1] + s_edges_fit[1:])
    return np.column_stack([T.xi_nw(edges=s_edges_fit), broadband_columns(sc)])


def _gls(X, Ci, D):
    """Closed-form generalised least squares for every design matrix in X (..., N, p) and every
    data vector in D (M, N) (Chen+24 App. F, eq. F4-F8).  B^2 >= 0 is enforced by refitting
    without column 0 where it comes out negative.
    Returns chi2 (..., M), theta (..., M, p), log det F (...)."""
    XtCi = np.einsum('...np,nm->...pm', X, Ci)
    F = XtCi @ X
    dg = 1.0 / np.sqrt(np.einsum('...pp->...p', F))
    Ft = F * dg[..., :, None] * dg[..., None, :]
    b = np.einsum('...pn,mn->...mp', XtCi, D)
    th = np.linalg.solve(Ft[..., None, :, :], (b * dg[..., None, :])[..., None])[..., 0] * dg[..., None, :]
    dCd = np.einsum('mn,nk,mk->m', D, Ci, D)
    chi2 = dCd - np.sum(b * th, -1)
    logdetF = np.linalg.slogdet(Ft)[1] - 2 * np.sum(np.log(dg), -1)
    neg = th[..., 0] < 0
    if np.any(neg) and X.shape[-1] > 1:
        Fr, br = F[..., 1:, 1:], b[..., 1:]
        dgr = 1.0 / np.sqrt(np.einsum('...pp->...p', Fr))
        Frt = Fr * dgr[..., :, None] * dgr[..., None, :]
        thr = np.linalg.solve(Frt[..., None, :, :], (br * dgr[..., None, :])[..., None])[..., 0] * dgr[..., None, :]
        chi2 = np.where(neg, dCd - np.sum(br * thr, -1), chi2)
        th0 = np.zeros_like(th); th0[..., 1:] = thr
        th = np.where(neg[..., None], th0, th)
    elif np.any(neg):
        chi2 = np.where(neg, dCd, chi2); th = np.where(neg[..., None], 0.0, th)
    return chi2, th, logdetF


def _interval(x, chi2, delta=1.0):
    """Best x (parabolic refinement) and the Delta-chi2 = delta interval; NaN if open."""
    i0 = int(np.argmin(chi2))
    if 0 < i0 < len(x) - 1:
        y0, y1, y2 = chi2[i0 - 1], chi2[i0], chi2[i0 + 1]
        den = y0 - 2 * y1 + y2
        xb = x[i0] + 0.5 * (y0 - y2) / den * (x[1] - x[0]) if den > 0 else x[i0]
    else:
        xb = x[i0]
    thr = chi2[i0] + delta
    lo = hi = np.nan
    left = np.nonzero(chi2[:i0] > thr)[0]
    if len(left):
        j = left[-1]; lo = np.interp(thr, [chi2[j + 1], chi2[j]], [x[j + 1], x[j]])
    right = np.nonzero(chi2[i0:] > thr)[0]
    if len(right):
        j = i0 + right[0]; hi = np.interp(thr, [chi2[j - 1], chi2[j]], [x[j - 1], x[j]])
    return xb, lo, hi, dict(at_edge=(i0 == 0 or i0 == len(x) - 1), lo_open=np.isnan(lo), hi_open=np.isnan(hi))


def _summ_1d(x, p):
    """mean, std and 16/50/84 percentiles of a gridded 1D posterior."""
    if len(x) == 1:
        return dict(mean=x[0], std=0.0, q=(x[0], x[0], x[0]))
    p = p / np.trapezoid(p, x)
    mean = np.trapezoid(x * p, x)
    std = np.sqrt(np.trapezoid((x - mean)**2 * p, x))
    cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    q = np.interp([0.1587, 0.5, 0.8413], cdf / cdf[-1], x)
    return dict(mean=mean, std=std, q=tuple(q))


def fit_bao(s, xi, cov_full, T, fit_range=(50, 150), Sigma_prior=('gauss', 'theory', 2.0),
            Sigma_theory=None, alpha_grid=None, s_edges=None):
    """
    Fit  xi(s) = B^2 [xi_nw(s) + xi_wig(alpha s; Sigma)] + sum_p a_p (s/100)^p  on the
    (alpha, Sigma) grid; B^2 and the a_p are solved exactly at every grid point.
    """
    s_edges = edges_from_centres(s) if s_edges is None else s_edges
    Sigma_theory = Sigma_theory if Sigma_theory is not None else sigma_nl_theory(T.lin)['Sigma_mono']
    ag = np.arange(0.80, 1.2001, 0.001) if alpha_grid is None else alpha_grid
    Sg, pri = parse_sigma_prior(Sigma_prior, Sigma_theory)
    m = (s >= fit_range[0]) & (s <= fit_range[1])
    ef = np.concatenate([s_edges[:-1][m], [s_edges[1:][m][-1]]])
    d, C = xi[m], cov_full[np.ix_(m, m)]
    Ci = np.linalg.inv(C)

    X = build_design(T, ef, ag, Sg)
    chi2, th, ldF = _gls(X, Ci, d[None, :])
    chi2, th = chi2[..., 0], th[..., 0, :]                     # (A,S), (A,S,p)
    chi2p = chi2 + pri[None, :]

    # profile likelihoods
    prof_a = chi2p.min(1); prof_S = chi2p.min(0)
    a_best, a_lo, a_hi, a_fl = _interval(ag, prof_a)
    if len(Sg) > 1:
        S_best, S_lo, S_hi, S_fl = _interval(Sg, prof_S)
    else:
        S_best, S_lo, S_hi, S_fl = Sg[0], np.nan, np.nan, dict(at_edge=False, lo_open=True, hi_open=True)
    ia, iS = np.unravel_index(np.argmin(chi2p), chi2p.shape)

    # marginal posterior
    m2 = chi2 + ldF + pri[None, :]
    post2d = np.exp(-0.5 * (m2 - m2.min()))
    pa = np.trapezoid(post2d, Sg, axis=1) if len(Sg) > 1 else post2d[:, 0]
    pS = np.trapezoid(post2d, ag, axis=0) if len(ag) > 1 else post2d[0, :]
    ma, mS = _summ_1d(ag, pa), _summ_1d(Sg, pS)

    # no-wiggle model (BAO detection)
    Xn = design_nowiggle(T, ef)
    chi2_nw = float(_gls(Xn[None], Ci, d[None, :])[0][0, 0])

    n_lin = X.shape[-1]
    n_nl = 1 + (len(Sg) > 1)
    dof = int(m.sum()) - n_lin - n_nl
    cmin = float(chi2p[ia, iS])
    return dict(alpha=a_best, lo=a_lo, hi=a_hi, sigma_alpha=0.5 * (a_hi - a_lo), flags=a_fl,
                alpha_marg=ma, Sigma=S_best, S_lo=S_lo, S_hi=S_hi,
                sigma_Sigma=0.5 * (S_hi - S_lo), S_flags=S_fl, Sigma_marg=mS,
                chi2_min=cmin, dof=dof, p_low=stats.chi2.cdf(cmin, dof), p_high=stats.chi2.sf(cmin, dof),
                B2=float(th[ia, iS, 0]), theta=th[ia, iS], ij_best=(ia, iS), chi2_nowiggle=chi2_nw,
                dchi2_wiggle=chi2_nw - cmin,
                alpha_grid=ag, Sigma_grid=Sg, Sigma_prior=Sigma_prior, prior_m2lnp=pri,
                Sigma_theory=Sigma_theory, chi2_grid=chi2, post2d=post2d,
                prof_alpha=prof_a, prof_Sigma=prof_S, post_alpha=pa, post_Sigma=pS,
                s_fit=s[m], edges_fit=ef, d=d, C=C, Ci=Ci, X_best=X[ia, iS], Xn=Xn,
                fit_range=fit_range, T=T, mask=m)


def best_models(fit, s_all, s_edges_all):
    """Best-fit BAO model and best-fit no-wiggle model on all bins (for plots)."""
    T, d, Ci = fit['T'], fit['d'], fit['Ci']
    Sg = np.array([fit['Sigma']]); ag = np.array([fit['alpha']])
    Xa = build_design(T, fit['edges_fit'], ag, Sg)
    th = _gls(Xa, Ci, d[None, :])[1][0, 0, 0]
    m_w = build_design(T, s_edges_all, ag, Sg)[0, 0] @ th
    thn = _gls(fit['Xn'][None], Ci, d[None, :])[1][0, 0]
    m_nw = design_nowiggle(T, s_edges_all) @ thn
    return m_w, m_nw


# =============================================================================================
# 8. Gaussian mocks
# ---------------------------------------------------------------------------------------------
# run_mocks(fit, truth='nowiggle'): null distribution of the detection Delta chi2 -> p-value.
# run_mocks(fit, truth='best')    : calibration of the errors (pull std, coverage); used in
#                                   notebooks/exploratory/calibration_mocks.ipynb, not in run_all.
# difference_test: correlated mocks of ALL samples from the joint covariance, each refit with
# its own settings -> sigma(Delta alpha), sigma(Delta Sigma) and a null p-value.
# These mocks validate the machinery, not the covariance (they are drawn from it).
# =============================================================================================
def fit_many(D, fit, alpha_step=0.002, Sigma_step=0.5, chunk=150):
    """Refit many data vectors D (M, N) with the SAME settings as `fit` (coarser grid for
    speed).  Returns profile best alpha, Sigma, their Delta-chi2=1 half-widths, chi2_min,
    marginal summaries and the no-wiggle Delta chi2 for each."""
    T, Ci = fit['T'], fit['Ci']
    D = np.atleast_2d(D)
    if D.shape[1] == len(fit['mask']) and D.shape[1] != int(fit['mask'].sum()):
        D = D[:, fit['mask']]                                          # all bins given -> keep fit bins
    ag = np.arange(fit['alpha_grid'][0], fit['alpha_grid'][-1] + 1e-9, alpha_step)
    Sg0 = fit['Sigma_grid']
    if len(Sg0) > 1:
        Sg = np.arange(Sg0[0], Sg0[-1] + 1e-9, Sigma_step)
        pri = np.interp(Sg, Sg0, fit['prior_m2lnp'])
    else:
        Sg, pri = Sg0, fit['prior_m2lnp']
    X = build_design(T, fit['edges_fit'], ag, Sg)
    chi2_nw = _gls(fit['Xn'][None], Ci, D)[0][0]
    M = D.shape[0]
    keys = ('alpha', 'sa', 'Sigma', 'sS', 'chi2_min', 'a_mean', 'a_std', 'a_q16', 'a_q84',
            'S_mean', 'S_std', 'S_q16', 'S_q84')
    out = {key: np.full(M, np.nan) for key in keys}
    for c0 in range(0, M, chunk):
        sl = slice(c0, min(c0 + chunk, M))
        c2, _, ldF = _gls(X, Ci, D[sl])
        chi2 = c2 + pri[None, :, None]                                # (A, S, m)
        m2 = chi2 + ldF[..., None]
        post = np.exp(-0.5 * (m2 - m2.min(axis=(0, 1), keepdims=True)))
        pa, pS = chi2.min(1), chi2.min(0)
        for j, mi in enumerate(range(sl.start, sl.stop)):
            if len(ag) > 1:
                ma = _summ_1d(ag, np.trapezoid(post[..., j], Sg, axis=1) if len(Sg) > 1 else post[:, 0, j])
                out['a_mean'][mi], out['a_std'][mi] = ma['mean'], ma['std']
                out['a_q16'][mi], out['a_q84'][mi] = ma['q'][0], ma['q'][2]
            if len(Sg) > 1:
                mS = _summ_1d(Sg, np.trapezoid(post[..., j], ag, axis=0) if len(ag) > 1 else post[0, :, j])
                out['S_mean'][mi], out['S_std'][mi] = mS['mean'], mS['std']
                out['S_q16'][mi], out['S_q84'][mi] = mS['q'][0], mS['q'][2]
            a, lo, hi, _ = _interval(ag, pa[:, j])
            out['alpha'][mi], out['sa'][mi] = a, 0.5 * (hi - lo)
            out['chi2_min'][mi] = pa[:, j].min()
            if len(Sg) > 1:
                S, slo, shi, _ = _interval(Sg, pS[:, j])
                out['Sigma'][mi], out['sS'][mi] = S, 0.5 * (shi - slo)
            else:
                out['Sigma'][mi] = Sg[0]
    out['dchi2'] = chi2_nw - out['chi2_min']
    return out


def run_mocks(fit, n_mock=300, seed=1, truth='nowiggle'):
    """
    Draw d = m_true + L z with C = L L^T (the covariance used in the fit) and refit.
    truth='nowiggle' : best no-wiggle model -> null distribution of the detection Delta chi2
    truth='best'     : best-fit model -> bias, scatter vs reported error, pulls, coverage
    """
    rng = np.random.default_rng(seed)
    if truth == 'nowiggle':
        th = _gls(fit['Xn'][None], fit['Ci'], fit['d'][None, :])[1][0, 0]
        mtrue = fit['Xn'] @ th
    else:
        mtrue = fit['X_best'] @ fit['theta']
    Lc = np.linalg.cholesky(fit['C'])
    D = mtrue[None, :] + (Lc @ rng.standard_normal((len(mtrue), n_mock))).T
    r = fit_many(D, fit)
    if truth == 'best':
        a_true, S_true = fit['alpha_grid'][fit['ij_best'][0]], fit['Sigma_grid'][fit['ij_best'][1]]
        pa = (r['alpha'] - a_true) / r['sa']
        r.update(alpha_true=a_true, Sigma_true=S_true, bias=np.nanmean(r['alpha'] - a_true),
                 sigma_mock=np.nanstd(r['alpha']), sigma_med=np.nanmedian(r['sa']),
                 pull_std=np.nanstd(pa), coverage68=np.nanmean(np.abs(pa[np.isfinite(pa)]) < 1),
                 frac_open=np.mean(~np.isfinite(pa)),
                 coverage68_marg=np.nanmean((r['a_q16'] <= a_true) & (a_true <= r['a_q84'])),
                 marg_std_med=np.nanmedian(r['a_std']), marg_mean_scatter=np.nanstd(r['a_mean']))
        if len(fit['Sigma_grid']) > 1:
            r.update(Sigma_bias=np.nanmean(r['S_mean'] - S_true), Sigma_scatter=np.nanstd(r['S_mean']),
                     Sigma_std_med=np.nanmedian(r['S_std']),
                     Sigma_coverage68=np.nanmean((r['S_q16'] <= S_true) & (S_true <= r['S_q84'])),
                     Sigma_open=np.mean(~np.isfinite(r['sS'])))
    return r


def difference_test(results, pairs, n_mock=400, seed=11, key='fit'):
    """
    Errors on differences between samples of the SAME box (e.g. Filament - Non-filament).
    The monopoles are correlated (measured cross-spectra), so sigma(Delta) is NOT
    sqrt(sigma_A^2 + sigma_B^2).  Correlated mocks of all samples, each refit; two truths:
      own  : each sample's own best fit        -> sigma(Delta)
      null : all samples share (alpha, Sigma) of the first sample in `pairs`
             -> p-value of the observed difference under "no difference"
    Estimators: alpha = profile best fit; Sigma = marginal posterior mean.
    """
    labs = list(results)
    fits = [results[l][key] for l in labs]
    edges = results[labs[0]]['s_edges']
    Cj = gaussian_cov_pk(edges, results[labs[0]]['spec'], [results[l]['spec_index'] for l in labs],
                         [results[l]['bias']['b'] for l in labs], results[labs[0]][key]['T'].lin,
                         results[labs[0]]['L'])
    nb = len(edges) - 1
    idx = np.concatenate([np.nonzero(f['mask'])[0] + i * nb for i, f in enumerate(fits)])
    Cf = Cj[np.ix_(idx, idx)]
    Lc = np.linalg.cholesky(Cf)
    rng = np.random.default_rng(seed)
    Z = (Lc @ rng.standard_normal((len(idx), n_mock))).T
    sizes = [int(f['mask'].sum()) for f in fits]
    cuts = np.cumsum([0] + sizes)

    def truth_vec(f, which, ref):
        if which == 'own':
            return f['X_best'] @ f['theta']
        # common (alpha, Sigma) = ref's; this sample's own B^2 and broadband refitted there
        Xr = build_design(f['T'], f['edges_fit'], np.array([ref['alpha']]), np.array([ref['Sigma']]))[0, 0]
        th = _gls(Xr[None], f['Ci'], f['d'][None, :])[1][0, 0]
        return Xr @ th

    out = {}
    for which in ('own', 'null'):
        ref = results[pairs[0][0]][key]
        est = {}
        for i, (l, f) in enumerate(zip(labs, fits)):
            mvec = truth_vec(f, which, ref)
            est[l] = fit_many(mvec[None, :] + Z[:, cuts[i]:cuts[i + 1]], f)
        out[which] = est
    rows = []
    for A, B in pairs:
        fa, fb = results[A][key], results[B][key]
        row = dict(pair=f"{A} - {B}")
        for par, kk, get in (('alpha', 'alpha', lambda f: f['alpha']),
                             ('Sigma', 'S_mean', lambda f: f['Sigma_marg']['mean'])):
            if par == 'Sigma' and len(fa['Sigma_grid']) < 2:
                continue
            dobs = get(fa) - get(fb)
            dm = out['own'][A][kk] - out['own'][B][kk]
            dn = out['null'][A][kk] - out['null'][B][kk]
            ok, okn = np.isfinite(dm), np.isfinite(dn)
            sig = np.std(dm[ok])
            naive = np.hypot(np.std(out['own'][A][kk][ok]), np.std(out['own'][B][kk][ok]))
            rho = np.corrcoef(out['own'][A][kk][ok], out['own'][B][kk][ok])[0, 1]
            p_null = np.mean(np.abs(dn[okn] - np.median(dn[okn])) >= np.abs(dobs))
            row[par] = dict(obs=dobs, sigma=sig, naive_quadrature=naive, rho=rho, z=dobs / sig,
                            p_null=p_null)
        rows.append(row)
    return rows, out


# =============================================================================================
# 9. Plots
# =============================================================================================
COLORS = ['steelblue', 'tomato', 'seagreen', 'purple', 'darkorange',
          'goldenrod', 'black', 'deeppink', 'teal', 'sienna']


def plot_fit(s, xi, m_w, m_nw, cov, fit, label=''):
    """Monopole with Gaussian errors, best BAO and no-wiggle fits, normalised residuals."""
    sig = np.sqrt(np.diag(cov))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw=dict(height_ratios=(3, 1.3)))
    ax1.errorbar(s, s**2 * xi, yerr=s**2 * sig, fmt='.', color='steelblue', capsize=2, label='data (Gaussian errors)')
    ax1.plot(s, s**2 * m_w, 'r-', lw=1.5, label='BAO fit')
    ax1.plot(s, s**2 * m_nw, 'k--', lw=1.2, alpha=0.6, label='no-wiggle fit')
    ax1.set_ylabel(r'$s^2\xi_0(s)$'); ax1.legend(fontsize=8)
    ax2.errorbar(s, (xi - m_w) / sig, yerr=1.0, fmt='.', color='steelblue', capsize=2)
    ax2.axhline(0, color='gray', ls='--'); ax2.set_ylim(-4, 4); ax2.set_ylabel(r'$\Delta\xi/\sigma$')
    ax2.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
    for ax in (ax1, ax2):
        ax.set_xlim(fit['fit_range'])
    fig.suptitle(f"{label}: alpha = {fit['alpha']:.4f} ± {fit['sigma_alpha']:.4f}, "
                 f"Sigma = {fit['Sigma']:.2f}, chi2 = {fit['chi2_min']:.1f}/{fit['dof']}", fontsize=10)
    fig.tight_layout()
    return fig


def _levels_2d(post):
    """Density levels enclosing 68% and 95% of a gridded 2D posterior."""
    p = np.sort(post.ravel())[::-1]; c = np.cumsum(p) / p.sum()
    return [p[np.searchsorted(c, q)] for q in (0.95, 0.68)]


def plot_alpha_sigma(results, key='fit_free', theory=None):
    """68/95% marginal contours in (alpha, Sigma) and the 1D marginals, all samples overlaid."""
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=(3, 1.2), height_ratios=(1.2, 3), hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0]); axa = fig.add_subplot(gs[0, 0], sharex=ax); axS = fig.add_subplot(gs[1, 1], sharey=ax)
    for (lab, r), col in zip(results.items(), COLORS):
        f = r[key]
        if len(f['Sigma_grid']) < 2:
            continue
        ax.contour(f['alpha_grid'], f['Sigma_grid'], f['post2d'].T, levels=_levels_2d(f['post2d']),
                   colors=col, linestyles=('--', '-'))
        ax.plot([], [], color=col, label=lab)
        axa.plot(f['alpha_grid'], f['post_alpha'] / f['post_alpha'].max(), color=col)
        axS.plot(f['post_Sigma'] / f['post_Sigma'].max(), f['Sigma_grid'], color=col)
    if theory is not None:
        for a_ in (ax, axS):
            a_.axhline(theory, color='k', ls=':', lw=1)
        ax.text(ax.get_xlim()[0], theory, '  pairwise Zel\'dovich (theory)', fontsize=7, va='bottom')
    ax.axvline(1.0, color='k', ls=':', lw=1)
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$\Sigma_{\rm nl}\ [h^{-1}{\rm Mpc}]$'); ax.legend(fontsize=8)
    plt.setp(axa.get_xticklabels(), visible=False); plt.setp(axS.get_yticklabels(), visible=False)
    axa.set_yticks([]); axS.set_xticks([])
    lo = min(r[key]['alpha_marg']['q'][0] for r in results.values()) - 0.03
    hi = max(r[key]['alpha_marg']['q'][2] for r in results.values()) + 0.03
    ax.set_xlim(lo, hi)
    return fig


# =============================================================================================
# 10. Drivers
# =============================================================================================
def analyse_sample(monopole_file, spec, label=None, spec_key=None, L=L_BOX, fit_range=(50, 150),
                   alpha_prior_Sigma=('gauss', 'theory', 2.0), free_Sigma=('flat', 0.0, 16.0),
                   n_mock=300, plot=True, save_plots=False, seed=1, T=None, lin=None):
    """
    One sample: bias -> covariance from its measured P(k) -> two fits -> null mocks -> plot.
      fit      : alpha measurement.  Sigma ~ N(theory, 2) (Chen+24 Sec. 7.2), the same prior for
                 every sample, so alpha differences are not driven by the Sigma assumption.
      fit_free : Sigma measurement.  Flat prior on Sigma, alpha free and marginalised.
                 Its 2D posterior gives the (alpha, Sigma) contours.
    Detection: Delta chi2 (no-wiggle - BAO) of `fit`, p-value from n_mock no-wiggle mocks.
    spec = load_spectra(pk_file); spec_key = this sample's label in it (default: from the
    monopole file name).
    """
    lin = lin or linear_theory()
    if T is None:
        k, P_w, P_nw = generate_templates(lin=lin)
        T = BAOTemplate(k, P_w, P_nw, lin)
    th = sigma_nl_theory(lin)
    s, xi, n_gal = load_monopole(monopole_file)
    label = label or os.path.basename(monopole_file)
    if n_gal is None:
        raise ValueError(f"{label}: file has no n_gal (re-run multidark_analysis.py)")
    s_edges = edges_from_centres(s)
    nbar = n_gal / L**3

    bi = estimate_bias_independent(s, xi, s_edges, nbar, T, L)
    spec_key = spec_key or spectrum_key_from_file(monopole_file)
    if spec_key not in spec['labels']:
        raise KeyError(f"{label}: '{spec_key}' not in spectra labels {spec['labels']}")
    spec_index = spec['labels'].index(spec_key)
    if int(spec['n_gal'][spec_index]) != int(n_gal):
        print(f"WARNING {label}: n_gal {n_gal} != {int(spec['n_gal'][spec_index])} in the spectra file")
    cov = gaussian_cov_pk(s_edges, spec, [spec_index], [bi['b']], T.lin, L)
    cov_g = gaussian_cov_box(s_edges, nbar, bi['b'], T.k, T.P_w, L)     # only for the printout

    kw = dict(T=T, fit_range=fit_range, Sigma_theory=th['Sigma_mono'], s_edges=s_edges)
    fit = fit_bao(s, xi, cov, Sigma_prior=alpha_prior_Sigma, **kw)
    fit_free = fit_bao(s, xi, cov, Sigma_prior=free_Sigma, **kw)
    null = run_mocks(fit, n_mock=n_mock, seed=seed + 1, truth='nowiggle')
    p_det = float(np.mean(null['dchi2'] >= fit['dchi2_wiggle']))

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"n_gal = {n_gal}   nbar = {nbar:.3e}   b (below-BAO) = {bi['b']:.3f} "
          f"[{bi['b_min']:.3f}-{bi['b_max']:.3f}]   fit {fit_range}")
    m_ = (s >= fit_range[0]) & (s <= fit_range[1])
    rat = np.sqrt(np.diag(cov)[m_] / np.diag(cov_g)[m_])
    print(f"covariance: Gaussian with MEASURED P(k)   (sigma / sigma_Poisson-Gaussian = {rat.min():.2f}-{rat.max():.2f})")
    print(f"Sigma_nl theory (pairwise Zel'dovich, Chen+24 eq.9) = {th['Sigma_mono']:.2f} h^-1 Mpc")
    for tag, f in (("alpha fit  [Sigma ~ N(theory,2)]", fit), ("Sigma free [flat 0-16]      ", fit_free)):
        q, qS = f['alpha_marg']['q'], f['Sigma_marg']['q']
        print(f"  {tag}: alpha = {f['alpha']:.4f} [{f['lo']:.4f}, {f['hi']:.4f}]  marg {q[1]:.4f} "
              f"(+{q[2]-q[1]:.4f}/-{q[1]-q[0]:.4f}) | Sigma = {f['Sigma']:.2f} [{f['S_lo']:.2f}, {f['S_hi']:.2f}] "
              f"marg {qS[1]:.2f} (+{qS[2]-qS[1]:.2f}/-{qS[1]-qS[0]:.2f}) | chi2 = {f['chi2_min']:.1f}/{f['dof']}")
    print(f"  BAO detection: Delta chi2 (no-wiggle - BAO) = {fit['dchi2_wiggle']:.1f}; null-mock p = {p_det:.3f}")

    m_w, m_nw = best_models(fit, s, s_edges)
    if plot:
        fig = plot_fit(s, xi, m_w, m_nw, cov, fit, label)
        if save_plots:
            os.makedirs(FIG_DIR, exist_ok=True)
            fig.savefig(os.path.join(FIG_DIR, f"bao_fit_{spec_key}.png"), dpi=150)
        plt.show()
    return dict(fit=fit, fit_free=fit_free, null=null, p_detection=p_det, bias=bi, cov=cov,
                model=m_w, model_nw=m_nw, s=s, xi=xi, s_edges=s_edges, n_gal=n_gal, L=L, theory=th,
                spec=spec, spec_index=spec_index)


def run_all(samples, pk_file, pairs=None, n_mock_diff=400, plot=True, save_plots=False,
            pk_keys=None, **kw):
    """
    samples = {'Full': monopole file, 'Filament': file, ...}; pk_file = spectra written by
    multidark_analysis.py in the same run (all samples must be in it).
    pairs = [('Filament', 'Non-filament'), ...]: differences to report (decide them before
    looking at the results).  pk_keys = {sample: label in pk_file} if the labels are not the
    <bin_desc> part of the monopole file names.  Other keywords go to analyse_sample.
    Returns {sample: analyse_sample output}; prints the summary table and the differences.
    """
    lin = linear_theory()
    k, P_w, P_nw = generate_templates(lin=lin)
    T = BAOTemplate(k, P_w, P_nw, lin)
    spec = load_spectra(pk_file)
    out = {lab: analyse_sample(f, spec, label=lab, spec_key=(pk_keys or {}).get(lab), T=T, lin=lin,
                               plot=plot, save_plots=save_plots, **kw)
           for lab, f in samples.items()}
    th = sigma_nl_theory(lin)['Sigma_mono']
    sep = '-' * 96
    print(f"\n{sep}\n SUMMARY   alpha: Sigma ~ N({th:.2f}, 2)    Sigma: flat prior, alpha marginalised\n{sep}")
    print(f"{'sample':<14}{'n_gal':>10}{'b':>6}{'alpha':>9}{'+/-':>8}{'<Sigma>':>9}{'+/-':>6}"
          f"{'chi2/dof':>10}{'det dchi2':>10}{'p':>8}")
    for lab, r in out.items():
        f, g = r['fit'], r['fit_free']
        print(f"{lab:<14}{r['n_gal']:>10d}{r['bias']['b']:>6.2f}{f['alpha']:>9.4f}{f['alpha_marg']['std']:>8.4f}"
              f"{g['Sigma_marg']['mean']:>9.2f}{g['Sigma_marg']['std']:>6.2f}"
              f"{f['chi2_min']:>6.1f}/{f['dof']:<3d}{f['dchi2_wiggle']:>10.1f}{r['p_detection']:>8.3f}")
    print("alpha +/- = marginal posterior std; <Sigma> = marginal posterior mean +/- std; "
          "p = fraction of no-wiggle mocks with a larger detection Delta chi2")
    print(sep)
    if pairs:
        for key, name in (('fit', 'alpha fit'), ('fit_free', 'Sigma-free fit')):
            rows, _ = difference_test(out, pairs, n_mock=n_mock_diff, key=key)
            print(f"Differences ({name}; correlated Gaussian mocks of the same box, n = {n_mock_diff}):")
            for rw in rows:
                for par in ('alpha', 'Sigma'):
                    if key == 'fit' and par == 'Sigma':
                        continue
                    v = rw[par]
                    print(f"  {rw['pair']:<26} Delta {par:<5} = {v['obs']:+.4f} ± {v['sigma']:.4f}  "
                          f"(z = {v['z']:+.2f}, null p = {v['p_null']:.3f}; corr = {v['rho']:+.2f}; "
                          f"error if independent {v['naive_quadrature']:.4f})")
    if plot:
        fig = plot_alpha_sigma(out, 'fit_free', theory=th)
        if save_plots:
            os.makedirs(FIG_DIR, exist_ok=True)
            fig.savefig(os.path.join(FIG_DIR, "alpha_sigma_contours.png"), dpi=150)
        plt.show()
    return out
