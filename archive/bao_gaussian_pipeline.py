# ============================================================
# NOTEBOOK CELL: Gaussian-covariance BAO pipeline  (steps 1-3 + template fix)
#
# Paste in a new cell AFTER the existing function definitions of
# bao_fitting.ipynb (it reuses generate_templates, compute_sigma_nl,
# xi_damped (legacy A/B only) and load_monopole).
#
#   Step 0  ring-free template (the old hard k-cut gave Gibbs ringing) -> Template
#   Step 1  independent bias estimate                -> estimate_bias_independent
#   Step 2  Gaussian covariance with V = V_box        -> gaussian_cov_box
#   Step 3  exact profile/marginal chi2(alpha), sigma_alpha, Gaussian-mock
#           validation, Fourier-space forecast         -> fit_sample, run_mocks,
#                                                         fisher_forecast_alpha
#   Driver: run_all({'Full': file, ...}, L=1000., fit_range=(50,150), n_poly=3)
#
# Design rules (each is explained in the accompanying notes):
#   * the covariance never sees the fitted B  (no bias iteration)
#   * the covariance always uses the full box volume (no V_eff)
#   * linear nuisance parameters (B^2, broadband) are solved in closed form for
#     every alpha, so chi2(alpha) is an exact profile, not an optimiser output
#   * default model dilates ONLY the BAO wiggle (mode='dilate_wiggle'); the
#     whole-template dilation of the old code is kept as a cross-check
#     (mode='dilate_all') because it also uses the smooth shape to fix alpha
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ------------------------------------------------------------
# A0. Ringing-free BAO template
#
# PROBLEM in xi_damped(): the k-integral is cut hard at k_max = 0.8 while the
# SMOOTH part of P(k) is still large there (k P = 89 (Mpc/h)^2).  A hard cut
# gives Gibbs ringing  -k_max P(k_max) cos(k_max s) / (2 pi^2 s^2)  in xi(s),
# period 2 pi / 0.8 = 7.9 Mpc/h and amplitude up to ~2x the BAO bump itself.
# Since the ringing phase is cos(0.8 alpha s), it also moves fast with alpha.
#
# FIX: xi_model = xi_nw(s) + xi_wiggle(s; Sigma)
#   * xi_nw     : integral of the no-wiggle P extended to k = 8 with a mild
#                 Gaussian taper (a = 1 Mpc/h), so the integrand dies smoothly
#   * xi_wiggle : integral of (P_w - P_nw) e^{-k^2 Sigma^2/2} over the Hamann
#                 grid; the damping already kills it at k = 0.8 (no truncation)
# ------------------------------------------------------------
class Template:
    COSMO = dict(H0=67.7, ombh2=0.022, omch2=0.12, As=2.1e-9, ns=0.96)  # == generate_templates

    def __init__(self, k, P_w, P_nw, cache='template_ringfree.npz', a_taper=1.0,
                 kmax_ext=8.0, s_grid=None, force=False):
        import os
        self.k, self.P_w, self.P_nw = k, P_w, P_nw
        self.s_grid = np.arange(1.0, 400.001, 0.25) if s_grid is None else s_grid
        if (not force) and os.path.exists(cache):
            self.xi_nw_grid = np.load(cache)['xi_nw']
            print(f"Loaded ring-free template from {cache}")
        else:
            import camb
            c = self.COSMO
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=c['H0'], ombh2=c['ombh2'], omch2=c['omch2'])
            pars.InitPower.set_params(As=c['As'], ns=c['ns'])
            pars.set_matter_power(redshifts=[0], kmax=kmax_ext)
            r = camb.get_results(pars)
            kh, _, pk = r.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax_ext, npoints=60000)
            kd = np.linspace(1e-4, kmax_ext, 160001)
            Pd = np.exp(np.interp(kd, kh, np.log(pk[0])))
            Pnw = Pd.copy()
            lo = kd <= k.max()                      # Hamann no-wiggle below k_max,
            Pnw[lo] = np.interp(kd[lo], k, P_nw)    # smooth CAMB P above it
            integrand = kd * Pnw * np.exp(-(kd * a_taper)**2 / 2)
            self.xi_nw_grid = np.array([np.trapezoid(integrand * np.sin(kd * x), kd)
                                        / (2 * np.pi**2 * x) for x in self.s_grid])
            np.savez(cache, xi_nw=self.xi_nw_grid, s=self.s_grid)
            print(f"Built ring-free template -> {cache}")
        self._spl = CubicSpline(self.s_grid, self.xi_nw_grid)
        self._dW = (P_w - P_nw)

    def xi_nw(self, s):
        return self._spl(np.asarray(s, float))

    def xi_wig(self, s, Sigma):
        """damped BAO wiggle only: FT of (P_w - P_nw) e^{-k^2 Sigma^2/2}."""
        s = np.atleast_1d(np.asarray(s, float))
        damp = np.exp(-self.k**2 * Sigma**2 / 2.0)
        integrand = (self.k * self._dW * damp)[None, :] * np.sin(np.outer(s, self.k))
        return np.trapezoid(integrand, self.k, axis=1) / (2 * np.pi**2 * s)

    def xi(self, s, Sigma):
        """smooth + damped wiggle; vectorised over s."""
        return self.xi_nw(s) + self.xi_wig(s, Sigma)


class LegacyTemplate:
    """Your CURRENT template (hard cut at k_max): use only for A/B comparison."""
    def __init__(self, k, P_w, P_nw): self.k, self.P_w, self.P_nw = k, P_w, P_nw
    def xi(self, x, Sigma):  return xi_damped(x, self.k, self.P_w, self.P_nw, Sigma)
    def xi_nw(self, x):      return xi_damped(x, self.k, self.P_nw, self.P_nw, 0.0)
    def xi_wig(self, x, Sigma): return self.xi(x, Sigma) - self.xi_nw(x)


# ------------------------------------------------------------
# A. Gaussian covariance of the binned monopole in a periodic box
# ------------------------------------------------------------
def shell_avg_j0(k, s_lo, s_hi):
    """Volume-weighted average of j0(k s) over the shell s_lo<s<s_hi.
    int s^2 j0(ks) ds = [sin(ks) - ks cos(ks)] / k^3."""
    k = np.asarray(k, float)[None, :]
    a = np.asarray(s_lo, float)[:, None]
    b = np.asarray(s_hi, float)[:, None]
    F = lambda s: (np.sin(k * s) - k * s * np.cos(k * s)) / k**3
    return 3.0 * (F(b) - F(a)) / (b**3 - a**3)


_LATTICE_CACHE = {}
def lattice_mode_counts(L, kmax):
    """Number of Fourier modes n in Z^3\\{0} at each integer |n|^2 = m
    with k = (2 pi / L) |n| <= kmax.  Returns (k_m, count_m)."""
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
        v = v[v <= nmax**2]
        cnt += np.bincount(v, minlength=nmax**2 + 1)
    cnt[0] = 0.0                                   # drop k = 0
    m = np.nonzero(cnt)[0]
    out = (kf * np.sqrt(m), cnt[m])
    _LATTICE_CACHE[key] = out
    return out


def gaussian_cov_box(s_edges, nbar, bias, k, P_lin, L=1000.0, kmax=None,
                     shell_average=True, return_parts=False):
    """
    Gaussian covariance of the binned monopole xi_0 in a periodic cube of side L:

        Cov_ij = (2 / V^2) * sum_{n != 0}  P_t(k_n)^2  <j0>_i(k_n) <j0>_j(k_n),
        P_t    = bias^2 P_lin + 1/nbar,         V = L^3 (always the full box)

    * discrete lattice modes (the box has k_f = 2pi/L, not a continuum)
    * <j0>_i = shell average over the bin, because DD/RR averages over the bin
    * V is the full box.  A subsample is a biased tracer of the SAME field, so
      it has the same number of independent modes; its different density and
      bias enter only through nbar and bias.
    """
    s_edges = np.asarray(s_edges, float)
    s_lo, s_hi = s_edges[:-1], s_edges[1:]
    V = L**3
    # modes above k~3 change the binned variance by <1e-4 (5 Mpc shell average kills them) but cost minutes
    kmax = min(float(k.max()), 3.0) if kmax is None else min(float(kmax), float(k.max()))
    kk, cnt = lattice_mode_counts(L, kmax)
    Pt = bias**2 * np.interp(kk, k, P_lin) + 1.0 / nbar
    w = cnt * Pt**2
    if shell_average:
        J = shell_avg_j0(kk, s_lo, s_hi)
    else:
        J = np.sinc(np.outer(0.5 * (s_lo + s_hi), kk) / np.pi)
    cov = (2.0 / V**2) * (J * w) @ J.T
    if not return_parts:
        return cov
    hi = kk > 0.6                                   # convergence diagnostic
    cov_hi = (2.0 / V**2) * (J[:, hi] * w[hi]) @ J[:, hi].T
    return cov, dict(frac_var_k_gt_0p6=np.diag(cov_hi) / np.diag(cov))


# ------------------------------------------------------------
# B. Step 1: bias from large scales, independent of the BAO fit
# ------------------------------------------------------------
def estimate_bias_independent(s, xi, s_edges, nbar, T, L=1000.0,
                              ranges=((30, 60), (40, 70), (35, 75)),
                              primary=1, n_iter=4):
    """
    b^2 from xi_gal(s) = b^2 xi_lin(s) on scales BELOW the BAO bump, by
    generalised least squares with the Gaussian covariance (the covariance
    depends only weakly on b, so a few fixed-point iterations are enough).
    The BAO region is never used, so the covariance cannot be tuned to the
    quantity being fitted.  Returns the primary estimate plus the spread over
    ranges (the systematic from nonlinearity / scale-dependent bias).

    NOTE: b is defined relative to the TEMPLATE amplitude.  Because the
    covariance uses b^2 * P_template, any sigma8 mismatch cancels.
    """
    k, P_w = T.k, T.P_w
    xi_lin = T.xi(s, 0.0)                         # undamped linear xi (Sigma=0)
    out = []
    for (lo, hi) in ranges:
        m = (s >= lo) & (s <= hi)
        b = 1.0
        for _ in range(n_iter):
            C = gaussian_cov_box(s_edges, nbar, b, k, P_w, L)[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            x = xi_lin[m]
            b2 = (x @ Ci @ xi[m]) / (x @ Ci @ x)
            b = np.sqrt(max(b2, 1e-6))
        sig_b2 = 1.0 / np.sqrt(x @ Ci @ x)
        chi2 = (xi[m] - b2 * x) @ Ci @ (xi[m] - b2 * x)
        out.append(dict(range=(lo, hi), b=b, b2=b2, sigma_b2=sig_b2,
                        sigma_b=0.5 * sig_b2 / b, chi2=chi2, dof=int(m.sum()) - 1))
    bs = np.array([o['b'] for o in out])
    return dict(b=out[primary]['b'], b_min=bs.min(), b_max=bs.max(), detail=out)


# ------------------------------------------------------------
# C. Step 3: exact profile / marginal likelihood in alpha
#    model = B^2 xi_damped(alpha s; Sigma) + sum_i A_i (100/s)^i
#    -> linear in theta = (B^2, A_0..A_{n_poly-1}) for fixed (alpha, Sigma)
# ------------------------------------------------------------
def build_design(alpha_grid, s_fit, Sigma, T, n_poly, mode='dilate_wiggle'):
    """mode='dilate_wiggle': column = xi_nw(s) + xi_wig(alpha s)   (BAO-only dilation)
       mode='dilate_all'   : column = xi_nw(alpha s) + xi_wig(alpha s)  (whole template
                             dilated: also uses the SMOOTH SHAPE to constrain alpha)"""
    A, N = len(alpha_grid), len(s_fit)
    X = np.empty((A, N, 1 + n_poly))
    broad = np.column_stack([(100.0 / s_fit)**i for i in range(n_poly)])
    for a, al in enumerate(alpha_grid):
        if mode == 'dilate_all':
            X[a, :, 0] = T.xi(al * s_fit, Sigma)
        else:
            X[a, :, 0] = T.xi_nw(s_fit) + T.xi_wig(al * s_fit, Sigma)
        X[a, :, 1:] = broad
    return X


def design_nowiggle(s_fit, T, n_poly):
    X = np.empty((len(s_fit), 1 + n_poly))
    X[:, 0] = T.xi_nw(s_fit)
    X[:, 1:] = np.column_stack([(100.0 / s_fit)**i for i in range(n_poly)])
    return X


def _gls(X, Ci, D):
    """Closed-form GLS for every design matrix in X (shape (...,N,p)) and every
    data vector in D (shape (M,N)).  B^2 >= 0 enforced by refitting without the
    BAO column when the unconstrained B^2 is negative (exact for a convex
    quadratic with one bound).
    Returns chi2 (...,M), theta (...,M,p), logdetF (...)."""
    XtCi = np.einsum('...np,nm->...pm', X, Ci)
    F = XtCi @ X
    dg = 1.0 / np.sqrt(np.einsum('...pp->...p', F))           # column scaling
    Ft = F * dg[..., :, None] * dg[..., None, :]
    b = np.einsum('...pn,mn->...mp', XtCi, D)
    bt = b * dg[..., None, :]
    th = np.linalg.solve(Ft[..., None, :, :], bt[..., None])[..., 0] * dg[..., None, :]
    dCd = np.einsum('mn,nk,mk->m', D, Ci, D)
    chi2 = dCd - np.sum(b * th, -1)
    logdetF = np.linalg.slogdet(Ft)[1] - 2 * np.sum(np.log(dg), -1)
    neg = th[..., 0] < 0
    if np.any(neg):                                           # constrained refit
        Fr = F[..., 1:, 1:]; br = b[..., 1:]
        dgr = 1.0 / np.sqrt(np.einsum('...pp->...p', Fr))
        Frt = Fr * dgr[..., :, None] * dgr[..., None, :]
        thr = np.linalg.solve(Frt[..., None, :, :],
                              (br * dgr[..., None, :])[..., None])[..., 0] * dgr[..., None, :]
        chi2r = dCd - np.sum(br * thr, -1)
        chi2 = np.where(neg, chi2r, chi2)
        th0 = np.zeros_like(th); th0[..., 1:] = thr
        th = np.where(neg[..., None], th0, th)
    return chi2, th, logdetF


def _interval(alpha, chi2, delta=1.0):
    """Delta-chi2 = `delta` interval by linear interpolation. Returns
    (alpha_best, lo, hi, flags). lo/hi = nan if the crossing is outside the grid."""
    i0 = int(np.argmin(chi2))
    if 0 < i0 < len(alpha) - 1:                               # parabolic refinement
        y0, y1, y2 = chi2[i0 - 1], chi2[i0], chi2[i0 + 1]
        den = y0 - 2 * y1 + y2
        a_best = alpha[i0] + 0.5 * (y0 - y2) / den * (alpha[1] - alpha[0]) if den > 0 else alpha[i0]
    else:
        a_best = alpha[i0]
    thr = chi2[i0] + delta
    lo = hi = np.nan
    left = np.nonzero(chi2[:i0] > thr)[0]
    if len(left):
        j = left[-1]; lo = np.interp(thr, [chi2[j + 1], chi2[j]], [alpha[j + 1], alpha[j]])
    right = np.nonzero(chi2[i0:] > thr)[0]
    if len(right):
        j = i0 + right[0]; hi = np.interp(thr, [chi2[j - 1], chi2[j]], [alpha[j - 1], alpha[j]])
    return a_best, lo, hi, dict(at_edge=(i0 == 0 or i0 == len(alpha) - 1),
                                lo_open=np.isnan(lo), hi_open=np.isnan(hi))


def fit_sample(s, xi, cov_full, T, fit_range=(50, 150), n_poly=3,
               Sigma=None, alpha_grid=None, mode='dilate_wiggle'):
    """Profile + marginal likelihood in alpha for ONE covariance matrix."""
    Sigma = compute_sigma_nl(T.k, T.P_w) if Sigma is None else Sigma
    alpha_grid = np.arange(0.80, 1.2001, 0.0005) if alpha_grid is None else alpha_grid
    m = (s >= fit_range[0]) & (s <= fit_range[1])
    s_fit, d = s[m], xi[m]
    C = cov_full[np.ix_(m, m)]
    Ci = np.linalg.inv(C)

    X = build_design(alpha_grid, s_fit, Sigma, T, n_poly, mode)
    chi2, th, ldF = _gls(X, Ci, d[None, :])
    chi2, th, ldF = chi2[:, 0], th[:, 0, :], ldF
    a_best, lo, hi, flags = _interval(alpha_grid, chi2)
    i0 = int(np.argmin(chi2))

    # marginal posterior (flat priors on alpha and on the linear parameters)
    m2lnL = chi2 + ldF
    post = np.exp(-0.5 * (m2lnL - m2lnL.min()))
    post /= np.trapezoid(post, alpha_grid)
    mean = np.trapezoid(alpha_grid * post, alpha_grid)
    std = np.sqrt(np.trapezoid((alpha_grid - mean)**2 * post, alpha_grid))
    cdf = np.cumsum(post) * (alpha_grid[1] - alpha_grid[0])
    q16, q50, q84 = np.interp([0.1587, 0.5, 0.8413], cdf, alpha_grid)

    # curvature (Fisher-like) error from the profile
    j = np.clip(i0, 2, len(alpha_grid) - 3)
    h = alpha_grid[1] - alpha_grid[0]
    d2 = (chi2[j - 2] - 2 * chi2[j] + chi2[j + 2]) / (2 * h)**2
    sig_curv = np.sqrt(2.0 / d2) if d2 > 0 else np.nan

    # no-wiggle comparison (alpha irrelevant; linear fit)
    Xn = design_nowiggle(s_fit, T, n_poly)
    chi2_nw, _, _ = _gls(Xn[None], Ci, d[None, :])
    chi2_nw = float(chi2_nw[0, 0])

    n_lin = 1 + n_poly
    dof = len(s_fit) - n_lin - 1
    chi2_min = float(chi2[i0])
    return dict(alpha=a_best, lo=lo, hi=hi, sigma_dchi2=0.5 * (hi - lo),
                sigma_curv=sig_curv, post_mean=mean, post_std=std,
                post_q=(q16, q50, q84), flags=flags,
                chi2_min=chi2_min, dof=dof, p_low=stats.chi2.cdf(chi2_min, dof),
                p_high=stats.chi2.sf(chi2_min, dof),
                B2=float(th[i0, 0]), B2_neg_clipped=bool(th[i0, 0] == 0.0),
                chi2_nowiggle=chi2_nw, dchi2_wiggle=chi2_nw - chi2_min,
                alpha_grid=alpha_grid, chi2_profile=chi2, post=post,
                s_fit=s_fit, X=X, Ci=Ci, C=C, d=d, Xn=Xn, th_best=th[i0],
                n_poly=n_poly, Sigma=Sigma, fit_range=fit_range,
                T=T, mode=mode)


# ------------------------------------------------------------
# D. Gaussian-mock validation of the whole fitting machinery
# ------------------------------------------------------------
def run_mocks(fit, n_mock=500, seed=1, truth='wiggle', alpha_step=0.002):
    """
    Draw mocks d = m_true + L z,  C = L L^T  (the SAME covariance used in the
    fit), refit each with the same code, and report:
      * <chi2_min> and its distribution vs chi2(dof)
      * bias and scatter of alpha_hat vs the Delta-chi2=1 error
      * coverage of the 68% Delta-chi2=1 intervals and pull distribution
      * truth='nowiggle' -> null distribution of Delta chi2 (wiggle vs no-wiggle)
    This validates the machinery (grid, bounds, B^2>=0, profile errors).  It
    CANNOT validate the covariance itself: mocks are drawn from it.
    """
    rng = np.random.default_rng(seed)
    ag = np.arange(0.80, 1.2001, alpha_step)
    ia = int(np.argmin(np.abs(fit['alpha_grid'] - fit['alpha'])))
    Xtrue = fit['X'][ia] if truth == 'wiggle' else fit['Xn']
    th = fit['th_best'] if truth == 'wiggle' else None
    if truth == 'nowiggle':                       # best no-wiggle truth
        XtCi = fit['Xn'].T @ fit['Ci']
        th = np.linalg.solve(XtCi @ fit['Xn'], XtCi @ fit['d'])
        th[0] = max(th[0], 0.0)
    mtrue = Xtrue @ th
    Lc = np.linalg.cholesky(fit['C'])
    D = mtrue[None, :] + (Lc @ rng.standard_normal((len(mtrue), n_mock))).T

    Xg = build_design(ag, fit['s_fit'], fit['Sigma'], fit['T'], fit['n_poly'], fit['mode'])
    chi2, _, _ = _gls(Xg, fit['Ci'], D)           # (A, M)
    chi2_nw, _, _ = _gls(fit['Xn'][None], fit['Ci'], D)
    chi2_nw = chi2_nw[0]
    a_hat = np.empty(n_mock); half = np.empty(n_mock); cmin = np.empty(n_mock)
    for mi in range(n_mock):
        a, lo, hi, _ = _interval(ag, chi2[:, mi])
        a_hat[mi], cmin[mi] = a, chi2[:, mi].min()
        half[mi] = 0.5 * (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
    res = dict(chi2_min=cmin, dchi2=chi2_nw - cmin)
    if truth == 'wiggle':
        a_true = fit['alpha_grid'][ia]
        pull = (a_hat - a_true) / half
        ok = np.isfinite(pull)
        res.update(alpha_true=a_true, alpha_hat=a_hat, sigma_mock=np.nanstd(a_hat),
                   sigma_dchi2_median=np.nanmedian(half), pull_std=np.nanstd(pull),
                   coverage68=np.mean(np.abs(pull[ok]) < 1.0), frac_open=np.mean(~ok),
                   bias=np.nanmean(a_hat - a_true))
    return res


# ------------------------------------------------------------
# E. Independent forecast (Fourier space, Seo & Eisenstein style)
# ------------------------------------------------------------
def fisher_forecast_alpha(k, P_w, P_nw, bias, nbar, L, Sigma, kmin=0.02, kmax=0.4):
    """sigma_alpha from the wiggle part only, isotropic, Gaussian modes:
        F = V/(4 pi^2) int k^2 dk [ k dW/dk / P_t ]^2 ,  W = b^2 (P_w-P_nw) e^{-k^2 Sigma^2/2}
    Ignores broadband marginalisation and the finite fit range.  Compare with the
    mode='dilate_wiggle' config-space sigma_alpha; if the config-space value is much
    SMALLER, something other than the wiggle (the smooth shape) is constraining alpha."""
    sel = (k >= kmin) & (k <= kmax)
    kk = k[sel]
    W = bias**2 * (P_w - P_nw)[sel] * np.exp(-kk**2 * Sigma**2 / 2)
    dW = np.gradient(W, kk)
    Pt = bias**2 * P_nw[sel] + 1.0 / nbar + W
    F = (L**3 / (4 * np.pi**2)) * np.trapezoid(kk**2 * (kk * dW / Pt)**2, kk)
    return 1.0 / np.sqrt(F)


# ------------------------------------------------------------
# F. Driver for one monopole file
# ------------------------------------------------------------
def analyse_sample(monopole_file, L=1000.0, fit_range=(50, 150), n_poly=3,
                   bias=None, bias_scan=(0.8, 1.0, 1.2), n_mock=300,
                   compare_jackknife=True, template_kwargs=None, plot=True,
                   label=None, seed=1, template='ringfree', mode='dilate_wiggle'):
    k, P_w, P_nw, *_ = generate_templates(**(template_kwargs or {}))
    T = Template(k, P_w, P_nw) if template == 'ringfree' else LegacyTemplate(k, P_w, P_nw)
    Sigma = compute_sigma_nl(k, P_w)
    s, xi, cov_jk, n_jk, n_gal = load_monopole(monopole_file)
    if n_gal is None:
        raise ValueError("file lacks n_gal (re-run multidark_analysis.py)")
    ds = s[1] - s[0]
    s_edges = np.concatenate([s - ds / 2, [s[-1] + ds / 2]])
    nbar = n_gal / L**3
    label = label or monopole_file.split('/')[-1]

    # Step 1: independent bias (fitted B is NOT used for the covariance)
    bi = estimate_bias_independent(s, xi, s_edges, nbar, T, L)
    b = bi['b'] if bias is None else bias

    # Step 2: Gaussian covariance, V = V_box
    cov = gaussian_cov_box(s_edges, nbar, b, k, P_w, L)

    # Step 3: profile likelihood
    fit = fit_sample(s, xi, cov, T, fit_range, n_poly, Sigma, mode=mode)
    other = 'dilate_all' if mode == 'dilate_wiggle' else 'dilate_wiggle'
    fit_o = fit_sample(s, xi, cov, T, fit_range, n_poly, Sigma, mode=other)
    mk = run_mocks(fit, n_mock=n_mock, seed=seed)
    null = run_mocks(fit, n_mock=n_mock, seed=seed + 1, truth='nowiggle')
    p_det = float(np.mean(null['dchi2'] >= fit['dchi2_wiggle']))
    sig_fisher = {kmx: fisher_forecast_alpha(k, P_w, P_nw, b, nbar, L, Sigma, kmax=kmx)
                  for kmx in (0.3, 0.8)}

    # sensitivity of sigma_alpha to the assumed bias
    scan = {}
    for fac in bias_scan:
        c2 = gaussian_cov_box(s_edges, nbar, b * fac, k, P_w, L)
        f2 = fit_sample(s, xi, c2, T, fit_range, n_poly, Sigma, mode=mode)
        scan[fac] = (f2['alpha'], f2['sigma_dchi2'], f2['chi2_min'])

    # optional: jackknife covariance for comparison only
    jk = None
    if compare_jackknife and cov_jk is not None and n_jk is not None:
        mfit = (s >= fit_range[0]) & (s <= fit_range[1])
        n_b = int(mfit.sum()); h = (n_jk - n_b - 2) / (n_jk - 1)
        fj = fit_sample(s, xi, cov_jk / h, T, fit_range, n_poly, Sigma, mode=mode)  # Ci*h == Hartlap
        jk = fj

    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"n_gal={n_gal}  nbar={nbar:.3e}  V=V_box  Sigma_nl(theory)={Sigma:.3f}  template={template}")
    print("Step 1  independent bias (GLS on xi, below the BAO bump):")
    for o in bi['detail']:
        print(f"   range {o['range']}: b = {o['b']:.3f} +/- {o['sigma_b']:.3f}"
              f"   (chi2/dof = {o['chi2']:.1f}/{o['dof']})")
    print(f"   -> b used = {b:.3f}   (spread over ranges {bi['b_min']:.3f}-{bi['b_max']:.3f});"
          f"   sqrt(B^2) from BAO fit = {np.sqrt(max(fit['B2'],0)):.3f}  [diagnostic only]")
    print(f"Step 3  alpha  [model: {mode}]:")
    fl = fit['flags']
    print(f"   profile:   alpha = {fit['alpha']:.4f}  [{fit['lo']:.4f}, {fit['hi']:.4f}]  "
          f"(sigma = {fit['sigma_dchi2']:.4f}; curvature sigma = {fit['sigma_curv']:.4f})"
          + ("  <-- OPEN/EDGE" if fl['at_edge'] or fl['lo_open'] or fl['hi_open'] else ""))
    q = fit['post_q']
    print(f"   marginal:  median = {q[1]:.4f}  [{q[0]:.4f}, {q[2]:.4f}]  (mean +/- std = "
          f"{fit['post_mean']:.4f} +/- {fit['post_std']:.4f})")
    print(f"   chi2_min = {fit['chi2_min']:.2f}/{fit['dof']}  "
          f"(P[chi2 <= obs] = {fit['p_low']:.3f}, P[chi2 >= obs] = {fit['p_high']:.3f})")
    print(f"   B^2 = {fit['B2']:.3f}" + ("  (clipped at 0)" if fit['B2_neg_clipped'] else ""))
    print(f"   BAO detection: dchi2 = {fit['dchi2_wiggle']:.2f}  (sqrt = {np.sqrt(max(fit['dchi2_wiggle'],0)):.2f} naive); "
          f"null-mock p = {p_det:.3f}" + (f"  (< {1/n_mock:.3f}: more extreme than all {n_mock} null mocks)" if p_det == 0 else ""))
    print(f"   cross-check [{other}]: alpha = {fit_o['alpha']:.4f}  sigma = {fit_o['sigma_dchi2']:.4f}  "
          f"chi2_min = {fit_o['chi2_min']:.2f}   (sigma ratio {fit_o['sigma_dchi2']/fit['sigma_dchi2']:.2f}; "
          f"a big gap = alpha is being driven by the smooth SHAPE, not the wiggle)")
    print(f"Validation (Gaussian mocks from this covariance, {n_mock}):")
    print(f"   <chi2_min> = {mk['chi2_min'].mean():.2f} (expect ~{fit['dof']});  "
          f"alpha bias = {mk['bias']:+.4f};  std(alpha_hat) = {mk['sigma_mock']:.4f} "
          f"vs median dchi2=1 error = {mk['sigma_dchi2_median']:.4f}")
    print(f"   coverage of 68% intervals = {mk['coverage68']:.3f};  pull std = {mk['pull_std']:.3f};  "
          f"intervals open on grid: {mk['frac_open']:.1%}")
    print(f"Independent forecast sigma_alpha (Fourier, wiggle-only): "
          f"kmax=0.3: {sig_fisher[0.3]:.4f},  kmax=0.8: {sig_fisher[0.8]:.4f}")
    print("Sensitivity to the assumed bias (covariance only):")
    for fac, (a, sg, c2) in scan.items():
        print(f"   b x {fac:.1f}: alpha = {a:.4f}, sigma_alpha = {sg:.4f}, chi2_min = {c2:.2f}")
    if jk is not None:
        print(f"Jackknife (Hartlap-corrected; diagnostic only): alpha = {jk['alpha']:.4f}, "
              f"sigma_alpha = {jk['sigma_dchi2']:.4f}, chi2_min = {jk['chi2_min']:.2f}/{jk['dof']}"
              f"   |  Gaussian/JK sigma_alpha = {fit['sigma_dchi2']/jk['sigma_dchi2']:.2f}")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(fit['alpha_grid'], fit['chi2_profile'] - fit['chi2_min'], label='Gaussian (V_box)')
        if jk is not None:
            ax[0].plot(jk['alpha_grid'], jk['chi2_profile'] - jk['chi2_min'], '--', label='jackknife')
        ax[0].axhline(1, color='gray', ls=':'); ax[0].set_ylim(0, 6)
        ax[0].set_xlim(max(0.8, fit['alpha'] - 0.15), min(1.2, fit['alpha'] + 0.15))
        ax[0].set_xlabel(r'$\alpha$'); ax[0].set_ylabel(r'$\chi^2_{\rm profile}-\chi^2_{\min}$')
        ax[0].legend(fontsize=8); ax[0].set_title(label, fontsize=8)
        ax[1].hist(mk['alpha_hat'], bins=30, density=True, alpha=0.6, label='mock $\\hat\\alpha$')
        ax[1].plot(fit['alpha_grid'], fit['post'], 'k', label='marginal posterior (data)')
        ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_xlabel(r'$\alpha$'); ax[1].legend(fontsize=8)
        plt.tight_layout(); plt.show()
    return dict(fit=fit, fit_other=fit_o, bias=bi, cov=cov, mocks=mk, null=null, p_detection=p_det,
                fisher=sig_fisher, bias_scan=scan, jackknife=jk)


# ------------------------------------------------------------
# G. Run every sample and print one comparison table
# ------------------------------------------------------------
def run_all(samples, **kw):
    """samples = {'Full': 'path.npz', 'Filament': 'path.npz', ...}; kw -> analyse_sample."""
    out = {lab: analyse_sample(f, label=lab, **kw) for lab, f in samples.items()}
    sep = '-' * 100
    print(f"\n{sep}\n  SUMMARY  (Gaussian covariance, V = V_box, bias from large scales)\n{sep}")
    print(f"{'sample':<14}{'b':>6}{'alpha':>9}{'sigma(dchi2=1)':>16}{'68% marginal':>20}"
          f"{'chi2/dof':>10}{'P(<=)':>7}{'null-p(BAO)':>13}")
    for lab, r in out.items():
        f = r['fit']; q = f['post_q']
        print(f"{lab:<14}{r['bias']['b']:>6.2f}{f['alpha']:>9.4f}{f['sigma_dchi2']:>16.4f}"
              f"{'['+format(q[0],'.3f')+', '+format(q[2],'.3f')+']':>20}"
              f"{f['chi2_min']:>6.1f}/{f['dof']:<3d}{f['p_low']:>7.2f}{r['p_detection']:>13.3f}")
    print(sep)
    print("NOTE: all samples live in the same box -> their alpha errors are CORRELATED; the error on a")
    print("difference (e.g. filament - non-filament) is not the quadrature sum.  See notes (multi-tracer).")
    return out
