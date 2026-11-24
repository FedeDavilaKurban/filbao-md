# Reproduce the template-free peak fit of desilike_fits.ipynb (section 2/3c) for Full, Q1 and the 51
# random dilutions of Full with N(Q1); save results for the figure.
import sys, os, glob, re, json
import numpy as np
from scipy.special import erf
repo, data_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(repo, 'src'))
import bao_fitting as bf

mag, binsep, mono_sep = -20.8, 5.0, (5.0, 200.0)
fit_range = (50., 150.)
peak_mu_range, peak_w_range, peak_step = (85., 125.), (2., 25.), 0.5
lin = bf.linear_theory()
k, P_w, P_nw = bf.generate_templates(lin=lin)
T = bf.BAOTemplate(k, P_w, P_nw, lin)
Sigma_th = bf.sigma_nl_theory(lin)['Sigma_mono']
spec = bf.load_spectra(os.path.join(data_dir, 'pk/box/pk_XISMU_box_mag-20.8_dist_fil_cumQ4.npz'))
mono = lambda key: os.path.join(data_dir, f'monopoles/box/mag={mag:.1f}_sep={mono_sep[0]:.1f}-{mono_sep[1]:.1f}_binsep={binsep:.1f}_{key}.npz')
mu_grid = np.arange(peak_mu_range[0], peak_mu_range[1] + 1e-9, peak_step)
w_grid = np.arange(peak_w_range[0], peak_w_range[1] + 1e-9, peak_step)

def interval_68(x, dchi2):
    i0 = int(np.argmin(dchi2))
    up = [i for i in range(i0, len(x)) if dchi2[i] > 1]
    dn = [i for i in range(i0, -1, -1) if dchi2[i] > 1]
    hi = np.interp(1., dchi2[[up[0] - 1, up[0]]], x[[up[0] - 1, up[0]]]) if up else np.nan
    lo = np.interp(1., dchi2[[dn[0] + 1, dn[0]]], x[[dn[0] + 1, dn[0]]]) if dn else np.nan
    return lo, hi
def gauss_bins(edges, mu, w):
    z = (edges - mu) / (np.sqrt(2) * w)
    return np.sqrt(np.pi / 2) * w * np.diff(erf(z)) / np.diff(edges)
def refine(x, p):
    i = int(np.argmin(p))
    if 0 < i < len(x) - 1:
        y0, y1, y2 = p[i - 1:i + 2]; den = y0 - 2 * y1 + y2
        if den > 0:
            return float(x[i] + 0.5 * (x[1] - x[0]) * (y0 - y2) / den)
    return float(x[i])
def peak_fit(s, xi, Cf):
    edges = bf.edges_from_centres(s); Ci = np.linalg.inv(Cf)
    X0 = bf.design_nowiggle(T, edges)
    def gls(X):
        th = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ xi); r = xi - X @ th
        return float(r @ Ci @ r), th
    chi2_0 = gls(X0)[0]
    chi2 = np.full((len(mu_grid), len(w_grid)), chi2_0)
    for i, mu in enumerate(mu_grid):
        for j, w in enumerate(w_grid):
            c2, th = gls(np.column_stack([X0, gauss_bins(edges, mu, w)]))
            if th[-1] > 0:
                chi2[i, j] = c2
    pm, pw = chi2.min(1) - chi2.min(), chi2.min(0) - chi2.min()
    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    th = gls(np.column_stack([X0, gauss_bins(edges, mu_grid[i], w_grid[j])]))[1]
    return dict(mu=refine(mu_grid, pm), w=refine(w_grid, pw), h=max(th[-1], 0.), mu_iv=interval_68(mu_grid, pm),
                w_iv=interval_68(w_grid, pw), dchi2_peak=chi2_0 - chi2.min(), s=s, smooth=X0 @ th[:-1],
                sig=np.sqrt(np.diag(Cf)), xi=xi)

def prepare(key, cov_key=None, b_cov=None):
    s, xi, n = bf.load_monopole(mono(key)); edges = bf.edges_from_centres(s)
    ck = cov_key or key
    i_spec = spec['labels'].index(ck)
    if b_cov is None:
        b_cov = bf.estimate_bias_independent(s, xi, edges, n / bf.L_BOX**3, T)['b']
    C = bf.gaussian_cov_pk(edges, spec, [i_spec], [b_cov], lin)
    m = (s >= fit_range[0]) & (s <= fit_range[1])
    return s[m], xi[m], C[np.ix_(m, m)], b_cov

res = {}
for lab, key in [('Full', 'full'), ('Q1', 'dist_fil_Q1')]:
    s, xi, C, b = prepare(key); r = peak_fit(s, xi, C); r['b'] = b; res[lab] = r
    print(lab, 'b =', round(b, 3), 'mu =', round(r['mu'], 1), 'w =', round(r['w'], 1), r['w_iv'])
ref_key = 'FullrandomN-1249969r20261016'
s, xi, Cdil, bdil = prepare(ref_key)
r = peak_fit(s, xi, Cdil); r['b'] = bdil; res['dil'] = r
print('dil (ref seed) b =', round(bdil, 3), 'w =', round(r['w'], 1))
ws = []
for f in sorted(glob.glob(mono('FullrandomN-1249969r*'))):
    key = re.search(r'binsep=5\.0_(FullrandomN-\d+r\d+)\.npz', f).group(1)
    s_, xi_, n_ = bf.load_monopole(f); m = (s_ >= fit_range[0]) & (s_ <= fit_range[1])
    ws.append(peak_fit(s_[m], xi_[m], Cdil)['w'])
ws = np.array(ws)
print('ensemble N =', len(ws), 'w median/16/84:', np.percentile(ws, [50, 16, 84]).round(2),
      'fraction <= w(Q1):', np.mean(ws <= res['Q1']['w']).round(3))
np.savez(out, ws=ws, **{f'{lab}_{k}': np.asarray(v, dtype=float) for lab, r in res.items()
                       for k, v in r.items() if k in ('s', 'xi', 'smooth', 'sig', 'mu', 'w', 'h', 'w_iv', 'mu_iv', 'b')})
