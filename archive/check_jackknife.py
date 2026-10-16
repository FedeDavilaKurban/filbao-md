"""
Brute-force check of the jackknife pair-subtraction identity

    H_loo = H_full - H_in - 2*H_cross
          == DDsmu(autocorr=1) run directly on the N - N_k catalogue

Where to put it: same folder as multidark_correlation.py / multidark_data_loader.py.
Run with:  python check_jackknife.py

It calls the real worker (mc._jk_worker_shared), so it tests the code you
actually run, on a random 200k-galaxy subsample so it finishes in seconds.
Pair counts are integers (stored as float64, exact below 2^53), so the
match should be EXACT, not approximate.
"""
import numpy as np
from Corrfunc.theory.DDsmu import DDsmu
from multidark_data_loader import load_catalog, select_sample
import multidark_correlation as mc

# ---- settings: keep identical to multidark_analysis.py ----
L = 1000.0
mag_max = -21.2
min_sep, max_sep, bin_size = 5.0, 150.0, 5.0
n_sub_per_side = 4
n_test = 200_000          # subsample size for speed
nthreads = 8
nbins_mu = 10             # any value works; both sides use the same one
test_subs = [0, 21, 63]   # a corner, an interior, and the last cell

# ---- data ----
cat = select_sample(load_catalog(), mag_max, test_dilute=1.0)
rng = np.random.default_rng(42)
idx = rng.choice(len(cat), size=min(n_test, len(cat)), replace=False)
x = np.ascontiguousarray(cat["x"].values[idx], dtype=np.float64)
y = np.ascontiguousarray(cat["y"].values[idx], dtype=np.float64)
z = np.ascontiguousarray(cat["z"].values[idx], dtype=np.float64)

nbins_s = int(round((max_sep - min_sep) / bin_size))
s_bins = np.linspace(min_sep, max_sep, nbins_s + 1)

# ---- same sub-volume assignment as compute_jackknife_monopole_covariance ----
sub_edges = np.linspace(0, L, n_sub_per_side + 1)
i_idx = np.clip(np.searchsorted(sub_edges, x, side='right') - 1, 0, n_sub_per_side - 1)
j_idx = np.clip(np.searchsorted(sub_edges, y, side='right') - 1, 0, n_sub_per_side - 1)
k_idx = np.clip(np.searchsorted(sub_edges, z, side='right') - 1, 0, n_sub_per_side - 1)
particle_sub = i_idx * n_sub_per_side**2 + j_idx * n_sub_per_side + k_idx


def dd_auto(X, Y, Z):
    res = DDsmu(autocorr=1, nthreads=nthreads, binfile=s_bins, mu_max=1.0,
                nmu_bins=nbins_mu, X1=X, Y1=Y, Z1=Z,
                periodic=True, boxsize=L, verbose=False)
    return res['npairs'].reshape(nbins_s, nbins_mu).astype(np.float64)


print(f"N = {len(x)}, {n_sub_per_side**3} sub-volumes, "
      f"{nbins_s} s-bins x {nbins_mu} mu-bins")
H_full = dd_auto(x, y, z)

# initialise the module-level globals the worker reads (1 thread, as in the Pool)
mc._init_worker(x, y, z, particle_sub, s_bins, 1.0, nbins_mu, L, 1)

all_ok = True
for sub in test_subs:
    _, H_in, H_cross = mc._jk_worker_shared(sub)

    H_loo_new = H_full - H_in - 2.0 * H_cross     # current code
    H_loo_old = H_full - H_in - H_cross           # previous (buggy) formula

    m_out = particle_sub != sub
    H_brute = dd_auto(x[m_out], y[m_out], z[m_out])

    d_new = np.max(np.abs(H_loo_new - H_brute))
    d_old = np.max(np.abs(H_loo_old - H_brute))
    ok = np.array_equal(H_loo_new, H_brute)
    all_ok &= ok
    print(f"sub {sub:2d}: N_in={int((~m_out).sum()):5d} | "
          f"max|new - brute| = {d_new:.1f} | max|old - brute| = {d_old:.3e} | "
          f"{'PASS' if ok else 'FAIL'}")

print("\nALL PASS: pair subtraction is exact." if all_ok else
      "\nFAIL: pair subtraction does not match brute force - do NOT launch the long run.")
