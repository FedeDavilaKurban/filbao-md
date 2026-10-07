# ============================================================
# NOTEBOOK CELL: covariance diagnostics
# Paste in a new cell AFTER the function definitions of bao_fitting.ipynb
# (needs generate_templates, compute_sigma_nl, compute_gaussian_covariance,
#  load_monopole, _build_inv_cov, fit_bao) and BEFORE / instead of the
# `if __name__ == "__main__"` block.
#
# What it does for one monopole file:
#   1. compares sigma(s) from jackknife vs Gaussian covariance
#   2. eigenvalues / condition number / rank of the fit-range correlation matrix
#   3. refits the Version-B model under each covariance and reports chi2/dof, alpha
#   4. plots errors and correlation matrices
#
# Use a FIXED, independently motivated bias (not the fitted B), otherwise the
# Gaussian covariance is tuned to the data it is being used to fit.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt


def cov_diagnostics(monopole_file, bias, box_volume=1000.0**3,
                    effective_volumes=None,      # e.g. {'V_box': 1e9, 'V_eff': 2.33e8}
                    fit_range=(50, 150), n_poly=3,
                    template_kwargs=None, plot=True):

    k, P_w, P_nw, *_ = generate_templates(**(template_kwargs or {}))
    sigma_th = compute_sigma_nl(k, P_w)
    s, xi, cov_jk, n_jk, n_gal = load_monopole(monopole_file)
    if cov_jk is None or n_jk is None or n_gal is None:
        raise ValueError("File needs cov, n_jk and n_gal. Re-run multidark_analysis.py.")

    nbar = n_gal / box_volume
    m = (s >= fit_range[0]) & (s <= fit_range[1])
    n_bins = int(m.sum())
    n_jk = int(n_jk)

    if effective_volumes is None:
        effective_volumes = {'V_box': box_volume}

    covs = {'jackknife': cov_jk}
    for name, V in effective_volumes.items():
        covs[f'gauss[{name}]'] = compute_gaussian_covariance(s, nbar, bias, V, k, P_w)

    # ---- 1. error comparison -------------------------------------------
    sig = {n: np.sqrt(np.diag(c)) for n, c in covs.items()}
    print(f"\nFile: {monopole_file}")
    print(f"n_gal={n_gal}, nbar={nbar:.3e}, bias={bias}, n_jk={n_jk}, "
          f"fit bins={n_bins}, Hartlap={(n_jk - n_bins - 2) / (n_jk - 1):.3f}")
    header = f"{'s':>7} {'sigma_JK':>11}" + "".join(
        f" {n:>16} {'JK/' + n[:9]:>14}" for n in covs if n != 'jackknife')
    print("\n" + header)
    for i in np.where(m)[0]:
        row = f"{s[i]:7.1f} {sig['jackknife'][i]:11.3e}"
        for n in covs:
            if n == 'jackknife':
                continue
            row += f" {sig[n][i]:16.3e} {sig['jackknife'][i] / sig[n][i]:14.2f}"
        print(row)
    for n in covs:
        if n != 'jackknife':
            r = sig['jackknife'][m] / sig[n][m]
            print(f"  median JK/{n} = {np.median(r):.2f}   (range {r.min():.2f}-{r.max():.2f})")

    # ---- 2. conditioning --------------------------------------------------
    print("\nFit-range correlation-matrix conditioning:")
    corrs = {}
    for n, c in covs.items():
        sub = c[np.ix_(m, m)]
        d = np.sqrt(np.diag(sub))
        corr = sub / np.outer(d, d)
        corrs[n] = corr
        ev = np.linalg.eigvalsh(corr)
        print(f"  {n:<20} min eig={ev.min(): .2e}  max eig={ev.max():7.2f}  "
              f"cond={ev.max() / max(abs(ev.min()), 1e-300):.2e}  "
              f"rank={np.linalg.matrix_rank(sub)}/{n_bins}")

    # ---- 3. refit under each covariance --------------------------------
    print(f"\nVersion-B fit (Sigma_nl fixed at {sigma_th:.3f}, n_poly={n_poly}); "
          f"expected chi2/dof = 1 +/- {np.sqrt(2 / (n_bins - 2 - n_poly)):.2f}")
    for n, c in covs.items():
        src = 'jackknife' if n == 'jackknife' else 'gaussian'
        inv, sg, _ = _build_inv_cov(c, s, fit_range, src, n_jk, False)
        res, s_fit, xi_fit, _, n_free = fit_bao(
            s, xi, k, P_w, P_nw, inv_cov=inv, sigma_xi=sg,
            fit_range=fit_range, sigma_nl_fixed=sigma_th, n_poly=n_poly)
        dof = len(s_fit) - n_free
        print(f"  {n:<20} chi2={res.fun:7.2f}/{dof}  chi2/dof={res.fun / dof:.3f}  "
              f"alpha={res.x[0]:.4f}  B={res.x[1]:.3f}")

    # ---- 4. plots --------------------------------------------------------
    if plot:
        ncov = len(covs)
        fig, axes = plt.subplots(1, 1 + ncov, figsize=(5 * (1 + ncov), 4))
        for n in covs:
            axes[0].plot(s[m], s[m]**2 * sig[n][m], 'o-', label=n)
        axes[0].set_xlabel(r'$s\ [h^{-1}$Mpc]'); axes[0].set_ylabel(r'$s^2\sigma_\xi$')
        axes[0].legend(fontsize=8)
        for ax, (n, corr) in zip(axes[1:], corrs.items()):
            im = ax.imshow(corr, origin='lower', vmin=-1, vmax=1, cmap='RdBu_r',
                           extent=[s[m][0], s[m][-1], s[m][0], s[m][-1]])
            ax.set_title(n, fontsize=9)
        fig.colorbar(im, ax=axes[1:].tolist(), shrink=0.8)
        plt.show()

    return covs


# ---- example usage (edit paths / biases) ------------------------------
# V_fil = 559189 / (2400560 / 1e9)   # the old "Fix 4" volume, to test it explicitly
# cov_diagnostics(
#     '../data/monopoles/box/mag=-21.2_sep=5.0-150.0_binsep=5.0_full.npz',
#     bias=0.9,
#     template_kwargs=dict(save_path='templates_hamman.npz', force_recompute=False))
# cov_diagnostics(
#     '../data/monopoles/box/mag=-21.2_sep=5.0-150.0_binsep=5.0_0.0-d_mathrmfilleq3.0.npz',
#     bias=1.5, effective_volumes={'V_box': 1e9, 'V_eff': V_fil},
#     template_kwargs=dict(save_path='templates_hamman.npz', force_recompute=False))
