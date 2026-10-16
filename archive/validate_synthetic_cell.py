# ============================================================
# NOTEBOOK CELL: validate the pipeline on synthetic data with KNOWN truth
# Paste AFTER the bao_gaussian_pipeline cell.  ~2-4 min.
#
# What it establishes (and what it cannot):
#   [V1] template accuracy vs a converged high-k reference   (needs camb)
#   [V2] end-to-end calibration: pull ~ N(0,1), <chi2_min> ~ dof, unbiased alpha
#   [V3] wiggle-only vs whole-template dilation vs Fourier forecast
# It cannot validate the covariance against reality: synthetic noise is drawn
# from the same Gaussian formula.  That needs independent boxes / mocks.
# ============================================================
def validate_synthetic(L=1000.0, n_gal=2400560, b_true=1.25, a_true=1.0,
                       fit_range=(50, 150), n_poly=3, n_runs=100, seed=5,
                       check_reference=True, template_kwargs=None):
    k, P_w, P_nw, *_ = generate_templates(**(template_kwargs or {}))
    T = Template(k, P_w, P_nw); Leg = LegacyTemplate(k, P_w, P_nw)
    Sigma = compute_sigma_nl(k, P_w)
    s_edges = np.arange(5.0, 150.1, 5.0); s = 0.5 * (s_edges[:-1] + s_edges[1:])
    nbar = n_gal / L**3
    ag = np.arange(0.8, 1.2001, 0.002)

    if check_reference:
        import camb
        c = Template.COSMO
        pars = camb.CAMBparams(); pars.set_cosmology(H0=c['H0'], ombh2=c['ombh2'], omch2=c['omch2'])
        pars.InitPower.set_params(As=c['As'], ns=c['ns']); pars.set_matter_power(redshifts=[0], kmax=8.0)
        kh, _, pk = camb.get_results(pars).get_matter_power_spectrum(minkh=1e-4, maxkh=8.0, npoints=60000)
        kd = np.linspace(1e-4, 8.0, 160001); Pd = np.exp(np.interp(kd, kh, np.log(pk[0])))
        w = np.exp(-(kd * 1.5)**2 / 2); sr = np.arange(50.0, 150.1, 5.0)
        ref = np.array([np.trapezoid(kd * Pd * w * np.sin(kd * x), kd) / (2 * np.pi**2 * x) for x in sr])
        for name, tm in (("legacy", Leg), ("ring-free", T)):
            e = tm.xi(sr, 0.0) - ref
            print(f"[V1] {name:10s} template: max|xi - xi_ref| = {np.abs(e).max():.2e}  "
                  f"rms = {np.sqrt(np.mean(e**2)):.2e}   (BAO bump ~ 6e-4)")

    cov = gaussian_cov_box(s_edges, nbar, b_true, k, P_w, L)
    Lc = np.linalg.cholesky(cov)
    truth = b_true**2 * (T.xi_nw(s) + T.xi_wig(a_true * s, Sigma)) + 1.5e-4 * (100 / s) - 1e-4
    rng = np.random.default_rng(seed)

    print(f"\n[V2] end-to-end, {n_runs} synthetic boxes, alpha_true = {a_true}, b_true = {b_true}")
    for name, tm in (("ring-free", T), ("legacy", Leg)):
        pulls, ah, sg, ch, bs = [], [], [], [], []
        rr = np.random.default_rng(seed)
        for _ in range(n_runs):
            d = truth + Lc @ rr.standard_normal(len(s))
            bi = estimate_bias_independent(s, d, s_edges, nbar, tm, L)
            c = gaussian_cov_box(s_edges, nbar, bi['b'], k, P_w, L)
            f = fit_sample(s, d, c, tm, fit_range, n_poly, Sigma, alpha_grid=ag)
            ah.append(f['alpha']); sg.append(f['sigma_dchi2']); ch.append(f['chi2_min']); bs.append(bi['b'])
            if np.isfinite(f['sigma_dchi2']): pulls.append((f['alpha'] - a_true) / f['sigma_dchi2'])
        pulls = np.array(pulls)
        print(f"   {name:10s}: alpha_hat = {np.mean(ah):.4f} +/- {np.std(ah):.4f} (median profile sigma "
              f"{np.nanmedian(sg):.4f}) | pull std = {pulls.std():.2f}, coverage68 = {np.mean(np.abs(pulls)<1):.2f} "
              f"| <chi2_min> = {np.mean(ch):.1f} (dof {f['dof']}) | b_est = {np.mean(bs):.2f} (true {b_true})")

    print("\n[V3] sigma_alpha: wiggle-only dilation vs whole-template dilation vs Fourier forecast")
    d = truth + Lc @ rng.standard_normal(len(s))
    sw = fit_sample(s, d, cov, T, fit_range, n_poly, Sigma, alpha_grid=ag, mode='dilate_wiggle')['sigma_dchi2']
    sa = fit_sample(s, d, cov, T, fit_range, n_poly, Sigma, alpha_grid=ag, mode='dilate_all')['sigma_dchi2']
    F = fisher_forecast_alpha(k, P_w, P_nw, b_true, nbar, L, Sigma, kmax=0.3)
    print(f"   wiggle-only: {sw:.4f}   whole-template: {sa:.4f}   Fourier forecast: {F:.4f}")
    print("   (whole-template << forecast  =>  alpha is being constrained by the smooth shape)")
