# Processed data

Written by `scripts/multidark_analysis.py`, read by `notebooks/final/bao_fitting.ipynb`.
Not tracked in git, except `cache/`.

- `pair_counts/box/`  DD(s, mu) per sample (reused unless `force_recompute_*` is set)
- `monopoles/box/`    xi_0(s) per sample: `s`, `xi0`, `n_gal`
- `xis/box/`          1D xi(s) per sample (only for the diagnostic plots)
- `pk/box/`           auto- and cross-power spectra of all samples of one run
                      (`pk_<run>.npz`, used for the covariance) and the Gaussian
                      covariance used for the error bands of the plots (`cov_xi0_<run>.npz`)
- `cache/`            CAMB linear P(k) and the wiggle / no-wiggle templates (tracked;
                      delete a file to recompute it, e.g. after changing the cosmology)
