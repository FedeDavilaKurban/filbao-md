# Archive

Code that is not part of the current pipeline, kept for reference. Nothing here is
imported by `src/`, `scripts/` or `notebooks/`. Paths inside these files point to the
old layout (`../data`, `../plots`), so they may need editing before they run again.

- `bao_fitting_gaussian_v3.ipynb`: the fitting notebook before the move to `src/bao_fitting.py`
  (October 2026), with its outputs. It is self-contained and has everything that was removed
  from the pipeline: the synthetic validation V0-V6 (why the Wallisch split, the ring-free
  template, wiggle-only dilation, the measured-P(k) covariance), the robustness options
  (other de-wiggle methods and broadbands), the Sigma fit with alpha = 1, the jackknife
  comparison, `compare_shapes`, and the methods table / literature notes.
- Earlier BAO-fitting notebooks and cells: `bao_fitting*.ipynb`, `bao_gaussian_pipeline.py`,
  `validate_synthetic_cell.py`, `covariance_diagnostics_cell.py`, `check_jackknife.py`,
  `flexible_bao_example.ipynb`
- Lightcone / observational pipeline (SDSS, redshift distributions, xi(sigma, pi)):
  `lightcone.py`, `lightcone_testing.ipynb`, `create_mock.py`, `multidark_2d_*.py`,
  `multidark_2dsmu_master.py`, `xi_s_3_axis.py`, `plot_monopoles_box_lightcone.ipynb`
- Exploration notebooks: `explore_monopoles.ipynb`, `inspect_paircouts.ipynb`,
  `sigma3_vs_dfil.ipynb`, `xi_s_plot.ipynb`
- Unused template caches: `template_ringfree.npz`, `templates_gauss_z0.00.npz`,
  `templates_hamann_old_z0.00.npz`, `templates_hamman.npz`
- `output_s_space_0_factor_1/`: old pair-count output
