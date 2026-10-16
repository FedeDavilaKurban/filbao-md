# FilBAO

BAO in galaxies near cosmic filaments: is the BAO peak of filament galaxies different in
position (α) or damping (Σ) from that of the full sample, beyond what their number density
and local density explain? Measured in the MultiDark box (SDSS and Flagship to follow).

## The plan

1. **Subsamples:** Full, Filament, Non-filament, random dilutions of Full matched to the
   Filament and Non-filament numbers, and Filament / Non-filament × dense / underdense.
2. **Measurements:** ξ₀(s) and the auto/cross P(k) of every subsample
   (`scripts/multidark_analysis.py`).
3. **BAO fit:** `run_all` (`src/bao_fitting.py`, run from `notebooks/final/bao_fitting.ipynb`):
   covariance from the measured P(k), fit on the (α, Σ) grid, null mocks, differences.
4. **Results:** one table with α, Σ, Δχ²_det and p per sample, Δα and ΔΣ for the chosen
   pairs, the monopoles with their fits and the (α, Σ) contours.

## Repository structure

```
src/                    Functions used by the pipeline
  multidark_data_loader.py   catalogue, magnitude cut, rho_3, subsamples
  multidark_correlation.py   pair counts -> xi(s, mu) -> monopole; 1D xi(s)
  multidark_power.py         auto/cross power spectra; Gaussian covariance for plot errors
  multidark_plotter.py       diagnostic plots
  bao_fitting.py             template, covariance, fit, mocks, run_all, BAO-fit figures
scripts/
  multidark_analysis.py      plan steps 1-2 (configuration at the top)
notebooks/final/
  bao_fitting.ipynb          plan steps 3-4
notebooks/exploratory/
  calibration_mocks.ipynb    error calibration of the fits (pull std, coverage)
data/raw/               Input catalogues (not tracked, see data/raw/README.md)
data/processed/         Pair counts, monopoles, spectra (not tracked); cache/ (tracked)
figures/                BAO-fit figures; diagnostics/<run>/ for the script's plots (not tracked)
docs/                   Notes
archive/                Earlier code, not used by the pipeline (see archive/README.md)
```

## Running

1. Put `to_mock.csv` and `mock_withfilament.csv` in `data/raw/`.
2. Run `python scripts/multidark_analysis.py` once per split (edit `split_vars` at the top):
   * `dist_fil` only, with `dilution_controls = 'all'`: Full, Filament, Non-filament and the
     two dilution controls;
   * `dist_fil` and `log_rho_3`, with `dilution_controls = []`: the four filament × density cells.

   Each run writes the monopoles, one spectra file for all its samples, and the diagnostic
   plots (`figures/diagnostics/<run>/`). Pair counts are reused unless `force_recompute_*` is set.
3. Open `notebooks/final/bao_fitting.ipynb`: one `run_all` per script run (all samples of a
   `run_all` must be in the same spectra file).

Settings (binning, cuts, fit range, priors) are in the configuration block of the script and
in the notebook's `run_all` call; the cosmology and snapshot redshift at the top of
`src/bao_fitting.py`.

## Requirements

Python 3 with numpy, scipy, pandas, matplotlib and Corrfunc (needs GSL). CAMB is only needed
to recompute the caches in `data/processed/cache/` (e.g. after changing the cosmology).
