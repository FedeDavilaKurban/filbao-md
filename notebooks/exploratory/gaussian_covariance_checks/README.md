# Gaussian covariance checks for the filament samples

Is the Gaussian covariance of the fits (`bf.gaussian_cov_pk`, built from each sample's measured P(k)) valid for samples as
clumpy as Fil (excess noise ≈ 16× Poisson at k = 0.15)?

## random_halves.ipynb

Splits a sample at random into two halves. The difference of their monopoles has no cosmic-variance (signal × signal)
term, so it tests only the noise part of the covariance, which is where Fil is unusual. The Gaussian formula predicts the
covariance of the difference, and ⟨χ²⟩/n should be 1. Fil is compared with its random control (Full dil nFil).

Checked on synthetic catalogues before use: a clumpy sample (clumps of ~8 galaxies) and a Poisson one both pass
(⟨χ²⟩/n = 1.04 and 0.98 ± 0.03). Giving the clumpy sample a deliberately wrong (Poisson-only) noise model gives
⟨χ²⟩/n = 4.65, so the test detects a wrong noise model. With 50 splits it resolves about a 6% error in the noise
covariance at 2σ.

## Why there is no jackknife test

Considered: compare jackknife / Gaussian variance ratios of Fil and of its random control, expecting the jackknife's bias
to cancel. It does not. On Poisson catalogues in the 1 h⁻¹Gpc box (125 jackknife cubes, true variance known exactly), the
delete-one jackknife overestimates the variance of ξ₀ at s ≈ 130 h⁻¹Mpc by a factor of 18, 67 and 258 for N = 10⁵,
3×10⁵ and 9×10⁵. With the cross-pair weighting of Mohammad & Percival (2022) the factor is still 2–23. Removing a cube
also removes all its pairs with the surroundings, and their number fluctuates with the Poisson count of galaxies around
the cube; in the full periodic box this fluctuation cancels, so it is not part of the true variance. The bias therefore
depends on N and on how noise-dominated a sample is, and would differ between Fil and a random sample. This is consistent
with the 1.5–6× overestimate noted when the jackknife was dropped from the pipeline.

The old jackknife code itself (git history before commit 6b4343c) was correct as code: the pair subtraction
H_full − H_in − 2 H_cross is exact. Its one imprecision, the random pairs of the box with one cube removed taken as
N(N−1)/V_rem × V_shell, biases the mean of the leave-one-out monopoles (by ≈ −0.006 at 145 h⁻¹Mpc for 125 cubes) but
hardly affects the covariance. The exact count, used for the tests above, replaces V_rem V_shell by
(L³ − 2a³) V_shell + ∫_shell A(r) d³r, with a the cube side and A(r) the overlap of the cube with its translate by r.
