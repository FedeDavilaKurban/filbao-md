# Raw data

Input catalogues, not tracked in git (too large).

- `to_mock.csv`: MultiDark galaxy mock (box, L = 1000 h^-1 Mpc). Columns used:
  `x, y, z` (h^-1 Mpc), `vx, vy, vz`, `magstarsdssr` (absolute r magnitude).
- `mock_withfilament.csv`: same galaxies, same order, with `dist_fil`
  (distance to the nearest DisPerSE filament, h^-1 Mpc).

`scripts/multidark_analysis.py` reads both from this folder.
