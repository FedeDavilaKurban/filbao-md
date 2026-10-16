"""
Galaxy catalogue and subsamples for the FilBAO box analysis.

    load_catalog          read the MultiDark mock and attach the filament distance
    select_sample         absolute-magnitude cut
    compute_knn_distance  distance to the k-th neighbour (for rho_3, the local density)
    split_by_variable     subsamples in one variable, or the 2D (cartesian) product of two
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load_catalog(data_path, filament_path, mag_col='magstarsdssr', rename_mag='mag_abs_r',
                 dist_col='dist_fil'):
    """Galaxy positions, velocities and magnitude, plus the distance to the nearest filament
    (same galaxies, same row order in both files)."""
    df = pd.read_csv(data_path, usecols=['x', 'y', 'z', 'vx', 'vy', 'vz', mag_col])
    df_fil = pd.read_csv(filament_path)
    if dist_col in df_fil.columns:
        df[dist_col] = df_fil[dist_col].values
    elif 'dfil' in df_fil.columns:
        df[dist_col] = df_fil['dfil'].values
    else:
        raise ValueError(f"No {dist_col} or dfil column found in filament file")
    df.rename(columns={mag_col: rename_mag}, inplace=True)
    return df


def select_sample(cat, mag_max, mag_col='mag_abs_r'):
    """Galaxies brighter than mag_max."""
    return cat[cat[mag_col] < mag_max].copy()


def compute_knn_distance(points, k=3, boxsize=None):
    """Distance from each point to its k-th nearest neighbour (periodic if boxsize is given)."""
    if boxsize is not None:
        points = points % boxsize
    tree = cKDTree(points, boxsize=boxsize)
    dists, _ = tree.query(points, k=k + 1)        # the first neighbour is the point itself
    return dists[:, k]


# LaTeX names used in the labels (the labels also become the file names, see
# make_bin_desc_from_label in scripts/multidark_analysis.py)
COL_TO_LATEX = {
    'dist_fil': r'd_{\mathrm{fil}}',
    'log_rho_3': r'\mathrm{log_{10}}\rho_3',
}


def _get_1d_bins(values, col, mode, custom_intervals=None, percentile_intervals=None):
    """Masks and labels for one variable.
      'custom_intervals'     : [(lo, hi), ...] in the variable's units, lo <= x <= hi
      'percentile_intervals' : [(p_lo, p_hi), ...] percentiles of the variable in this catalogue
    """
    col_latex = COL_TO_LATEX.get(col, col)
    masks, labels = [], []
    if mode == "custom_intervals":
        if custom_intervals is None:
            raise ValueError("custom_intervals must be provided")
        for lo, hi in custom_intervals:
            masks.append((values >= lo) & (values <= hi))
            labels.append(f"${lo:.1f} < {col_latex} \\leq {hi:.1f}$")
    elif mode == "percentile_intervals":
        if percentile_intervals is None:
            raise ValueError("percentile_intervals must be provided")
        for lo_pct, hi_pct in percentile_intervals:
            lo_val = np.percentile(values, lo_pct)
            hi_val = np.percentile(values, hi_pct)
            masks.append((values >= lo_val) & (values <= hi_val))
            labels.append(f"${col_latex} \\in [{lo_val:.1f}-{hi_val:.1f}]$ Mpc/h")
    else:
        raise ValueError(f"Unknown binning mode: {mode}")
    return masks, labels


def split_by_variable(cat, var_config):
    """
    Subsamples of `cat`.
    var_config: one dict, or a list of one or two dicts, each with
        'col' (column name), 'mode' ('custom_intervals' or 'percentile_intervals'),
        and 'custom_intervals' or 'percentile_intervals'.
    Two dicts give the cartesian product (e.g. filament x density); empty cells are skipped.
    Returns (bins, labels): list of DataFrames and their plot labels.
    """
    if isinstance(var_config, dict):
        var_config = [var_config]
    if len(var_config) not in (1, 2):
        raise ValueError("split_by_variable supports one or two variables")

    per_var = [_get_1d_bins(cat[c['col']].values, c['col'], c['mode'],
                            c.get('custom_intervals'), c.get('percentile_intervals'))
               for c in var_config]

    if len(var_config) == 1:
        masks, labels = per_var[0]
        return [cat.loc[m].copy() for m in masks], list(labels)

    (masks0, labels0), (masks1, labels1) = per_var
    bins, labels = [], []
    for m0, l0 in zip(masks0, labels0):
        for m1, l1 in zip(masks1, labels1):
            mask = m0 & m1
            if np.any(mask):
                bins.append(cat.loc[mask].copy())
                labels.append(f"{l0}, {l1}")
    return bins, labels
