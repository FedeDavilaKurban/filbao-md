import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import os
from scipy.spatial import cKDTree

def load_catalog(data_path="../data/to_mock.csv",
                 filament_path="../data/mock_withfilament.csv",
                 mag_col='magstarsdssr',
                 rename_mag='mag_abs_r',
                 dist_col='dist_fil'):
    """Load galaxy catalog and filament distances."""
    df = pd.read_csv(data_path, usecols=['x','y','z','vx','vy','vz', mag_col])
    df_fil = pd.read_csv(filament_path)
    if dist_col in df_fil.columns:
        df[dist_col] = df_fil[dist_col].values
    elif 'dfil' in df_fil.columns:
        df[dist_col] = df_fil['dfil'].values
    else:
        raise ValueError(f"No {dist_col} or dfil column found in filament file")
    df.rename(columns={mag_col: rename_mag}, inplace=True)
    return df

def select_sample(cat, mag_max, mag_col='mag_abs_r', test_dilute=1.0):
    """Apply magnitude cut and optional dilution."""
    cat = cat[cat[mag_col] < mag_max].copy()
    if test_dilute < 1.0:
        cat = cat.sample(frac=test_dilute, random_state=42).reset_index(drop=True)
    return cat

def split_by_dist_fil_bins(cat, mode, custom_intervals=None,
                           percentile_intervals=None,
                           nbins=4, fixed_edges=None,
                           dist_col='dist_fil'):
    """
    Split catalog by distance to filament.
    Returns (list of DataFrames, list of labels, edges).
    """
    values = cat[dist_col].values

    if mode == "custom_intervals":
        intervals = custom_intervals
        bins, labels = [], []
        for lo, hi in intervals:
            mask = (values >= lo) & (values <= hi)
            bins.append(cat.loc[mask].copy())
            labels.append(f"${lo:.1f} < r_{{fil}} \\leq {hi:.1f}$")
        return bins, labels, None

    elif mode == "percentile_intervals":
        if percentile_intervals is None:
            raise ValueError("percentile_intervals must be provided for mode 'percentile_intervals'")
        bins, labels = [], []
        for lo_pct, hi_pct in percentile_intervals:
            lo_val = np.percentile(values, lo_pct)
            hi_val = np.percentile(values, hi_pct)
            mask = (values >= lo_val) & (values <= hi_val)
            bins.append(cat.loc[mask].copy())
            labels.append(f"$r_{{fil}} \\in [{lo_val:.1f}-{hi_val:.1f}]$ Mpc/h")
        return bins, labels, None

    elif mode == "percentile":
        percentiles = np.linspace(0, 100, nbins + 1)
        edges = np.percentile(values, percentiles)
    elif mode == "equal_width":
        vmin, vmax = values.min(), values.max()
        edges = np.linspace(vmin, vmax, nbins + 1)
    elif mode == "fixed":
        edges = np.array(fixed_edges)
    else:
        raise ValueError(f"Unknown dist_bin_mode: {mode}")

    bins, labels = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        bins.append(cat.loc[mask].copy())
        labels.append(f"${lo:.1f} < r_{{fil}} \\leq {hi:.1f}$")
    return bins, labels, edges

def plot_magnitude_hist(cat, mag_max, output_folder, mag_col='mag_abs_r'):
    """Save magnitude histogram with cut line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cat[mag_col], bins=40, color="C00", alpha=0.8)
    ax.set_yscale("log")
    ax.axvline(mag_max, color="k", linestyle=":")
    ax.set_xlabel("Absolute magnitude r")
    ax.set_ylabel("Count")
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, "magnitude_hist.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

def plot_bin_data(gxs, label, output_folder, filename=None):
    """Plot 2D spatial slice and distance histogram."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    gxs_plot = gxs[gxs["z"] < 100.]
    h, xedges, yedges, im = axes[0].hist2d(
        gxs_plot["x"], gxs_plot["y"],
        bins=100, cmap="Blues",
        norm=SymLogNorm(linthresh=1, vmin=1, vmax=gxs_plot.shape[0]/500)
    )
    fig.colorbar(im, ax=axes[0], label="N. of galaxies")
    axes[0].set_xlabel("x [Mpc/h]")
    axes[0].set_ylabel("y [Mpc/h]")
    axes[0].set_title(label + f" (z < 100 Mpc/h, N={len(gxs_plot)})")
    axes[0].set_aspect('equal')

    axes[1].hist(gxs["dist_fil"], bins=40, density=True, color="C03", alpha=0.9)
    axes[1].set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
    axes[1].set_ylabel("PDF")
    plt.tight_layout()
    if filename is None:
        filename = f"bin_{label.replace('$','').replace(' ','_')}.png"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


from heapq import heappush, heappushpop

def compute_knn_distance(points, k=3, boxsize=None):
    """
    Compute distance to the k-th nearest neighbor for each point.
    If boxsize is given, uses periodic boundary conditions via 27-image queries.
    """
    if boxsize is None:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k+1)
        return dists[:, k]

    tree = cKDTree(points)
    # Generate the 27 periodic image shifts (including zero shift)
    shifts = np.array([(dx, dy, dz)
                       for dx in (-1, 0, 1)
                       for dy in (-1, 0, 1)
                       for dz in (-1, 0, 1)]) * boxsize

    # For each point, maintain a max-heap of the k smallest distances
    # (store negative to simulate a max‑heap)
    heaps = [[] for _ in range(len(points))]

    for shift in shifts:
        # Query all points shifted by this offset
        q_points = points + shift
        dists, indices = tree.query(q_points, k=k+1)
        for i in range(len(points)):
            d_i = dists[i]
            idx_i = indices[i]
            for d, j in zip(d_i, idx_i):
                if j == i or d == 0:   # skip self (distance zero)
                    continue
                # Insert into max-heap (store -d so smallest distances become largest negatives)
                if len(heaps[i]) < k:
                    heappush(heaps[i], -d)
                elif -heaps[i][0] > d:
                    heappushpop(heaps[i], -d)

    # Extract the k-th smallest distance from each heap
    knn_dists = []
    for i, heap in enumerate(heaps):
        if len(heap) < k:
            knn_dists.append(np.inf)
        else:
            # The largest negative corresponds to the k-th smallest distance
            knn_dists.append(-heap[0])
    return np.array(knn_dists)
    
# At the top of data_loader.py, after imports
COL_TO_LATEX = {
    'dist_fil': r'd_{\mathrm{fil}}',
    'Sigma_3': r'\Sigma_3',
}

def _get_1d_bins(values, col, mode, custom_intervals=None, percentile_intervals=None,
                 nbins=4, fixed_edges=None):
    """Return (masks, labels, edges) for a single variable."""
    col_latex = COL_TO_LATEX.get(col, col)  # use LaTeX name if available
    
    if mode == "custom_intervals":
        intervals = custom_intervals
        if intervals is None:
            raise ValueError("custom_intervals must be provided")
        masks, labels = [], []
        for lo, hi in intervals:
            mask = (values >= lo) & (values <= hi)
            masks.append(mask)
            labels.append(f"${lo:.1f} < {col_latex} \\leq {hi:.1f}$")
        return masks, labels, None

    elif mode == "percentile_intervals":
        if percentile_intervals is None:
            raise ValueError("percentile_intervals must be provided")
        masks, labels = [], []
        for lo_pct, hi_pct in percentile_intervals:
            lo_val = np.percentile(values, lo_pct)
            hi_val = np.percentile(values, hi_pct)
            mask = (values >= lo_val) & (values <= hi_val)
            masks.append(mask)
            labels.append(f"${col_latex} \\in [{lo_val:.1f}-{hi_val:.1f}]$ Mpc/h")
        return masks, labels, None

    elif mode == "percentile":
        percentiles = np.linspace(0, 100, nbins + 1)
        edges = np.percentile(values, percentiles)
    elif mode == "equal_width":
        vmin, vmax = values.min(), values.max()
        edges = np.linspace(vmin, vmax, nbins + 1)
    elif mode == "fixed":
        edges = np.array(fixed_edges)
    else:
        raise ValueError(f"Unknown binning mode: {mode}")

    masks, labels = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        masks.append(mask)
        labels.append(f"${lo:.1f} < {col_latex} \\leq {hi:.1f}$")
    return masks, labels, edges

def split_by_variable(cat, var_config):
    """
    Split catalog based on one or two variables.
    
    Parameters
    ----------
    cat : pd.DataFrame
        Input catalog.
    var_config : dict or list of dicts
        If dict: single variable binning.
        If list of two dicts: 2‑D binning (cartesian product).
        Each dict must contain:
            - 'col': column name
            - 'mode': binning mode (custom_intervals, percentile_intervals, percentile, equal_width, fixed)
            - optional parameters: custom_intervals, percentile_intervals, nbins, fixed_edges
    
    Returns
    -------
    bins : list of pd.DataFrame
        Subsets of the original catalog.
    labels : list of str
        Labels for each bin (suitable for plotting).
    metadata : list of dict
        For each bin, a dictionary containing bin edges and indices.
    """
    # Normalize to list
    if isinstance(var_config, dict):
        var_config = [var_config]

    # Collect per‑variable masks, labels, edges
    per_var_masks = []
    per_var_labels = []
    per_var_edges = []
    for config in var_config:
        col = config['col']
        values = cat[col].values
        mode = config['mode']
        custom_intervals = config.get('custom_intervals')
        percentile_intervals = config.get('percentile_intervals')
        nbins = config.get('nbins', 4)
        fixed_edges = config.get('fixed_edges')
        masks, labels, edges = _get_1d_bins(
            values, col, mode, custom_intervals, percentile_intervals, nbins, fixed_edges
        )
        per_var_masks.append(masks)
        per_var_labels.append(labels)
        per_var_edges.append(edges)

    # Combine
    if len(var_config) == 1:
        # 1D
        bins = [cat.loc[mask].copy() for mask in per_var_masks[0]]
        labels = per_var_labels[0]
        edges0 = per_var_edges[0]
        metadata = []
        for i, mask in enumerate(per_var_masks[0]):
            meta = {}
            if edges0 is not None:
                meta['edges'] = edges0
                meta['bin_index'] = i
            # Also store the interval limits for custom modes
            # (can be retrieved from labels, but store raw)
            meta['col'] = var_config[0]['col']
            metadata.append(meta)
        return bins, labels, metadata

    elif len(var_config) == 2:
        # 2D cartesian product
        bins = []
        labels = []
        metadata = []
        for i, mask1 in enumerate(per_var_masks[0]):
            for j, mask2 in enumerate(per_var_masks[1]):
                mask = mask1 & mask2
                if np.any(mask):
                    bins.append(cat.loc[mask].copy())
                    label = f"{per_var_labels[0][i]}, {per_var_labels[1][j]}"
                    labels.append(label)
                    meta = {
                        'col0': var_config[0]['col'],
                        'col1': var_config[1]['col'],
                    }
                    if per_var_edges[0] is not None:
                        meta['edges0'] = per_var_edges[0]
                        meta['bin_index0'] = i
                    if per_var_edges[1] is not None:
                        meta['edges1'] = per_var_edges[1]
                        meta['bin_index1'] = j
                    # Also store interval limits for custom modes
                    # Could add raw intervals if needed
                    metadata.append(meta)
        return bins, labels, metadata

    else:
        raise ValueError("Only 1 or 2 variables supported for splitting.")