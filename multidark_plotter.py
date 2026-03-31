import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.interpolate import RegularGridInterpolator
import os

def xi_s_mu_to_xi_sigma_pi(xi_s_mu, s_edges, mu_edges, sigma_edges, pi_rebin):
    """Convert ξ(s, μ) to ξ(σ, π) via interpolation."""
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    interp = RegularGridInterpolator(
        (s_centers, mu_centers), xi_s_mu,
        bounds_error=False, fill_value=0.0
    )
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    max_pimax = sigma_edges[-1]
    pi_edges = np.arange(0, max_pimax + pi_rebin, pi_rebin)
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])
    S, P = np.meshgrid(sigma_centers, pi_centers, indexing='ij')
    s = np.sqrt(S**2 + P**2)
    mu = P / s
    mu[~np.isfinite(mu)] = 0.0
    points = np.stack([s.ravel(), mu.ravel()], axis=-1)
    xi_interp = interp(points).reshape(s.shape)
    return xi_interp, sigma_edges, max_pimax

def plot_xi_s_mu(xi, s_edges, mu_edges, title=None, output_folder=None, plotname="xi_s_mu.png",
                 min_sep=0.0, vmin_global=None, vmax_global=None,
                 contour_levels=None, contour_colors='black', contour_linewidths=1,
                 contour_kwargs=None):
    """Plot ξ(s, μ) with μ on X‑axis (decreasing) and s on Y‑axis."""
    mu_edges_rev = mu_edges[::-1]
    xi_rev = xi[:, ::-1]
    X, Y = np.meshgrid(mu_edges_rev, s_edges)
    C = xi_rev

    linthresh = 0.001
    if vmin_global is None or vmax_global is None:
        vmin = np.percentile(xi, 1)
        vmax = np.percentile(xi, 99)
        if vmin >= 0:
            vmin = -vmax / 2
        if vmax <= 0:
            vmax = -vmin / 2
    else:
        vmin, vmax = vmin_global, vmax_global
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X, Y, C, shading='flat', cmap='plasma', norm=norm)

    if contour_levels is not None:
        ckwargs = {'colors': contour_colors, 'linewidths': contour_linewidths}
        if contour_kwargs:
            ckwargs.update(contour_kwargs)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        mu_centers_rev = 0.5 * (mu_edges_rev[:-1] + mu_edges_rev[1:])
        Xc, Yc = np.meshgrid(mu_centers_rev, s_centers)
        ax.contour(Xc, Yc, C, levels=contour_levels, **ckwargs)

    ax.axhline(105, color='white', linestyle=':', linewidth=2, label='BAO scale', alpha=0.8)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$s$ [$h^{-1}$ Mpc]')
    ax.set_title(title if title else r'$\xi(s, \mu)$')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\xi(s,\mu)$')
    ax.set_ylim(bottom=min_sep, top=s_edges[-1])
    ax.set_xlim(right=0, left=1)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, plotname), dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_xi_s_combined(xi_s_list, labels, output_folder=None, filename='xi_s_combined.png'):
    """Plot combined ξ(s) curves."""
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.tab10.colors
    for i, ((s_centers, xi_s), label) in enumerate(zip(xi_s_list, labels)):
        ax.plot(s_centers, xi_s * s_centers**2, marker='o', linestyle='-',
                color=colors[i % len(colors)], label=label)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.axvline(105, color='k', linestyle=':', label='BAO scale')
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^{2}\xi(s)$')
    ax.set_title('ξ(s)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, filename), dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_monopoles_combined(monopoles_list, labels, output_folder=None, filename='xi0_combined.png'):
    """Plot combined monopoles ξ₀(s)."""
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.tab10.colors
    for i, ((s_centers, xi0), label) in enumerate(zip(monopoles_list, labels)):
        ax.plot(s_centers, xi0 * s_centers**2, marker='o', linestyle='-',
                color=colors[i % len(colors)], label=label)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.axvline(105, color='k', linestyle=':', label='BAO scale')
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^{2}\xi_0(s)$')
    ax.set_title('Monopoles ξ₀(s)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, filename), dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_monopole_to_xi_s_ratio(monopoles_list, xi_s_list, labels_list, output_folder):
    """Plot ratio ξ₀(s)/ξ(s)."""
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.tab10.colors
    ax.axhline(1, color='gray', linestyle='-', linewidth=1)
    ax.axvline(105, color='k', linestyle=':', label='BAO scale')
    for i, ((s_xi_s, xi_s), (s_xi0, xi0), label) in enumerate(zip(xi_s_list, monopoles_list, labels_list)):
        ratio = xi0 / xi_s
        ax.plot(s_xi_s, ratio, marker='o', linestyle='-',
                color=colors[i % len(colors)], label=label)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$\xi_0(s) / \xi(s)$')
    ax.set_title('Monopole to ξ(s) ratio')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, 'monopole_to_xi_s_ratio.png'), dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_xi_sigma_pi_from_xi_s_mu(xi_s_mu, s_edges, mu_edges, sigma_edges, pi_rebin,
                                  title=None, output_folder=None, plotname="xi_sigma_pi.png",
                                  min_sep=0.0, vmin_global=None, vmax_global=None):
    """Convert ξ(s, μ) to ξ(σ, π) and plot."""
    xi, sigma_edges, max_pimax = xi_s_mu_to_xi_sigma_pi(
        xi_s_mu, s_edges, mu_edges, sigma_edges, pi_rebin
    )
    pi_edges = np.arange(0, max_pimax + pi_rebin, pi_rebin)
    X, Y = np.meshgrid(sigma_edges, pi_edges)
    C = xi.T

    linthresh = 0.001
    if vmin_global is None or vmax_global is None:
        vmin = np.percentile(xi, 1)
        vmax = np.percentile(xi, 99)
        if vmin >= 0:
            vmin = -vmax / 2
        if vmax <= 0:
            vmax = -vmin / 2
    else:
        vmin, vmax = vmin_global, vmax_global
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X, Y, C, shading='flat', cmap='plasma', norm=norm)
    ax.set_xlabel(r'$\sigma$ [$h^{-1}$ Mpc]')
    ax.set_ylabel(r'$\pi$ [$h^{-1}$ Mpc]')
    ax.set_title(title if title else r'$\xi(\sigma, \pi)$')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\xi(\sigma,\pi)$')

    # Optional contour
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])
    Xc, Yc = np.meshgrid(sigma_centers, pi_centers)
    mask_region = (Xc >= 50) & (Xc <= 150) & (Yc >= 50) & (Yc <= 150)
    values_region = C[mask_region]
    if len(values_region) > 0:
        levels = np.percentile(values_region, [50, 70, 90])
        if len(np.unique(levels)) == len(levels) and np.all(np.diff(levels) > 0):
            ax.contour(Xc, Yc, C, levels=levels, colors='k', linewidths=1.5)

    # BAO circle
    theta = np.linspace(0, np.pi/2, 100)
    sigma_circle = 105 * np.cos(theta)
    pi_circle = 105 * np.sin(theta)
    ax.plot(sigma_circle, pi_circle, 'w--', linewidth=2, alpha=0.8, label=f's = {105} Mpc/h')
    ax.legend(loc='upper right')
    ax.set_xlim(left=min_sep)
    ax.set_ylim(bottom=min_sep)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, plotname), dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)