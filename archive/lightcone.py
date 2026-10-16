import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

h = 1
Om0 = 0.31

# ============================================================
# 1. Build angular mask from random catalog (chunked)
# ============================================================
nside = 256
npix = hp.nside2npix(nside)
counts = np.zeros(npix, dtype=np.int32)   # counter array

chunksize_rand = 5_000_000
for chunk in pd.read_csv("../data/lss_randoms_combined_cut_LARGE.csv",
                         usecols=['ra', 'dec'],
                         dtype={'ra': 'float32', 'dec': 'float32'},
                         chunksize=chunksize_rand):
    theta = np.radians(90.0 - chunk['dec'].values)
    phi   = np.radians(chunk['ra'].values)
    pix = hp.ang2pix(nside, theta, phi, nest=True)
    np.add.at(counts, pix, 1)
    del chunk, theta, phi, pix

min_rand = 5
mask_pix = counts >= min_rand

# ============================================================
# 2. Process galaxy file in chunks with periodic tiling
# ============================================================
main_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'magstarsdssr']
# Use int64 for ID to keep exact values
dtype_main = {col: 'float32' for col in main_cols}

df_fil = pd.read_csv("../data/mock_withfilament.csv", usecols=['dfil'])
if 'dist_fil' in df_fil.columns:
    dist_fil_all = df_fil['dist_fil'].values
elif 'dfil' in df_fil.columns:
    dist_fil_all = df_fil['dfil'].values
else:
    raise ValueError("No dist_fil/dfil column")

# Cosmology with H0=100 (so distances are in Mpc/h)
cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
z_grid = np.linspace(0, 1, 1000)
r_grid = cosmo.comoving_distance(z_grid).value
r_to_z = interp1d(r_grid, z_grid, bounds_error=False, fill_value="extrapolate")
c = 3e5  # km/s

# Maximum distance to consider (from z=0.2, safe upper bound)
z_max = 0.22
R_max = cosmo.comoving_distance(z_max).value   # Mpc/h

# Precompute all possible box shifts (only -1,0,1 because R_max < L)
n = np.array([-1, 0, 1])
nx_arr, ny_arr, nz_arr = np.meshgrid(n, n, n, indexing='ij')
nx_arr = nx_arr.flatten()
ny_arr = ny_arr.flatten()
nz_arr = nz_arr.flatten()
n_copies = len(nx_arr)   # = 27

survivors = []
chunksize_gal = 1_000_000
start_idx = 0
L = 1000.0

for chunk in pd.read_csv("../data/to_mock.csv",
                         usecols=main_cols,
                         dtype=dtype_main,
                         chunksize=chunksize_gal):
    # Extract arrays
    x = chunk['x'].values
    y = chunk['y'].values
    z = chunk['z'].values
    vx = chunk['vx'].values
    vy = chunk['vy'].values
    vz = chunk['vz'].values
    mag_r = chunk['magstarsdssr'].values

    # Corresponding filament distances for this chunk
    end_idx = start_idx + len(chunk)
    dist_fil_chunk = dist_fil_all[start_idx:end_idx]
    start_idx = end_idx

    # Center coordinates around observer
    x_c = x - L/2
    y_c = y - L/2
    z_c = z - L/2

    N_gal = len(x_c)

    # Process in batches to keep memory under control
    batch_size = 10000
    for i in range(0, N_gal, batch_size):
        idx_slice = slice(i, min(i+batch_size, N_gal))
        xc_b = x_c[idx_slice]
        yc_b = y_c[idx_slice]
        zc_b = z_c[idx_slice]
        vx_b = vx[idx_slice]
        vy_b = vy[idx_slice]
        vz_b = vz[idx_slice]
        mag_b = mag_r[idx_slice]
        dist_b = dist_fil_chunk[idx_slice]

        # Expand to (batch_size, n_copies)
        x_shifted = xc_b[:, np.newaxis] + nx_arr * L
        y_shifted = yc_b[:, np.newaxis] + ny_arr * L
        z_shifted = zc_b[:, np.newaxis] + nz_arr * L
        r = np.sqrt(x_shifted**2 + y_shifted**2 + z_shifted**2)

        # Keep only copies within R_max
        valid = r <= R_max
        if not np.any(valid):
            continue

        # Flatten valid entries
        flat_valid = valid.flatten()
        x_flat = x_shifted.flatten()[flat_valid]
        y_flat = y_shifted.flatten()[flat_valid]
        z_flat = z_shifted.flatten()[flat_valid]
        r_flat = r.flatten()[flat_valid]

        # Map each flat entry back to its original galaxy (for velocities, magnitudes, distances, and ID)
        gal_idx = np.tile(np.arange(len(xc_b)), n_copies)[flat_valid]
        vx_flat = vx_b[gal_idx]
        vy_flat = vy_b[gal_idx]
        vz_flat = vz_b[gal_idx]
        mag_flat = mag_b[gal_idx]
        dist_flat = dist_b[gal_idx]

        # Compute angles
        ra_rad = np.arctan2(y_flat, x_flat)
        ra_deg = np.degrees(ra_rad)
        ra_deg = np.mod(ra_deg, 360.0)
        dec_rad = np.arcsin(z_flat / r_flat)
        dec_deg = np.degrees(dec_rad)

        # Cosmological redshift
        z_cosmo_flat = r_to_z(r_flat)

        # LOS velocities
        nx_flat = x_flat / r_flat
        ny_flat = y_flat / r_flat
        nz_flat = z_flat / r_flat
        v_los_flat = vx_flat * nx_flat + vy_flat * ny_flat + vz_flat * nz_flat
        z_obs_flat = z_cosmo_flat + v_los_flat / c

        # Apply initial cuts (redshift + magnitude)
        mask_z_mag = (z_obs_flat > 0.05) & (z_obs_flat < 0.22) & (mag_flat < -20.5)
        if not np.any(mask_z_mag):
            continue

        # Apply mask to all arrays
        x_flat = x_flat[mask_z_mag]
        y_flat = y_flat[mask_z_mag]
        z_flat = z_flat[mask_z_mag]
        ra_deg = ra_deg[mask_z_mag]
        dec_deg = dec_deg[mask_z_mag]
        ra_rad = ra_rad[mask_z_mag]
        dec_rad = dec_rad[mask_z_mag]
        z_cosmo_flat = z_cosmo_flat[mask_z_mag]
        z_obs_flat = z_obs_flat[mask_z_mag]
        mag_flat = mag_flat[mask_z_mag]
        dist_flat = dist_flat[mask_z_mag]

        # Angular mask
        theta_gal = np.pi/2.0 - dec_rad
        phi_gal = ra_rad
        pix_gal = hp.ang2pix(nside, theta_gal, phi_gal, nest=True)
        mask_ang = mask_pix[pix_gal]
        if not np.any(mask_ang):
            continue

        # Apply angular mask
        x_flat = x_flat[mask_ang]
        y_flat = y_flat[mask_ang]
        z_flat = z_flat[mask_ang]
        ra_deg = ra_deg[mask_ang]
        dec_deg = dec_deg[mask_ang]
        ra_rad = ra_rad[mask_ang]
        dec_rad = dec_rad[mask_ang]
        z_cosmo_flat = z_cosmo_flat[mask_ang]
        z_obs_flat = z_obs_flat[mask_ang]
        mag_flat = mag_flat[mask_ang]
        dist_flat = dist_flat[mask_ang]

        # Redshift-space positions
        r_obs_flat = cosmo.comoving_distance(z_obs_flat).value
        # Recompute unit vectors (they have changed due to mask)
        nx_flat = x_flat / np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
        ny_flat = y_flat / np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
        nz_flat = z_flat / np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
        x_rsd = r_obs_flat * nx_flat
        y_rsd = r_obs_flat * ny_flat
        z_rsd = r_obs_flat * nz_flat

        # Collect survivors
        chunk_survivors = pd.DataFrame({
            'x_real': x_flat,
            'y_real': y_flat,
            'z_real': z_flat,
            'x_rsd': x_rsd,
            'y_rsd': y_rsd,
            'z_rsd': z_rsd,
            'ra_deg': ra_deg,
            'dec_deg': dec_deg,
            'z_cosmo': z_cosmo_flat,
            'z_obs': z_obs_flat,
            'dist_fil': dist_flat,
            'mag_r': mag_flat
        })
        survivors.append(chunk_survivors)

    # Free memory of this chunk
    del chunk, x_c, y_c, z_c, vx, vy, vz, mag_r, dist_fil_chunk

# ============================================================
# 3. Combine all survivors
# ============================================================
if not survivors:
    print("No galaxies passed the cuts.")
    exit()

out_df = pd.concat(survivors, ignore_index=True)
# Final redshift cut (tighten to avoid edge effects)
out_df = out_df[(out_df['z_obs'] > 0.07) & (out_df['z_obs'] < 0.2)]
print(f"Number of galaxies: {len(out_df)}")

# ============================================================
# 4. Save and plot (same as before)
# ============================================================
output_file = "../data/lightcone_real_and_rsd_withfil.csv"
out_df.to_csv(output_file, index=False)
print(f"Saved lightcone to: {output_file}")

# Diagnostics
print("Real space: x_c range: [{:.1f}, {:.1f}]".format(out_df['x_real'].min(), out_df['x_real'].max()))
print("Real space: r range: [{:.1f}, {:.1f}]".format(
    np.sqrt(out_df['x_real']**2+out_df['y_real']**2+out_df['z_real']**2).min(),
    np.sqrt(out_df['x_real']**2+out_df['y_real']**2+out_df['z_real']**2).max()))
print("Redshift space: x_rsd range: [{:.1f}, {:.1f}]".format(out_df['x_rsd'].min(), out_df['x_rsd'].max()))
print("Redshift space: r_obs range: [{:.1f}, {:.1f}]".format(
    np.sqrt(out_df['x_rsd']**2+out_df['y_rsd']**2+out_df['z_rsd']**2).min(),
    np.sqrt(out_df['x_rsd']**2+out_df['y_rsd']**2+out_df['z_rsd']**2).max()))

# ============================================================
# Plots (unchanged)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

# RA/Dec plot using stored degrees
plt.figure(figsize=(10,6))
plt.scatter(out_df['ra_deg'], out_df['dec_deg'], s=1, alpha=0.5)
plt.title("Light-cone sky projection (RA/Dec)")
plt.grid(True)
plt.savefig('../plots/lightcone_ra_dec.png', dpi=100)
plt.show()

# Redshift distribution
plt.figure(figsize=(10,6))
bins = np.linspace(0.05, 0.22, 51)
print(min(out_df['z_obs']), max(out_df['z_obs']), min(out_df['z_cosmo']), max(out_df['z_cosmo']))
plt.hist(out_df['z_obs'], bins=bins, histtype='step', lw=3, label='Observed z')
plt.hist(out_df['z_cosmo'], bins=bins, histtype='step', lw=3, ls='--', label='Cosmological z')
plt.ylabel("Number of Galaxies")
plt.legend()
plt.title("Redshift distribution of light-cone")
plt.grid(True)
plt.savefig('../plots/lightcone_redshift_distribution.png', dpi=100)
plt.show()

# Real-space distance distribution
r_real = np.sqrt(out_df['x_real']**2 + out_df['y_real']**2 + out_df['z_real']**2)
plt.figure(figsize=(10,6))
plt.hist(r_real, bins=50)
plt.xlabel("Real-space distance [Mpc/h]")
plt.ylabel("Number of Galaxies")
plt.title("Distribution of real-space distances")
plt.grid(True)
plt.show()

# Wedge plots (unchanged)
r_obs = np.sqrt(out_df['x_rsd']**2 + out_df['y_rsd']**2 + out_df['z_rsd']**2)
N_plot = min(5_000_000, len(out_df))
idx = np.random.choice(len(out_df), size=N_plot, replace=False)

x_trans_real = out_df['x_real'].values[idx]
x_trans_rsd  = out_df['x_rsd'].values[idx]
r_los_real = r_real.values[idx]
r_los_rsd  = r_obs.values[idx]

bins_x = 150
bins_y = 150
vmin = None
vmax = None

fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)

h0 = axes[0].hist2d(x_trans_real, r_los_real, bins=[bins_x, bins_y],
                     cmap='Blues', density=True, vmin=vmin, vmax=vmax)
axes[0].set_xlabel("X [Mpc/h]")
axes[0].set_ylabel("LOS distance [Mpc/h]")
axes[0].set_title("Real-space projection")

h1 = axes[1].hist2d(x_trans_rsd, r_los_rsd, bins=[bins_x, bins_y],
                     cmap='Blues', density=True, vmin=vmin, vmax=vmax)
axes[1].set_xlabel("X [Mpc/h]")
axes[1].set_title("Redshift-space projection")

fig.colorbar(h0[3], ax=axes[0], label="Density")
fig.colorbar(h1[3], ax=axes[1], label="Density")

plt.suptitle("Light-cone wedge plots (2D histogram)")
plt.tight_layout()
plt.savefig("../plots/lightcone_wedge_plots.png", dpi=100)
plt.show()