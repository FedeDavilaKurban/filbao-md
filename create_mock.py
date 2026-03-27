import os
import time
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
import astropy.units as u
cosmo = FlatLambdaCDM(H0=100.0, Om0=0.3089)
cvel = const.c.to(u.km/u.second).value

zmin, zmax = 0.05, 0.15
#dmin = np.float64(cosmo.comoving_distance(zmin).value)
#dmax = np.float64(cosmo.comoving_distance(zmax).value)

z_space = np.linspace(0., 1., 500)
r_space = np.float64(cosmo.comoving_distance(z_space).value)
aux_dis2red = interp1d(r_space, z_space, kind="cubic")

lbox = 1000.0
print("Lbox: %f" % lbox)

chunksize = 1_000_000
folder = os.path.realpath('../data/')
archivo  = 'to_mock.csv'
filename = os.path.join(folder, archivo)
print(filename)

count = 0
data_index, data_real = [], []
for chunk in pd.read_csv(filename, sep=",", na_values=r'\N', chunksize=chunksize):

  ti = time.time()
  
  index = np.arange(len(chunk), dtype=np.int64) + count*chunksize
  x   = chunk["x"]
  y   = chunk["y"]
  z   = chunk["z"]
  vx  = chunk["vx"]
  vy  = chunk["vy"]
  vz  = chunk["vz"]
  mag = chunk["magstarsdssr"]

  mask = x < 0
  x[mask] += lbox
  mask = y < 0
  y[mask] += lbox
  mask = z < 0
  z[mask] += lbox
  
  mask = x >= lbox 
  x[mask] -= lbox
  mask = y >= lbox 
  y[mask] -= lbox
  mask = z >= lbox 
  z[mask] -= lbox

  x -= 0.5*lbox
  y -= 0.5*lbox
  z -= 0.5*lbox
  r  = np.sqrt(x**2 + y**2 + z**2)

  mask  = mag < -21.0
  
  mag = mag[mask]
  x   =  x[mask]
  y   =  y[mask]
  z   =  z[mask]
  vx  = vx[mask]
  vy  = vy[mask]
  vz  = vz[mask]  
  r   =  r[mask]
  index = index[mask]

  if np.sum(mask) == 0:
    continue

  x /= r
  y /= r
  z /= r
  longitud, latitud = hp.vec2ang(np.vstack((x, y, z)).T, lonlat=True)
  
  # calcula las velocidades peculiares
  # el producto interior entre las velocidades y el versor posicion
  vrad = (vx*x+vy*y+vz*z)
  redshift_true = aux_dis2red(r)
  redshift = redshift_true + vrad/cvel

  mask = (redshift > zmin)*(redshift < zmax)
  
  mag      = mag[mask]
  x        = x[mask]
  y        = y[mask]
  z        = z[mask]
  longitud = longitud[mask]
  latitud  = latitud[mask]
  redshift = redshift[mask]
  index    = index[mask]
  
  if np.sum(mask) == 0:
    continue

  dcom = cosmo.comoving_distance(redshift).value
  x *= dcom
  y *= dcom
  z *= dcom

  del vrad, vx, vy, vz, r, dcom
  
  tf = time.time()
  print("%d total time: %.2f" % (count, time.time()-ti))

  tmp = np.vstack((x,y,z,longitud,latitud,redshift,mag)).T 
  data_real.append(np.float32(tmp))
  data_index.append(np.int64(index))

  count += 1

print("Nloop %d" % count) 
data_real  = np.concatenate(data_real)
data_index = np.concatenate(data_index)

# Read ../data/mock_withfilament.csv
# Read in chunks
print('Reading filament data...')
dist_fil = []
for chunk in pd.read_csv('../data/mock_withfilament.csv', sep=",", na_values=r'\N', chunksize=chunksize):
    dist_fil.append(chunk["dfil"].values)
dist_fil = np.concatenate(dist_fil)
dist_fil = dist_fil[data_index]

cat = pd.DataFrame({'id':data_index, \
 'x':data_real[:,0],   'y':data_real[:,1],   'z':data_real[:,2], \
 'ra':data_real[:,3], 'dec':data_real[:,4], 'red':data_real[:,5], \
 'mag_abs_r':data_real[:,6], 'dist_fil':dist_fil})

# Angular mask
print('Applying angular mask...')
nside = 128
ang_mask = np.zeros(hp.nside2npix(nside), dtype=np.int32)
# Read from file ../data/lss_randoms_combined_cut_LARGE.csv
chunksize_rand = 5_000_000
for chunk in pd.read_csv("../data/lss_randoms_combined_cut_LARGE.csv",
                         usecols=['ra', 'dec'],
                         dtype={'ra': 'float32', 'dec': 'float32'},
                         chunksize=chunksize_rand):# Create haelpy mask from the randoms and apply to the data
    ra_chunk = chunk['ra'].values
    dec_chunk = chunk['dec'].values
    pix = hp.ang2pix(nside,ra_chunk,dec_chunk,lonlat=True,nest=True)
    np.add.at(ang_mask,pix,1)
min_rand = 1
mask_pix = ang_mask >= min_rand
galaxy_pix = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True, nest=True)
cat = cat[mask_pix[galaxy_pix]]

# Plot Ra Dec and redshift distribution
plt.scatter(cat["ra"], cat["dec"], s=1)
plt.xlabel("Ra")
plt.ylabel("Dec")
plt.title("Angular distribution")
plt.show()
plt.hist(cat["red"], bins=50)
plt.xlabel("Redshift")
plt.ylabel("Number of galaxies")
plt.title("Redshift distribution")
plt.show()


filename = '../data/mock_MULTIDARK_zmin_%.2f_zmax_%.2f.csv' % (zmin, zmax)
cat.to_csv(filename, encoding='utf-8', index=False)
