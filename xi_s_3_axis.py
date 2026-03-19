import sys
import os
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import multiprocessing 
from Corrfunc.utils import convert_3d_counts_to_cf
from Corrfunc.theory import DD

def rand_points(nrandom, lbox, axis, cat, rng):

  if axis == 0:
    x    = rng.uniform(0, lbox, size=nrandom)
  else:
    x    = rng.uniform(0, cat["x"].max()+1e-6, size=nrandom)
  
  if axis == 1:
    y    = rng.uniform(0, lbox, size=nrandom)
  else:
    y    = rng.uniform(0, cat["y"].max()+1e-6, size=nrandom)
  
  if axis == 2:
    z    = rng.uniform(0, lbox, size=nrandom)
  else:
    z    = rng.uniform(0, cat["z"].max()+1e-6, size=nrandom)

  return x, y, z

def calculate_xi(X_data, Y_data, Z_data, X_rand, Y_rand, Z_rand, bins, nthreads):
  """Calculate xi(s) using Corrfunc."""

  # Pair counts with Corrfunc
  DD_counts = DD(1, nthreads, bins, X_data, Y_data, Z_data, periodic=False)
  print("End DD")

  DR_counts = DD(0, nthreads, bins, X_data, Y_data, Z_data, X2=X_rand, Y2=Y_rand, Z2=Z_rand, periodic=False)
  print("End DR")
  
  RR_counts = DD(1, nthreads, bins, X_rand, Y_rand, Z_rand, periodic=False)
  print("End RR")

  # Convert to correlation function
  xi = convert_3d_counts_to_cf(len(X_data), len(X_data), len(X_rand), len(X_rand), \
      DD_counts, DR_counts, DR_counts, RR_counts)

  _DD    = DD_counts['npairs']
  _DR    = DR_counts['npairs']
  _RR    = RR_counts['npairs']

  return _DD, _DR, _RR, xi

seed = np.int32(sys.argv[1])
rng  = np.random.default_rng(seed)
factor_rand = 1
H_0  = 100.0#67.74
cosmo = FlatLambdaCDM(H0=H_0, Om0=0.3089)
lbox = 1000.0/cosmo.h
chunksize = 1_000_000
cut_mag = -21.0

smin, smax = 10, 150
ns_bins  = 30
bins_s  = np.linspace(smin, smax, ns_bins+1)
cbins = 0.5*(bins_s[1:] + bins_s[:-1])
nthreads = multiprocessing.cpu_count()
 
FILAMENTOS = False
folder = "./output_s_space_%d_factor_%d/" % (seed, factor_rand)
if not os.path.exists(folder):
  os.makedirs(folder)
   
filename = '../data/to_mock.csv'
count = 0
data_index, data_real, data_new = [], [], []
for chunk in pd.read_csv(filename, sep=",", na_values=r'\N', chunksize=chunksize):

  print("Step %d..." % (count))
  index = np.arange(len(chunk), dtype=np.int64) + count*chunksize
  x   = chunk["x"]/cosmo.h
  y   = chunk["y"]/cosmo.h
  z   = chunk["z"]/cosmo.h
  vx  = chunk["vx"]
  vy  = chunk["vy"]
  vz  = chunk["vz"]
  mag = chunk["magstarsdssr"]

  mask  = mag < cut_mag 
  
  mag = mag[mask]
  x   =  x[mask] % lbox # Periodicidad
  y   =  y[mask] % lbox #
  z   =  z[mask] % lbox #
  vx  = vx[mask]
  vy  = vy[mask]
  vz  = vz[mask]  
  index = index[mask]

  xn = x + vx/H_0
  yn = y + vy/H_0
  zn = z + vz/H_0
 
  mask  = xn >= 0
  mask *= yn >= 0
  mask *= zn >= 0
  mask *= xn < lbox # Podrian no estar
  mask *= yn < lbox # Podrian no estar
  mask *= zn < lbox # Podrian no estar

  mag = mag[mask]
  x   = x[mask]
  y   = y[mask]
  z   = z[mask]
  xn  = xn[mask]
  yn  = yn[mask]
  zn  = zn[mask]
  vx  = vx[mask]
  vy  = vy[mask]
  vz  = vz[mask]  
  index = index[mask]

  tmp     = np.vstack((x,y,z)).T 
  tmp_new = np.vstack((xn,yn,zn)).T 
  data_real.append(np.float64(tmp))
  data_new.append(np.float64(tmp_new))
  data_index.append(np.int64(index))

  print("End Step %d..." % (count))
  count += 1

print("Nloop %d" % count) 
data_real  = np.concatenate(data_real)
data_new   = np.concatenate(data_new)
data_index = np.concatenate(data_index)
cat = pd.DataFrame({'id':data_index, 'x':data_real[:,0], 'y':data_real[:,1], 'z':data_real[:,2], \
                                     'xn':data_new[:,0], 'yn':data_new[:,1], 'zn':data_new[:,2]})

if FILAMENTOS == True:

    # SI NECESITO EL MOCK SELECCIONADO LOS FILAMENTOS
    fname_fil = "../DATA/mock_withfilament.csv"
    cat_fil = pd.read_csv(fname_fil)
    # los campos de cat_fil son
    #'id','ifil', 'dfil', 'long', 'vx', 'vy', 'vz','vx_s', 'vy_s', 'vz_s'
    #
    #'id': id gal
    #'ifil': id_fil
    #'dfil': distancia perpendicular al filamento mas cercano [Mpc/h]
    #'long': longitud del filamento mas cercano [Mpc/h]
    #'vx',   'vy',   'vz':   versor del filamento mas cercano
    #'vx_s', 'vy_s', 'vz_s': versor suavizado del filamento mas cercano
    #'dfil', 'long', 'vx', 'vy', 'vz','vx_s', 'vy_s', 'vz_s'
    #'id','ifil', 'dfil', 'long', 'vx', 'vy', 'vz','vx_s', 'vy_s', 'vz_s'
    cat = pd.merge(cat, cat_fil, on='id', how='inner')
    cat = cat[cat["dfil"] < 2.0] # ACA FILTRO

for axis in range(3): 

  print("Axis %d" % axis)  
  assert(axis in [0,1,2])
      
  ndata = len(cat)
  nrandom = np.int64(factor_rand * ndata)
  X_rand, Y_rand, Z_rand = rand_points(nrandom, lbox, -1, cat, rng)
  X_data = cat["xn"].values if axis == 0 else cat["x"].values
  Y_data = cat["yn"].values if axis == 1 else cat["y"].values
  Z_data = cat["zn"].values if axis == 2 else cat["z"].values
        
  _DD, _DR, _RR, _xi = calculate_xi(X_data, Y_data, Z_data, X_rand, Y_rand, Z_rand, bins_s, nthreads)
  output_counts = folder + "axis_%d_counts_xi_ndata_%d_nrand_%d_seed_%d.out" % (axis, ndata, nrandom, seed)
  np.savetxt(output_counts, np.vstack((cbins, _DD, _DR, _RR, _xi)).T, delimiter=' ')
