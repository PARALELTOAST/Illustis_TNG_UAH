import numpy as np
import tqdm
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib import ticker
from matplotlib.colors import LogNorm
from numba import njit
from scipy.ndimage import gaussian_filter as gf
import illustris_python as il
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

halo_blues= np.loadtxt('blue_galaxies_TNG300_1__cut_rband_DUSTcorr_18_014_sdss_density.csv')#, delimiter=',', skiprows=0)
#halo_blues = np.loadtxt('red_galaxies_TNG300_1__cut_rband_DUSTcorr_18_014_sdss_density.csv')



x = halo_blues[:,0]
y = halo_blues[:,1]
z = halo_blues[:,2]
m_tot = halo_blues[:,4]
subhalo_mag_r_dust_blue = halo_blues[:,5]
print('Loaded The RvB')

print('length of pos',len(x))

# Applying the mask after filtering to maintain the length
#subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask * Flag_mask]
mag_sun_r = 4.42  # absolute magnitude of sun in r-band.

lum_r = 10.0 ** ((subhalo_mag_r_dust_blue - mag_sun_r) / -2.5)
lum_avg_den = sum(lum_r.flatten()) / 303. ** 3
print('avg lum den',lum_avg_den)
###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600#600  # 1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303.  # Units of Mpc
cell_len = box_len / num_cells
print('cell length',cell_len)
LD_Overden = np.zeros([num_cells, num_cells, num_cells])
#z = subhalo_coord[Lum_mask * Flag_mask, 2]  # / (0.6774*10**3)
#y = subhalo_coord[Lum_mask * Flag_mask, 1]  # / (0.6774*10**3)
#x = subhalo_coord[Lum_mask * Flag_mask, 0]  # / (0.6774*10**3)


x_bins = np.linspace(0.0, 303.0, num_cells + 1)
y_bins = np.linspace(0.0, 303.0, num_cells + 1)
z_bins = np.linspace(0.0, 303.0, num_cells + 1)

x_idx = np.digitize(x, x_bins)
y_idx = np.digitize(y, y_bins)
z_idx = np.digitize(z, z_bins)

lum_xyz = np.array([x,y,z]).T

LD_Overden, lum_edges = np.histogramdd(lum_xyz, bins = (x_bins, y_bins, z_bins), weights = lum_r / lum_avg_den / (box_len / num_cells) ** 3)

print("number of galaxies for smoothing:", np.size(x))
print("dimensions of LD grid", np.shape(LD_Overden))
print("number of bins with galaxies (lum values):", np.size(np.nonzero(LD_Overden.flatten())))

########### smoothing using Gaussian kernel ####################################
smooth_param = 1.2 / (box_len/num_cells) # 2 #Units of Mpc
LD_Overden = gf(LD_Overden, sigma=0.6 * smooth_param)

# plt.imshow(LD_Overden[0:2,:,:].sum(0).T, origin = 'lower', extent = (0,303, 0, 303), norm = LogNorm())
# plt.ylabel('Y [Mpc]')
# plt.xlabel('X [Mpc]')
# plt.title('$\delta_{LD}$')
# plt.colorbar()
# plt.show()
np.save('ld_overden_blue.npy', LD_Overden)