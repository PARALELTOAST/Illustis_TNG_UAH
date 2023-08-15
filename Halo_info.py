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

#########################################Luminosity Information#############################
############################ START OF HALO INFORMATION ###############################################
dust_corrected_file = '/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5'

with h5py.File(dust_corrected_file, "r") as partData:subhalo_mag_r_dust = np.asarray(partData["/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:, 2, 0], dtype='f4')

min_mag_r = min(subhalo_mag_r_dust)
max_mag_r = max(subhalo_mag_r_dust)
print("min and max of mag_r, dust corrected:", min_mag_r, max_mag_r)
path = '/home/benjamin/Thesis/Illustris_TNG/tng_300'

halo_fields = ['GroupPos', 'GroupMass', 'GroupMassType', 'Group_M_Crit200', 'Group_M_Crit500', 'Group_R_Crit200', 'Group_R_Crit500', 'GroupNsubs', 'GroupLenType', 'GroupLen']  # check how much of gas lies within R200, R500. See ratio of gas mass to total mass. Compare to EAGLE Toni's paper.
subhalo_fields = ['SubhaloFlag', 'SubhaloMass', 'SubhaloMassType', 'SubhaloPos', 'SubhaloStellarPhotometrics', 'SubhaloGrNr']
halos = il.groupcat.loadHalos(path, 99, fields=halo_fields)
subhalos = il.groupcat.loadSubhalos(path, 99, fields=subhalo_fields)

halo_coord_z = halos['GroupPos'][:, 2] / (0.6774 * 10 ** 3)
halo_coord_y = halos['GroupPos'][:, 1] / (0.6774 * 10 ** 3)
halo_coord_x = halos['GroupPos'][:, 0] / (0.6774 * 10 ** 3)
halo_rad_crit200 = halos['Group_R_Crit200'][:] / (0.6774 * 10 ** 3)
halo_mass = halos['GroupMass'] / 0.6774 * 10 ** 10  # cluster mass, includes all particle types
halo_mass_gas = halos['GroupMassType'][:, 0] / 0.6774 * 10 ** 10  # gas mass in cluster
halo_gal_count = halos['GroupNsubs'][:]

# applying filter to halo positions for 10^14, luminosity, mass, gal count, and flag masks applied.
# Creating a mask to choose magnitude range for r-band to match SDSS data. Refer to Toni's paper.
# ______SubHalos_____________
# subhalomass = subhalos['SubhaloMassType']
Flag_mask = subhalos['SubhaloFlag']  # array with value 1 or 0. 1 meaning, it's an actual galaxy, 0 values should be excluded.
subhalo_mass = subhalos['SubhaloMass']  # galaxy mass, includes all particle types in simulation
subhalo_coord = subhalos['SubhaloPos'] / (0.6774 * 10 ** 3)  # subhalo (galaxy) position
subhalo_mass = subhalo_mass / 0.6774 * 10 ** 10  # subhalo_mass in solar masses
subhalo_group_index = subhalos['SubhaloGrNr']

mask = np.ones(len(subhalo_mag_r_dust), dtype=np.bool)
Lum_mask = mask * (subhalo_mag_r_dust < -18.014)  # -18.4 for EAGLE...-18.014 for TNG300, according to Daniela. Provides the same number density as EAGLE and SDSS.
subhalo_group_index = list(set(subhalo_group_index[Lum_mask * Flag_mask]))

##### Halo Masks ########
mask_halo = np.ones(len(halo_gal_count), dtype=np.bool)
halo_gal_count_mask = mask_halo * (halo_gal_count > 0)


print('length of halo cord array pre mask:', len(halo_coord_x), len(halo_coord_y), len(halo_coord_z))

halo_coord_x = halo_coord_x[halo_gal_count_mask ] #* galaxy_mass_mask]
halo_coord_y = halo_coord_y[halo_gal_count_mask ] #* galaxy_mass_mask] # Chage to be based on the subhalo group index
halo_coord_z = halo_coord_z[halo_gal_count_mask ] #* galaxy_mass_mask]

print('Length of halo cord array x,y,z after masks', len(halo_coord_x), len(halo_coord_y), len(halo_coord_z))

# Applying the mask after filtering to maintain the length
subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask * Flag_mask]
print('The length of subhalo_mag_r post kd and masks (lum + flag)', len(subhalo_mag_r_dust))
mag_sun_r = 4.42  # absolute magnitude of sun in r-band.

lum_r = 10.0 ** ((subhalo_mag_r_dust - mag_sun_r) / -2.5)
lum_avg_den = sum(lum_r.flatten()) / 303. ** 3
print('avg lum den',lum_avg_den)
###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600#600  # 1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303.  # Units of Mpc
cell_len = box_len / num_cells
print('cell length',cell_len)
LD_Overden = np.zeros([num_cells, num_cells, num_cells])
z = subhalo_coord[Lum_mask * Flag_mask, 2]  # / (0.6774*10**3)
y = subhalo_coord[Lum_mask * Flag_mask, 1]  # / (0.6774*10**3)
x = subhalo_coord[Lum_mask * Flag_mask, 0]  # / (0.6774*10**3)

del subhalo_coord

x_bins = np.linspace(0.0, 303.0, num_cells + 1)
y_bins = np.linspace(0.0, 303.0, num_cells + 1)
z_bins = np.linspace(0.0, 303.0, num_cells + 1)

x_idx = np.digitize(x, x_bins)
y_idx = np.digitize(y, y_bins)
z_idx = np.digitize(z, z_bins)

lum_xyz = np.array([x,y,z]).T

#@njit
#def Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden):
#    #LD_smooth_param = 1.2 / cell_len  # smoothing 1.2 Mpc converted to Mpc/cell_len (dimensionless), Units of Mpc
#    for n in range(len(x)):
#        LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + lum_r[n] / lum_avg_den / (box_len / num_cells) ** 3  # matching luminosity values with binned coordinates
#    return LD_Overden
#Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden)  # running LD function

LD_Overden, lum_edges = np.histogramdd(lum_xyz, bins = (x_bins, y_bins, z_bins), weights = lum_r / lum_avg_den / (box_len / num_cells) ** 3)

print("number of galaxies for smoothing:", np.size(x))
print("dimensions of LD grid", np.shape(LD_Overden))
print("number of bins with galaxies (lum values):", np.size(np.nonzero(LD_Overden.flatten())))

########### smoothing using Gaussian kernel ####################################
smooth_param = 3 / (box_len/num_cells) # 2 #Units of Mpc
LD_Overden = gf(LD_Overden, sigma=0.6 * smooth_param)

#plt.imshow(LD_Overden[0:2,:,:].sum(0).T, origin = 'lower', extent = (0,303, 0, 303), norm = LogNorm())
#plt.ylabel('Y [Mpc]')
#plt.xlabel('X [Mpc]')
#plt.title('$\delta_{LD}$')
#plt.colorbar()
#plt.show()
np.save('ld_overden_3smoothing.npy', LD_Overden)