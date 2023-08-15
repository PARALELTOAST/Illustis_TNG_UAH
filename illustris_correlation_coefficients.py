import numpy as np
#import numba as nb
from numba import njit
#import pandas as pd
from matplotlib import pyplot as plt
#import scipy
from scipy.ndimage import gaussian_filter as gf
from scipy.stats import linregress
# import math as mth
# import matplotlib as mpl
# from matplotlib.colors import LogNorm
# from matplotlib import ticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib as mpl
# #import random
# from functools import partial
# from matplotlib.cm import register_cmap, cmap_d
import h5py
#import argparse #add options and arguments when running script in command line
#import sys
import illustris_python as il
#import filament_grid_construction as fgc
#import illustris_LD_WHIM_Grids as Grids
#import datashader as ds
#from datashader.mpl_ext import dsshow

################ Reading in zero redshift galaxies with dust correction #####################
dust_corrected_file = '/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5'
with h5py.File(dust_corrected_file, "r") as partData:
    subhalo_mag_r_dust = np.asarray(partData["/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:,2,0],dtype='f4')

#min_mag_r = min(subhalo_mag_r_dust)
#max_mag_r = max(subhalo_mag_r_dust)
#print("min and max of mag_r, dust corrected:",min_mag_r, max_mag_r)

################# Loading in subhalo and halo fields ##########################################
path = '/home/benjamin/Thesis/Illustris_TNG/tng_300'
halo_fields = ['GroupPos', 'GroupMass', 'Group_R_Crit200', 'GroupNsubs', 'Group_R_Crit500'] #check how much of gas lies within R200, R500. See ratio of gas mass to total mass. Compare to EAGLE Toni's paper. 'GroupMassType',Group_M_Crit500', 'Group_M_Crit200', 'Group_R_Crit500', 'GroupLenType', 'GroupLen'
subhalo_fields = ['SubhaloFlag','SubhaloMass','SubhaloMassType', 'SubhaloPos', 'SubhaloStellarPhotometrics', 'SubhaloGrNr']
halos = il.groupcat.loadHalos(path,99,fields=halo_fields)
subhalos = il.groupcat.loadSubhalos(path,99,fields=subhalo_fields)

#_____Halos________
halo_coord_x = halos['GroupPos'][:,2] / (0.6774 * 10**3)
halo_coord_y = halos['GroupPos'][:,1] / (0.6774 * 10**3)
halo_coord_z = halos['GroupPos'][:,0] / (0.6774 * 10**3)
halo_rad_crit200 = halos['Group_R_Crit200'][:] / (0.6774*10**3)
halo_rad_crit500 = halos['Group_R_Crit500'][:] / (0.6774*10**3)
halo_mass = halos['GroupMass'] / 0.6774 * 10**10 #cluster mass, includes all particle types
#halo_mass_gas = halos['GroupMassType'][:,0] /0.6774 *10**10#gas mass in cluster
halo_gal_count = halos['GroupNsubs'][:]
print('number of cluster positions and number of r200 radii',len(halo_coord_x), len(halo_rad_crit200))
print('checking the total number of subhalos in all halos, no filtering of halo or subhalos:',sum(halo_gal_count))
print('radii range for clusters',min(halo_rad_crit200), max(halo_rad_crit200))
print('halo(cluster) mass range:',min(halo_mass),max(halo_mass))

#______SubHalos_____________
#subhalomass = subhalos['SubhaloMassType']
Flag_mask = subhalos['SubhaloFlag'] #array with value 1 or 0. 1 meaning, it's an actual galaxy, 0 values should be excluded.
subhalo_mass = subhalos['SubhaloMass'] #galaxy mass, includes all particle types in simulation
#subhalo_mass_stellar = subhalos['SubhaloMassType'][:,4] #galaxy mass, only stellar particles.
subhalo_coord = subhalos['SubhaloPos'] #subhalo (galaxy) position
#subhalo_mag_r = subhalos['SubhaloStellarPhotometrics'][:,5] #galaxy magnitude r-band. Convert to luminosity
#print("min and max of mag_r, no filter:",min(subhalo_mag_r), max(subhalo_mag_r))
subhalo_mass = subhalo_mass /0.6774 *10**10 #subhalo_mass in solar masses
#subhalo_mass_stellar = subhalo_mass_stellar /0.6774 *10**10 #subhalo stellar mass in solar masses
subhalo_group_index = subhalos['SubhaloGrNr']

#Creating a mask to choose magnitude range for r-band to match SDSS data. Refer to Toni's paper.
mask = np.ones(len(subhalo_mag_r_dust),dtype=np.bool)
Lum_mask = mask*(subhalo_mag_r_dust < -18.014) #-18.4 for EAGLE...-18.014 for TNG300, according to Daniela. Provides the same number density as EAGLE and SDSS.

####### masks for groups/halos (clusters) ##############################
mask_halo = np.ones(len(halo_gal_count),dtype=np.bool)
halo_gal_count_mask = mask_halo*(halo_gal_count > 0)
halo_mass_mask = mask_halo*(halo_mass > 10**14)

x_halo = halo_coord_x[halo_gal_count_mask * halo_mass_mask]
y_halo = halo_coord_y[halo_gal_count_mask * halo_mass_mask]
z_halo = halo_coord_z[halo_gal_count_mask * halo_mass_mask]
halo_rad_mass_filtered = halo_rad_crit200[halo_gal_count_mask * halo_mass_mask]
#halo_rad_mass_filtered = halo_rad_crit500[halo_gal_count_mask * halo_mass_mask]

plt.hist(halo_rad_mass_filtered, bins=np.linspace(min(halo_rad_mass_filtered),max(halo_rad_mass_filtered), 40), density=False)
plt.ylabel('counts')
plt.xlabel('radius')
plt.savefig('Halo_mass_cut_radii_distribution.pdf')
plt.close()
##################################### Recording data ####################################
print('min and max of counts in halos:', min(halo_gal_count[halo_gal_count_mask*halo_mass_mask]), max(halo_gal_count[halo_gal_count_mask*halo_mass_mask]))
print('number of cluster positions after applying subhalo count filter and mass filter',len(halo_coord_x[halo_gal_count_mask*halo_mass_mask]))
print('halo(cluster) mass range after applying subhalo count filter and mass filter:',min(halo_mass[halo_gal_count*halo_mass_mask]),max(halo_mass[halo_gal_count*halo_mass_mask]))
#galaxy_mass_mask = mask*(subhalo_mass_stellar > 0.0) #(subhalo_mass_stellar <= 10**12)*(subhalo_mass_stellar >= 10**9) #mask for the galaxy stellar masses. Some seem unusual
galaxy_lum_r = 10.0**((subhalo_mag_r_dust - 4.42)/-2.5)
#lum_massflag_only = galaxy_lum_r[Flag_mask*galaxy_mass_mask]
galaxy_lum_r = galaxy_lum_r[Flag_mask*Lum_mask]

lum_avg_den = sum(galaxy_lum_r.flatten()) / 303.**3
del galaxy_lum_r

print("Average luminosity density of whole volume, with only flagged bad galaxies excluded and lum filter applied:",lum_avg_den)
#print("min and max of galaxy masses, no filtering:",min(subhalo_mass),max(subhalo_mass))
#print("min and max of galaxy masses, only flag and lum filtering:",min(subhalo_mass[Lum_mask*Flag_mask]),max(subhalo_mass[Lum_mask*Flag_mask]))
#print("min and max of galaxy stellar masses, only lum and flag filtering:", min(subhalo_mass_stellar[Lum_mask*Flag_mask]), max(subhalo_mass_stellar[Lum_mask*Flag_mask]))
#print("min and max of galaxy stellar masses,filtering:", min(subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]), max(subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]))
print("number of subhaloes, no filtering at all:",len(subhalo_mag_r_dust))
print("number of flagged subhaloes:",len(subhalo_mag_r_dust) - len(subhalo_mag_r_dust[Flag_mask]))
#print("number of galaxies only mass filtering:",np.shape(lum_massflag_only))
print("number of galaxies with only lumfiltering and flagged filter:",len(subhalo_mag_r_dust[Flag_mask*Lum_mask]))
subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask*Flag_mask]
#subhalo_mass = subhalo_mass[Lum_mask*Flag_mask*galaxy_mass_mask]
#subhalo_mass_stellar = subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]
subhalo_group_index = list(set(subhalo_group_index[Lum_mask*Flag_mask]))
############   Removing virialized WHIM gas within r200 from left-over halos (Halos that the galaxies belong   ##################
halo_coord_x = halo_coord_x[subhalo_group_index]
halo_coord_y = halo_coord_y[subhalo_group_index]
halo_coord_z = halo_coord_z[subhalo_group_index]
halo_rad_crit200 = halo_rad_crit200[subhalo_group_index]

plt.hist(halo_rad_crit200, bins=np.linspace(min(halo_rad_mass_filtered),max(halo_rad_mass_filtered), 50), density=False)
plt.ylabel('counts')
plt.xlabel('radius')
plt.savefig('Halo_radii_distribution.pdf')
plt.close()
print('radii range for halos after galaxy filtering:',min(halo_rad_crit200),max(halo_rad_crit200))
print('Number of halos left after subhalo filtering:',len(halo_rad_crit200))
#np.savetxt("subhalo_mass_total.txt",subhalo_mass,fmt="%s")
#np.savetxt("subhalo_stellar_mass.txt",subhalo_mass_stellar,fmt="%s")
#np.savetxt("subhalo_r_magnitude_dust.txt",subhalo_mag_r_dust,fmt="%s")

#subhalo_mag_r = subhalo_mag_r[Flag_mask] checking number of flagged galaxies
#subhalo_mass = subhalo_mass[Flag_mask] 
#subhalo_mag_r = subhalo_mag_r[Lum_mask*Flag_mask]
#subhalo_mass = subhalo_mass[Lum_mask*Flag_mask]
print("number of galaxies, filtered:",len(subhalo_mag_r_dust))
#quit()

##### checking max and min of r-band magnitude Also converting magnitude to luminosity ############
mag_sun_r = 4.42 #absolute magnitude of sun in r-band.
#print("min and max of magnitude, filtered:",min(subhalo_mag_r_dust), max(subhalo_mag_r_dust))
#print("min and max of masses, filtered:",min(subhalo_mass), max(subhalo_mass))
lum_r = 10.0**((subhalo_mag_r_dust - mag_sun_r)/-2.5)
lum_avg_den = sum(lum_r.flatten()) / 303.**3
print("average luminosity density of full volume, filtered:",lum_avg_den)
#print("min and max of luminosity, filtered:",min(lum_r),max(lum_r))
#histogram plot of masses after filtering
#gal_mass = np.log10(subhalo_mass_stellar)
#plt.hist(gal_mass,bins=np.linspace(min(gal_mass),max(gal_mass),50), histtype='bar',alpha=0.5, density=False)
#plt.xlabel(r'$\log(M*)$',fontsize=12)
#plt.savefig("subhalo_mass_histogram_dust.pdf")
#plt.close()
#quit()

###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600 #1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303. #Units of Mpc
LD_Overden = np.zeros([num_cells,num_cells,num_cells])
x = subhalo_coord[Lum_mask*Flag_mask,2] / (0.6774*10**3)
y = subhalo_coord[Lum_mask*Flag_mask,1] / (0.6774*10**3)
z = subhalo_coord[Lum_mask*Flag_mask,0] / (0.6774*10**3)

del subhalo_coord

np.savetxt("subhalo_positions.txt", np.transpose([x,y,z]),fmt="%s")


x_bins = np.linspace(0.0,303.0,num_cells + 1)
y_bins = np.linspace(0.0,303.0,num_cells + 1)
z_bins = np.linspace(0.0,303.0,num_cells + 1)


x_idx = np.digitize(x,x_bins)
y_idx = np.digitize(y,y_bins)
z_idx = np.digitize(z,z_bins)


@njit
def Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden):
    #LD_smooth_param = 1.2/cell_len #smoothing 1.2 Mpc converted to Mpc/cell_len (dimensionless), Units of Mpc
    for n in range(len(x)):
        LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + lum_r[n]/lum_avg_den/(box_len/num_cells)**3 #matching luminosity values with binned coordinates
    return LD_Overden 
#Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden) #running LD function
LD_Overden, LD_edges = np.histogramdd(np.array([x,y,z]).T,bins=[x_bins,y_bins,z_bins],density=False,weights=lum_r/lum_avg_den/(box_len/num_cells)**3)

#print("number of galaxies for smoothing:",np.size(x))
#print("dimensions of LD grid",np.shape(LD_Overden))
#print("number of bins with galaxies (lum values):",np.size(np.nonzero(LD_Overden.flatten())))
#################### Linear correlation coefficeint value arrays ###################

slope = np.zeros(30)
intercept = np.zeros(30)
corr_coeff = np.zeros(30)
pval = np.zeros(30)
std_err = np.zeros(30)

########### smoothing using Gaussian kernel ####################################
smooth_param = np.linspace(0.1,3.0,30) #units of Mpc
smooth_val = np.zeros(30)
for k in range(len(smooth_val)):
    smooth_val[k] = smooth_param[k]/(box_len/num_cells) #Units of Mpc to Mpc/cell_len (dimensionless)
    LD_Overden = gf(LD_Overden, sigma=0.6*smooth_val[k])

    #print('max and min lum overden',max(LD_Overden.flatten()),min(LD_Overden.flatten()))
    #quit()
########### Smoothing using B3 Spline #########################################
#### maybe use scipy.interpolate.CubicSpline for each coordinate x,y,z ########

####################### Reading in WHIM dataset from Daniela ###################

    # WHIM_Data = pd.read_csv('dataset_WHIMgas__tng3001_z0__ratio400.csv')
    # WHIM_Data = np.asarray(WHIM_Data)
    #
    # z_pos_WHIM = np.asarray(WHIM_Data[:,1]) #x coordinate for WHIM in Mpc
    # y_pos_WHIM = np.asarray(WHIM_Data[:,2]) #y ...
    # x_pos_WHIM = np.asarray(WHIM_Data[:,3]) #z ...
    # mass_WHIM = np.asarray(WHIM_Data[:,4]) #units in M_sun
    # den_WHIM = np.asarray(WHIM_Data[:,5]) #units in M_sun * Mpc^-3
    # temp_WHIM = np.asarray(WHIM_Data[:,8]) #units in Kelvin (K)
    #press_WHIM = WHIM_Data[:,9] #pressure in units of KeV*cm^-3           
    #volume = mass_WHIM / den_WHIM

##########t plotting distribution of WHIM ##################
    #quit()
    #mask_WHIM = np.ones(len(z_pos_WHIM),dtype=np.bool)
    #WHIM_denpart_mask = mask_WHIM*(den_WHIM > 10**12)
    #den_WHIM = den_WHIM[WHIM_denpart_mask]
    #z_pos_WHIM = z_pos_WHIM[WHIM_denpart_mask]
    #y_pos_WHIM = y_pos_WHIM[WHIM_denpart_mask]
    #x_pos_WHIM = x_pos_WHIM[WHIM_denpart_mask]
    
    '''
    x_idx = np.digitize(x_pos_WHIM,x_bins)
    y_idx = np.digitize(y_pos_WHIM,y_bins)
    z_idx = np.digitize(z_pos_WHIM,z_bins)

    halo_x_idx = np.digitize(halo_coord_x,x_bins)
    halo_y_idx = np.digitize(halo_coord_y,y_bins)
    halo_z_idx = np.digitize(halo_coord_z,z_bins)

    halo_pos_grid = np.zeros([num_cells, num_cells, num_cells])
    WHIM_pos_grid = np.zeros([num_cells, num_cells, num_cells])
    '''

    # @njit
    # def Halo_pos_func(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, halo_coord_x, halo_coord_y, halo_coord_z, halo_rad_crit200, halo_pos_grid):
    #     for j in range(len(halo_coord_x)):
    #         halo_pos_grid[halo_z_idx[i] - 1, halo_y_idx[i] - 1, halo_x_idx[i] - 1] = 1
    #     return halo_pos_grid
    #
    # @njit
    # def WHIM_gas_pos_func(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, halo_coord_x, halo_coord_y, halo_coord_z, halo_rad_crit200, WHIM_pos_grid):
    #     for i in range(len(x_pos_WHIM)):
    #         WHIM_pos_grid[z_idx[i] - 1, y_idx[i] - 1, x_idx[i] - 1] = 1
    #     return WHIM_pos_grid

    #Halo_pos_func(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, halo_coord_x, halo_coord_y, halo_coord_z, halo_rad_crit200, halo_pos_grid)
    #WHIM_gas_pos_func(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, halo_coord_x, halo_coord_y, halo_coord_z, halo_rad_crit200, WHIM_pos_grid)
    #
    # @njit
    # def WHIM_gas_removal(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, halo_coord_x, halo_coord_y, halo_coord_z, halo_rad_crit200, halo_pos_grid, WHIM_pos_grid):
    #     for j in range(len(halo_coord_x)):
    #         k_max = int(halo_rad_crit200[j] / (box_len/num_cells))
    #         #r_idx = np.transpose([x_idx, y_idx, z_idx])
    #         halo_r_idx = np.transpose([halo_x_idx, halo_y_idx, halo_z_idx])
    #         r = np.transpose([x_pos_WHIM, y_pos_WHIM, z_pos_WHIM])
    #         mask = np.ones(len(z_idx),dtype=np.bool)
    #         #r_idx_new = mask*(r_idx == halo_r_idx[j]) #seems to have issues
    #         #r_idx_new = mask*(x_idx == halo_x_idx[j] and y_idx == halo_y_idx[j] and z_idx == halo_z_idx[j]) #issues here too
    #         r_idx_new = mask*(x_idx == halo_x_idx[j])*(y_idx == halo_y_idx[j])*(z_idx == halo_z_idx[j]) #seems fine
    #         r_new = r[r_idx_new]
    #         #r_idx_new = [m for m in r_idx if m==halo_r_idx[j]]
    #         #for n in range(len(r_idx)):
    #         #   if r_idx[n]==halo_r_idx[j]:
    #         #      r_new.append(r[n])
    #         #r_new = [m for m in r if for n in r_idx n==halo_r_idx[j]]
    #         #for k in range(k_max + 1):
    #             #l = -k
    #         for m in range(len(r_new)):
    #             dist = np.sqrt((r_new[m][0] - halo_coord_x[j])**2 + (r_new[m][1] - halo_coord_y[j])**2 + (r_new[m][2] - halo_coord_z[j])**2)
    #             if dist < halo_rad_crit200[j]:
    #                 mass_WHIM[m]=0.0
    #     return mass_WHIM

    '''
    x_halo_idx = np.digitize(x_halo,x_bins)
    y_halo_idx = np.digitize(y_halo,y_bins)
    z_halo_idx = np.digitize(z_halo,z_bins)
    '''

    # @njit
    # def WHIM_gas_removal_inside_Clusters_new_method(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, x_halo, y_halo , z_halo, halo_rad_mass_filtered):
    #     for j in range(len(x_halo)):
    #         k_max = 3 #int(halo_rad_mass_filtered[j] / (box_len/num_cells))
    #         r = np.transpose([x_pos_WHIM, y_pos_WHIM, z_pos_WHIM])
    #         mask = np.ones(len(z_idx),dtype=np.bool)
    #         r_halo = np.transpose([x_halo, y_halo, z_halo])
    #         #r_idx_1 = mask * (x_idx == x_halo_idx[j])*(y_idx == y_halo_idx[j])*(z_idx == z_halo_idx[j])
    #         for k in range(-1*k_max, k_max + 1):
    #             for l in range(-1*k_max, k_max + 1):
    #                 for m in range(-1*k_max, k_max + 1):
    #                     if x_halo_idx[j] + k < 1 or x_halo_idx[j] + k > num_cells or y_halo_idx[j] + l < 1 or y_halo_idx[j] + l > num_cells or z_halo_idx[j] + m < 1 or z_halo_idx[j] + m > num_cells:
    #                         continue
    #                     r_idx_1 = mask * (x_idx == x_halo_idx[j] + k)*(y_idx == y_halo_idx[j] + l)*(z_halo_idx[j] + m)
    #                     #r_idx_2 = mask * (x_idx == x_halo_idx[j] - k)*(y_idx == y_halo_idx[j] - l)*(z_halo_idx[j] - m)
    #                     r_new_1 = r[r_idx_1]
    #                     #r_new_2 = r[r_idx_2]
    #                     dist_1 = (np.sum((r_new_1 - r_halo[j])**2,axis=1))**0.5
    #                     #dist_2 = (np.sum((r_new_2 - r_halo[j])**2,axis=1))**0.5
    #                     #for i in range(len(r_new)):
    #                         #dist = np.sqrt((r_new[i][0] - x_halo[j])**2 + (r_new[i][1] - y_halo[j])**2 + (r_new[i][2] - z_halo[j])**2)
    #                     for n in range(len(dist_1)):
    #                         if dist_1 < halo_rad_mass_filtered[j]:
    #                 #x_pos_WHIM[i]=0.0
    #                 #y_pos_WHIM[i]=0.0
    #                 #z_pos_WHIM[i]=0.0 #delete positions with zeros after finishing
    #                             mass_WHIM[i]=0.0
    #                 #temp_WHIM[i]=0.0
    #                 #x_pos_WHIM = x_pos_WHIM[np.nonzero(x_pos_WHIM)]
    #                 #y_pos_WHIM = y_pos_WHIM[np.nonzero(y_pos_WHIM)]
    #                 #z_pos_WHIM = z_pos_WHIM[np.nonzero(z_pos_WHIM)]
    #                 #mass_WHIM = mass_WHIM[np.nonzero(mass_WHIM)]
    #                 #temp_WHIM = temp_WHIM[np.nonzero(temp_WHIM)]
    #     return mass_WHIM
    #
    # #WHIM_mask = np.ones(len(mass_WHIM),dtype=np.bool)
    # @njit
    # def WHIM_gas_removal_inside_Clusters(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, x_halo, y_halo , z_halo, halo_rad_mass_filtered, WHIM_mask):
    #     for j in range(len(x_halo)):
    #         for i in range(len(WHIM_mask)):
    #             dist = np.sqrt((x_pos_WHIM[i] - x_halo[j])**2 + (y_pos_WHIM[i] - y_halo[j])**2 + (z_pos_WHIM[i] - z_halo[j])**2)
    #             if dist < halo_rad_mass_filtered[j]:
    #                 #x_pos_WHIM[i]=0.0
    #                 #y_pos_WHIM[i]=0.0
    #                 #z_pos_WHIM[i]=0.0 #delete positions with zeros after finishing
    #                 WHIM_mask[i] = False
    #                 #mass_WHIM[i] = 0.0
    #                 #temp_WHIM[i]=0.0
    #                 #x_pos_WHIM = x_pos_WHIM[np.nonzero(x_pos_WHIM)]
    #                 #y_pos_WHIM = y_pos_WHIM[np.nonzero(y_pos_WHIM)]
    #                 #z_pos_WHIM = z_pos_WHIM[np.nonzero(z_pos_WHIM)]
    #                 #mass_WHIM = mass_WHIM[np.nonzero(mass_WHIM)]
    #                 #temp_WHIM = temp_WHIM[np.nonzero(temp_WHIM)]
    #     return WHIM_mask #mass_WHIM

    #WHIM_gas_removal_inside_Clusters(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, x_halo, y_halo, z_halo, halo_rad_mass_filtered, WHIM_mask) 
    #WHIM_gas_removal_inside_Clusters(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, halo_coord_x, halo_coord_y, halo_coord_z, halo_rad_crit200, WHIM_mask)
    #WHIM_gas_removal_inside_Clusters_new_method(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, mass_WHIM, x_halo, y_halo , z_halo, halo_rad_mass_filtered)

    #num_WHIM_original = len(mass_WHIM)
    #mass_WHIM = mass_WHIM[~WHIM_mask] # tilde symbol is negation of WHIM_mask
    ##########################      Using Ben's KD Tree to Filter gas out of halos    ######################################################################

    # print("loading tree...")
    # halos = [(x, y, z) for x, y, z in zip(halo_coord_x, halo_coord_y, halo_coord_z)]
    #
    # # scipy.spatial.KDTree.query_ball_point(halos, halo_rs, p=2.0, eps=0, workers=1, return_sorted=None, return_length=False)
    # points = [(x, y, z) for x, y, z in zip(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM)]
    # tree = scipy.spatial.KDTree(points)
    # print("tree loaded")

    ############################## The filtering for interior to r200##########################################

    # total = 0
    # whim_in_halo_index = []
    # for i in range(len(halos)):
    #     points_in_radius = tree.query_ball_point(halos[i], halo_rad_crit200[i], workers=-1)  # setting workers to -1 maximizes parallelization
    #     total += len(points_in_radius)  # to keep track of the total number of points that got zeroed out (although this could double count, but there are no dual halo inhabitants for r200)
    #     for point in points_in_radius:
    #         den_WHIM[point] = 0
    #         mass_WHIM[point] = 0
    #         whim_in_halo_index.append([point, i])
    #     points_in_radius = np.array(points_in_radius)
    #     for j in range(len(points_in_radius)):
    #         whim_in_halo_index.append([points_in_radius[j], i])
    #
    # print('total number of WHIM excluded:',total)

    '''
    temp_WHIM = temp_WHIM[~WHIM_mask] #[np.nonzero(mass_WHIM)]
    x_pos_WHIM = x_pos_WHIM[~WHIM_mask] #[np.nonzero(mass_WHIM)]
    y_pos_WHIM = y_pos_WHIM[~WHIM_mask] #[np.nonzero(mass_WHIM)]
    z_pos_WHIM = z_pos_WHIM[~WHIM_mask] #[np.nonzero(mass_WHIM)]
    mass_WHIM = mass_WHIM[~WHIM_mask] #[np.nonzero(mass_WHIM)]

    plt.hist(temp_WHIM,np.linspace(min(temp_WHIM),max(temp_WHIM),50),density=False)
    plt.xlabel('temp')
    plt.ylabel('count')
    plt.savefig('WHIM_temp_dist_within_Clusters.pdf')
    plt.close()
    '''
    #del WHIM_mask 
    # print('radius range of halos above 10^14 solar masses:',min(halo_rad_mass_filtered),max(halo_rad_mass_filtered))
    # print('number of WHIM inside halos r200.', len(mass_WHIM)) #np.count_nonzero(mass_WHIM))
    #quit()            

    '''
    x_idx = np.digitize(x_pos_WHIM,x_bins)
    y_idx = np.digitize(y_pos_WHIM,y_bins)
    z_idx = np.digitize(z_pos_WHIM,z_bins)
    '''

    mean_baryon_den = 0.618 * 10**10 #mean baryon density of universe in units of M_sun * Mpc^-3
    # WHIM_Den_avg = sum(mass_WHIM) * 400. / 303.**3 #WHIM gas sampled selecting 1 out of 400.
    #
    # #WHIM_Overden = np.zeros([num_cells,num_cells,num_cells])
    # #Temp_grid = np.zeros([num_cells,num_cells,num_cells]) #mass weighted mean temperature of each cell
    # #Mass_grid = np.zeros([num_cells,num_cells,num_cells])
    # #Volume_grid = np.zeros([num_cells,num_cells,num_cells])
    # #Pressure_grid = np.zeros([num_cells, num_cells, num_cells])
    #
    # @njit
    # def WHIM_mass_func(box_len, num_cells, x, y, z, Mass_grid):
    #     #cell_len = box_len/num_cells
    #     for n in range(len(x)):
    #         Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]
    #     return Mass_grid
    #
    # #WHIM_mass_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Mass_grid)
    #
    # #@njit
    # #def WHIM_Pressure_func(box_len, num_cells, x, y, z, Pressure_grid):
    #     #cell_len = box_len/num_cells
    #     #for n in range(len(x)):
    #         #Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*press_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
    #     #return Pressure_grid #Mass weighted mean pressure for each cell
    #
    # #WHIM_Pressure_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Pressure_grid)
    #
    # @njit
    # def WHIM_Temp_func(box_len, num_cells, x, y, z, Temp_grid):
    #     #cell_len = box_len/num_cells
    #     for n in range(len(x)):
    #         Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*temp_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
    #     return Temp_grid #Mass weighted mean temperature for each cell
    #
    # @njit
    # def WHIM_OverDen_func(box_len, num_cells, x, y, z, WHIM_Overden):
    #     #cell_len = box_len/num_cells
    #     for n in range(len(x)):
    #         WHIM_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] / (box_len/num_cells)**3 * 400. / mean_baryon_den #Volume_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] /mean_baryon_den  #matching luminosity values with binned coordinates
    #     return WHIM_Overden
    
    #WHIM_OverDen_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, WHIM_Overden)
    #WHIM_Temp_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Temp_grid)
    
    '''
    del Mass_grid
    del WHIM_Data 
    del z_pos_WHIM 
    del y_pos_WHIM
    del x_pos_WHIM
    del mass_WHIM
    del den_WHIM
    del temp_WHIM
    del x_idx
    del y_idx
    del z_idx

    del halo_x_idx
    del halo_y_idx
    del halo_z_idx

    del halo_pos_grid
    del WHIM_pos_grid
    '''
    
    ################ Loading in density grid constructed by Daniela ##########################
    WHIM_den_file = 'GRID_WHIM_DENSITY_NOMASKhaloes.npy'
    WHIM_Overden = np.load(WHIM_den_file) #loading new WHIM densiyt file from Daniela
    WHIM_Overden = np.transpose(WHIM_Overden, (2,1,0))/mean_baryon_den
    print('WHIM density range for cells:',min(WHIM_Overden.flatten()),max(WHIM_Overden.flatten()))
    print("number of bins with WHIM (den values):",np.size(np.nonzero(WHIM_Overden.flatten())))
############## Plotting ##########################################
###############  Filaments #######################################
    #
    # filefils = 'filament_coords_and_lengths.npy'
    #
    # # Read filaments
    # seg_z, seg_y, seg_x, lengths = np.load(filefils, allow_pickle=True) #Note, each seg[i] has different lengths.
    # #fil_grid = fgc.fil_grid #calling filament_grid_construction.py file to obtain fil_grid
    # #quit()
    fil_grid = np.load('fil_gri.npy', allow_pickle=True)
    fil_grid = np.bool_(fil_grid)
############ Selecting WHIM and LD within filaments ##################################################
    #WHIM_scat = WHIM_Overden[fil_grid].flatten() #WHIM density in filaments
    #LD_scat = LD_Overden[fil_grid].flatten() #LD in filaments
    #Temp_scat = Temp_grid[fil_grid].flatten()
    WHIM_scat = WHIM_Overden[fil_grid].flatten() #WHIM den outside of filaments
    LD_scat = LD_Overden[fil_grid].flatten() #LD outside of filaments
    #Temp_scat = np.log10(Temp_scat)

    '''
    plt.hist(Temp_scat,bins=np.linspace(min(Temp_scat),max(Temp_scat),50),density=False)
    plt.xlabel(r'$\log T [K]$')
    plt.ylabel('count')
    plt.savefig('Temp_grid_distribution_filrad_05.pdf')
    plt.show()
    plt.close()
    
    plt.hist(np.log10(WHIM_scat[np.nonzero(WHIM_scat)]),bins=np.linspace(min(np.log10(WHIM_scat[np.nonzero(WHIM_scat)])),max(np.log10(WHIM_scat[np.nonzero(WHIM_scat)])),50),density=False)
    plt.xlabel(r'$\log \delta$')
    plt.ylabel('count')
    plt.savefig('WHIMOverden_grid_distribution_filrad_05.pdf')
    plt.show()
    plt.close()
    '''

    #Temp_scat = Temp_scat[np.nonzero(WHIM_scat)]
    LD_scat = LD_scat[np.nonzero(WHIM_scat)]
    WHIM_scat = WHIM_scat[np.nonzero(WHIM_scat)]
    #Temp_scat = Temp_scat[np.nonzero(LD_scat)]
    WHIM_scat = WHIM_scat[np.nonzero(LD_scat)]
    LD_scat = LD_scat[np.nonzero(LD_scat)]
    WHIM_scat = np.log10(WHIM_scat)
    LD_scat = np.log10(LD_scat)
    #Temp_scat = np.log10(Temp_scat)

    #print("average temperature (log) inside filaments with given filament radius",np.mean(Temp_scat))
    #print("temperature range of filaments with given filament radius",min(Temp_scat),max(Temp_scat))
    # print('length of WHIM scat and LD scat',len(WHIM_scat), len(LD_scat))
    #WHIM_Overden = WHIM_Overden*fil_grid
    #LD_Overden = LD_Overden*fil_grid

############# linear correlation coefficient test ###########################################
    slope[k], intercept[k], corr_coeff[k], pval[k], std_err[k] = linregress(LD_scat, WHIM_scat)

plt.plot(smooth_param, corr_coeff)
plt.xlabel('Smoothing Parameter [Mpc]')
plt.ylabel('Correlation Coefficient')
#plt.savefig('filament_corr_plots/correlation_smoothing_plot_filrad_30.pdf')
plt.grid()
plt.axvline(x=1.2, linestyle = '--', color = 'b')
plt.show()
quit()
# ##########################  Plotting Scatter of LD vs. WHIM density ###########################
#
# fig, ax=plt.subplots(1, figsize = (8,8))
#
# #dyn = partial(ds.tf.dynspread, max_px=40, threshold=0.5)
# #def datshader(ax,y,z):
# #    df=pd.DataFrame(dict(x=x, y=y))
# #    da1=dsshow(df,ds.Point('x','y'),spread_fn=dyn, aspect='auto',ax=ax)
# #    plt.colorbar(da1)
#
# nbins = 40
# #bins=[np.logspace(min(LD_scat),max(LD_scat),nbins+1), np.logspace(min(WHIM_scat),max(WHIM_scat),nbins+1)]
# bins=[np.linspace(min(LD_scat),max(LD_scat),nbins+1), np.linspace(min(WHIM_scat),max(WHIM_scat),nbins+1)]
# h, xedge, yedge = np.histogram2d(LD_scat, WHIM_scat, bins=bins)
# cmap = plt.get_cmap('Blues')
# vmin=min(h.flatten()) + 0.1
# vmax=max(h.flatten())
# print('vmin and vmax:',vmin,vmax)
# X, Y = np.meshgrid(xedge,yedge)
# im = plt.pcolormesh(X,Y,h,cmap=cmap,edgecolors='black',norm=LogNorm(vmin=vmin,vmax=vmax),linewidth=0.3)
# #im = plt.pcolormesh(X,Y,h,cmap=cmap, edgecolors='black',vmin=vmin, vmax=vmax,linewidth=0.3)
# #scat=datshader(ax,LD_scat,WHIM_scat)
# plt.legend(loc=1)
# plt.grid(False)
# plt.xlabel(r'LD [$\log \delta_{LD}$]',fontsize=15)
# plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]',fontsize=15)
# plt.savefig("Scatter_LD_WHIM_Illustris_smoothing_20.pdf")
# #plt.ion()
# plt.show()
# plt.close()
#
# ############################# Plotting slices of LD and WHIM ###################################
# print('Number of filaments and first value of seg_z:', np.shape(seg_z), seg_z[0])
# quit()
# for i in range(20):
#     z_in = 15*i #in Mpc
#     z_fin = 15*(i+1) #in Mpc
#     cmap_LD = mpl.cm.get_cmap('Oranges')
#     cmap_WHIM = mpl.cm.get_cmap('Purples')
#     fig, ax = plt.subplots(1, figsize = (8,8))
#
#     zslice = (z > z_in) * (z < z_fin)
#     zslice_halo = (z_halo > z_in) * (z_halo < z_fin)
#     zslice_WHIMden = (z_pos_WHIM > z_in) * (z_pos_WHIM < z_fin)
#     gridIndexMin = int((num_cells*z_in)/box_len)
#     gridIndexMax = int((num_cells*z_fin)/box_len)
#     imageSlice = np.mean(LD_Overden[gridIndexMin:gridIndexMax, :, :],axis=0)
#     #imageVolumeSlice=np.mean(Volume_grid[gridIndexMin:gridIndexMax,:,:],axis=0)
#     imageWHIMSlice=np.mean(WHIM_Overden[gridIndexMin:gridIndexMax,:,:],axis=0)
#     vmin = 1e-6 #min(imageSlice.flatten())
#     vmax = 1e2 #max(imageSlice.flatten())
#     vmin_WHIM = min(imageWHIMSlice.flatten())
#     vmax_WHIM = max(imageWHIMSlice.flatten())
#     vmin_nonzero = imageSlice.flatten()
#     vmin_nonzero = vmin_nonzero[np.nonzero(vmin_nonzero)]
#     print("min and max LD for image slice:",vmin, vmax)
#     print("min nonzero LD for image slice:",min(vmin_nonzero))
#     ####### plotting filaments. Filaments given by Daniela ##########
#     #for j in range(len(seg_x)):
#     #    if ( np.mean(seg_z[j]) > z_in ) & ( np.mean(seg_z[j]) <= z_fin ):
#     #        fil_scat = ax.plot(seg_x[j], seg_y[j], color='k')
#     #gal_scat = ax.scatter(x[zslice], y[zslice], marker='o', s=1.0, color='red', alpha=0.7, label='galaxies')
#     #cluster_scat = ax.scatter(x_halo[zslice_halo], y_halo[zslice_halo], marker='o', s=1.0, color='red', alpha=0.7, label='clusters')
#     #halo_scat = ax.scatter(halo_coord_x[zslice_halo], halo_coord_y[zslice_halo], marker='o', s=1.0, color='red', alpha=0.7, label='halos')
#     #WHIM_locations = ax.scatter(x_pos_WHIM[zslice_WHIMden], y_pos_WHIM[zslice_WHIMden], marker='o', s=1.0, color='red', alpha=0.7, label='WHIMpart')
#     LD_plot = ax.imshow(imageSlice, norm=LogNorm(vmin=1e-5, vmax=500.), extent=[0.0,303.0,0.0,303.0], aspect='auto', origin="lower", cmap=cmap_LD) #vmax=2000, vmin=1e-3, just like for EAGLE, only visual purposes
#     #Volume_plot = ax.imshow(imageWHIMSlice, cmap=cmap_WHIM, extent= [0.,303.,0.,303.0], origin='lower',alpha=0.6, label="Volume")
#     #WHIMden_plot = ax.imshow(imageWHIMSlice, cmap=cmap_WHIM, norm=LogNorm(vmin=vmin , vmax=vmax), extent= [0.,303.,0.,303.0], origin='lower',alpha=0.6, label="WHIM") #vmin = 1e-06, vmax = 1e05
#     plt.xlabel('X [Mpc]',fontsize=10)
#     plt.ylabel('Y [Mpc]',fontsize=10)
#     ax.legend(loc=1,fontsize=12)
#     divider = make_axes_locatable(ax)
#     cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
#     fig.add_axes(cax)
#     cbar = fig.colorbar(LD_plot, cax=cax, orientation = "horizontal", format=ticker.LogFormatter())
#     cbar.set_label(r'$\delta_{LD}$')#Luminosity Overdensity') #($10^{10} L_{\odot} Mpc^{-3}$)')
#     cbar.ax.tick_params(labelsize=10, width=0.7, size=8)
#     cbar.solids.set_edgecolor("face")
#     cbar.ax.xaxis.set_label_position('bottom')
#     ####### colorbar for WHIM #########################
#     #cax2 = divider.new_vertical(size='5%', pad=0.6)
#     #fig.add_axes(cax2)
#     #cbar2 = fig.colorbar(WHIMden_plot, cax=cax2, orientation = "horizontal") #, format=ticker.LogFormatter())
#     #cbar2.set_label(r'$\delta_{\rho}$')#WHIM Overdensity') #($10^{10} M_{\odot} Mpc^{-3}$)')
#     #cbar2.ax.tick_params(labelsize=10, width=0.7, size=8)
#     #cbar2.solids.set_edgecolor("face")
#     #cbar2.ax.xaxis.set_label_position('bottom')
#
#     #plt.savefig('WHIM_high_den_10to12_part_position_slice{i}.pdf'.format(i=i))
#     #plt.savefig('WHIM_within_clusters_slice{i}.pdf'.format(i=i))
#     #plt.savefig('WHIM_within_halos_slice{i}.pdf'.format(i=i))
#     #plt.savefig('halo_position_slice{i}.pdf'.format(i=i))
#     #plt.savefig('Volume_grid_slice{i}.pdf'.format(i=i))
#     #plt.savefig('WHIM_density_filaments_galaxies_slice{i}_300.pdf'.format(i=i))
#     #plt.savefig('WHIM_density_slice{i}.pdf'.format(i=i))
#     #plt.savefig('WHIM_density_slice{i}_300cells.pdf'.format(i=i))
#     #plt.savefig('WHIM_density_fil_filtered_with_fils_slice{i}.pdf'.format(i=i))
#     #plt.savefig('galaxy_positions_LD_slice{i}.pdf'.format(i=i))
#     #plt.savefig('galaxies_and_fils{i}.pdf'.format(i=i))
#     plt.savefig('Select_galaxy_positions_LD_slice{i}_300.pdf'.format(i=i))
#     #print('num of galaxies in slice:',np.size(x[zslice]))
#     plt.show()
#     plt.close()
