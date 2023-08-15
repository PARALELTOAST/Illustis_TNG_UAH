import numpy as np
# import numba as nb
# from numba import njit
# import pandas as pd
from matplotlib import pyplot as plt
import scipy
from numba import njit
from scipy.ndimage import gaussian_filter as gf
from scipy.stats import linregress
# import math as mth
# import matplotlib as mpl
# from matplotlib.colors import LogNorm
# from matplotlib import ticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib as mpl
#import random
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

fil_grid = np.load('fil_gri.npy')
fil_grid = np.bool_(fil_grid)
#WHIM_Overden_nofilter = np.load('GRID_WHIM_DENSITY_NOMASKhalos.npy')
#WHIM_Overden_nofilter = WHIM_Overden_nofilter / .610 * 10 ** 10
mean_baryon_den = (0.618 * 10 ** 10)

################## INSERT THE GRID CONSTRUCTION ####################################
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

@njit
def Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden):
    LD_smooth_param = 1.2 / cell_len  # smoothing 1.2 Mpc converted to Mpc/cell_len (dimensionless), Units of Mpc
    for n in range(len(x)):
        LD_Overden[x_idx[n] - 1, y_idx[n] - 1, z_idx[n] - 1] = LD_Overden[x_idx[n] - 1, y_idx[n] - 1, z_idx[n] - 1] + lum_r[n] / lum_avg_den / (box_len / num_cells) ** 3  # matching luminosity values with binned coordinates
    return LD_Overden
Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden)  # running LD function

#LD_Overden, lum_edges = np.histogramdd(lum_xyz, bins = (x_bins, y_bins, z_bins), weights = lum_r / lum_avg_den / (box_len / num_cells) ** 3)


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

####################### Reading in WHIM dataset from Daniela ###################
    ################ Loading in density grid constructed by Daniela ##########################
    WHIM_den_file = 'GRID_WHIM_DENSITY_NOMASKhaloes.npy'
    WHIM_Overden = np.load(WHIM_den_file) #loading new WHIM densiyt file from Daniela
    WHIM_Overden = WHIM_Overden #/ mean_baryon_den
    print('WHIM density range for cells:',min(WHIM_Overden.flatten()),max(WHIM_Overden.flatten()))
    print("number of bins with WHIM (den values):",np.size(np.nonzero(WHIM_Overden.flatten())))
############## Plotting ##########################################
############ Selecting WHIM and LD within filaments ##################################################

    WHIM_scat = WHIM_Overden[fil_grid].flatten() #WHIM den outside of filaments
    LD_scat = LD_Overden[fil_grid].flatten() #LD outside of filaments

    LD_scat = LD_scat[np.nonzero(WHIM_scat)]
    WHIM_scat = WHIM_scat[np.nonzero(WHIM_scat)]

    WHIM_scat = WHIM_scat[np.nonzero(LD_scat)]
    LD_scat = LD_scat[np.nonzero(LD_scat)]
    WHIM_scat = np.log10(WHIM_scat)
    LD_scat = np.log10(LD_scat)

    print('length of WHIM scat and LD scat',len(WHIM_scat), len(LD_scat))


############# linear correlation coefficient test ###########################################
    slope[k], intercept[k], corr_coeff[k], pval[k], std_err[k] = linregress(LD_scat, WHIM_scat)

plt.plot(smooth_param, corr_coeff)
plt.xlabel('Smoothing Parameter [Mpc]')
plt.ylabel('Correlation Coefficient')
#plt.savefig('filament_corr_plots/correlation_smoothing_plot_filrad_30.pdf')
#plt.close()
plt.axvline(x=1.2,color = 'b', linestyle = '--', label = 'Smoothing Parameter of 1.2')
plt.legend()
plt.grid()
plt.show()

quit()
##########################  Plotting Scatter of LD vs. WHIM density ###########################

# #fig, ax=plt.subplots(1, figsize = (8,8))
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
