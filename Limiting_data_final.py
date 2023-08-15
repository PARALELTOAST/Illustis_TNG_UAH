#import bisect
#import pickle
#from multiprocessing import Pool, Manager
#import numpy
import numpy as np
#import numba as nb
from numba import njit
import pandas as pd
from scipy.ndimage import gaussian_filter as gf
#import tqdm
#import json
#from multiprocessing import Process, Queue
#import scipy
#from scipy import spatial
#import math as mth
import matplotlib as mpl
from matplotlib.colors import LogNorm
#from matplotlib import ticker
#import os
import matplotlib.pyplot as plt
#from matplotlib.cm import register_cmap, cmap_d
import ill_functions as fun
import h5py
import illustris_python as il
from mpl_toolkits.axes_grid1 import make_axes_locatable

#fun.limiter('lum_dust', True, True, [0],[0], working_path='/home/benjamin/Thesis/Illustris_TNG')

# try 10^14 10^23 10^12 msun
# loading in the data
#dust_corrected_file = '/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5'
dust_corrected_file = '/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5'

with h5py.File(dust_corrected_file, "r") as partData:
    subhalo_mag_r_dust = np.asarray(partData["/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:,2,0],dtype='f4')

min_mag_r = min(subhalo_mag_r_dust)
max_mag_r = max(subhalo_mag_r_dust)
print("min and max of mag_r, dust corrected:",min_mag_r, max_mag_r)
#quit()

#path = '/home/pat/Physics/illustris_python-master/tng_300'
path = '/home/benjamin/Thesis/Illustris_TNG/tng_300'

halo_fields = ['GroupPos', 'GroupMass', 'GroupMassType', 'Group_M_Crit200', 'Group_M_Crit500', 'Group_R_Crit200', 'Group_R_Crit500', 'GroupNsubs', 'GroupLenType', 'GroupLen'] #check how much of gas lies within R200, R500. See ratio of gas mass to total mass. Compare to EAGLE Toni's paper.
subhalo_fields = ['SubhaloFlag','SubhaloMass','SubhaloMassType', 'SubhaloPos', 'SubhaloStellarPhotometrics', 'SubhaloGrNr']
halos = il.groupcat.loadHalos(path,99,fields=halo_fields)
subhalos = il.groupcat.loadSubhalos(path,99,fields=subhalo_fields)

halo_coord_x = halos['GroupPos'][:,2] / (0.6774*10**3)
halo_coord_y = halos['GroupPos'][:,1] / (0.6774*10**3)
halo_coord_z = halos['GroupPos'][:,0] / (0.6774*10**3)
halo_rad_crit200 = halos['Group_R_Crit200'][:] / (0.6774*10**3)
halo_mass = halos['GroupMass'] / 0.6774 * 10**10 #cluster mass, includes all particle types
halo_mass_gas = halos['GroupMassType'][:,0] /0.6774 *10**10#gas mass in cluster
halo_gal_count = halos['GroupNsubs'][:]
print('number of cluster positions and number of r200 radii',len(halo_coord_x), len(halo_rad_crit200))
print('checking the total number of subhalos in all halos, no filtering of halo or subhalos:',sum(halo_gal_count))
print('radii range for clusters',min(halo_rad_crit200), max(halo_rad_crit200))
#quit()
print('halo(cluster) mass range:',min(halo_mass),max(halo_mass))

# applying filter to halo positions for 10^14, luminosity, mass, gal count, and flag masks applied.
# Creating a mask to choose magnitude range for r-band to match SDSS data. Refer to Toni's paper.
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

mask = np.ones(len(subhalo_mag_r_dust),dtype=np.bool)
Lum_mask = mask*(subhalo_mag_r_dust < -18.014) #-18.4 for EAGLE...-18.014 for TNG300, according to Daniela. Provides the same number density as EAGLE and SDSS.
subhalo_group_index = list(set(subhalo_group_index[Lum_mask*Flag_mask]))

##### Halo Masks ########
mask_halo = np.ones(len(halo_gal_count),dtype=np.bool)
halo_gal_count_mask = mask_halo*(halo_gal_count > 0)
# galaxy_mass_mask = mask*(subhalo_mass_stellar > 14.0) #(subhalo_mass_stellar <= 10**12)*(subhalo_mass_stellar >= 10**9) #mask for the galaxy stellar masses. Some seem unusual
galaxy_mass_mask = mask_halo*(halo_mass > 10 ** 10) #10^14

#halo_pos_xyz = ([halo_coord_x[galaxy_mass_mask*Flag_mask*halo_gal_count_mask*Lum_mask], halo_coord_y[galaxy_mass_mask*Flag_mask*halo_gal_count_mask*Lum_mask], halo_coord_z[galaxy_mass_mask*Flag_mask*halo_gal_count_mask*Lum_mask]])
#halo_coord_x = halo_coord_x[halo_gal_count_mask * galaxy_mass_mask]
#halo_coord_y = halo_coord_y[halo_gal_count_mask * galaxy_mass_mask]
#halo_coord_z = halo_coord_z[halo_gal_count_mask * galaxy_mass_mask]
#halo_r200 = halo_rad_crit200[galaxy_mass_mask*halo_gal_count_mask]
halo_r200=halo_rad_crit200
halo_pos_xyz = np.transpose([halo_coord_x, halo_coord_y, halo_coord_z])
#halo_r200 = halo_rad_crit200
#halo_pos_xyz = np.loadtxt('halo_positions.txt')
#halo_r200 = np.loadtxt('halo_r200.txt')

x_halo = halo_coord_x
y_halo = halo_coord_y
z_halo = halo_coord_z

# Excluding the WHIM from the Halos, already ran commenting out
fun.limiter('lum_mass_flag_galcount', False, False, halo_pos_xyz, halo_r200, '/home/benjamin/Thesis/Illustris_TNG')


##########t plotting distribution of WHIM ##################
#plt.hist(np.log10(den_WHIM),np.linspace(min(np.log10(den_WHIM)),max(np.log10(den_WHIM)),50),density=False)
#plt.xlabel(r'$\log(\rho)$',fontsize=15)
#plt.savefig('WHIMden_dist.pdf')
#plt.close()
mag_sun_r = 4.42
lum_r = 10.0**((subhalo_mag_r_dust - mag_sun_r)/-2.5)
lum_avg_den = sum(lum_r.flatten()) / 303.**3
###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600 #1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303. #Units of Mpc
#cell_len = box_len/num_cells
LD_Overden = np.zeros([num_cells,num_cells,num_cells])
x = subhalo_coord[Lum_mask*Flag_mask,2] / (0.6774*10**3)
y = subhalo_coord[Lum_mask*Flag_mask,1] / (0.6774*10**3)
z = subhalo_coord[Lum_mask*Flag_mask,0] / (0.6774*10**3)

del subhalo_coord

x_bins = np.linspace(0.0,303.0,num_cells + 1)
y_bins = np.linspace(0.0,303.0,num_cells + 1)
z_bins = np.linspace(0.0,303.0,num_cells + 1)

x_idx = np.digitize(x,x_bins)
y_idx = np.digitize(y,y_bins)
z_idx = np.digitize(z,z_bins)


@njit
def Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden):
    cell_len = box_len/num_cells
    LD_smooth_param = 1.2/cell_len #smoothing 1.2 Mpc converted to Mpc/cell_len (dimensionless), Units of Mpc
    for n in range(len(x)):
        LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + lum_r[n]/lum_avg_den/(box_len/num_cells)**3 #matching luminosity values with binned coordinates
    return LD_Overden
Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden) #running LD function

print("number of galaxies for smoothing:",np.size(x))
print("dimensions of LD grid",np.shape(LD_Overden))
print("number of bins with galaxies (lum values):",np.size(np.nonzero(LD_Overden.flatten())))
########### smoothing using Gaussian kernel ####################################
smooth_param = 2.0 #Units of Mpc
LD_Overden = gf(LD_Overden, sigma=0.6*smooth_param)

print('max and min lum overden',max(LD_Overden.flatten()),min(LD_Overden.flatten()))
#quit()
########### Smoothing using B3 Spline #########################################
#### maybe use scipy.interpolate.CubicSpline for each coordinate x,y,z ########

####################### Reading in WHIM dataset from Daniela ###################

WHIM_Data = pd.read_csv('dataset_WHIMgas__tng3001_z0__ratio400.csv')
WHIM_Data = np.asarray(WHIM_Data)

z_pos_WHIM = WHIM_Data[:,1] #x coordinate for WHIM in Mpc
y_pos_WHIM = WHIM_Data[:,2] #y ...
x_pos_WHIM = WHIM_Data[:,3] #z ...
mass_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/mass_WHIM_lum_mass_flag_galcount.txt')
den_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/den_WHIM_lum_mass_flag_galcount.txt') #units in M_sun * Mpc^-3
temp_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/temp_WHIM_lum_mass_flag_galcount.txt') #units in Kelvin (K)
press_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/press_WHIM_lum_mass_flag_galcount.txt') #pressure in units of KeV*cm^-3
volume = []
for i in range(len(den_WHIM)):
    if den_WHIM[i]==0:
        volume.append(0)
    else:
        volume.append(mass_WHIM[i]/den_WHIM[i])
volume = np.array(volume)
#boltz_const = 1.380649 * 10**-23
#boltz_const * temp_WHIM * (6.242*10^15) #converting Joules to keV
x_idx = np.digitize(x_pos_WHIM,x_bins)
y_idx = np.digitize(y_pos_WHIM,y_bins)
z_idx = np.digitize(z_pos_WHIM,z_bins)

mean_baryon_den = 0.618 * 10**10 #mean baryon density of universe in units of M_sun * Mpc^-3
WHIM_Den_avg = sum(mass_WHIM) * 400. / 303.**3 #WHIM gas sampled selecting 1 out of 400.
WHIM_Den_avg_2 = sum(mass_WHIM)/ sum(volume)
print("WHIM density average of whole box:",WHIM_Den_avg, WHIM_Den_avg_2)
print("volume range and cell volume:",min(volume), max(volume), (box_len/num_cells)**3.)
#quit()
WHIM_Overden = np.zeros([num_cells,num_cells,num_cells])
Temp_grid = np.zeros([num_cells,num_cells,num_cells]) #mass weighted mean temperature of each cell
Mass_grid = np.zeros([num_cells,num_cells,num_cells])
#Volume_grid = np.zeros([num_cells,num_cells,num_cells])
Pressure_grid = np.zeros([num_cells, num_cells, num_cells])

@njit
def WHIM_mass_func(box_len, num_cells, x, y, z, Mass_grid):
    cell_len = box_len/num_cells
    for n in range(len(x)):
        Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]
    return Mass_grid

WHIM_mass_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Mass_grid)

@njit
def WHIM_Pressure_func(box_len, num_cells, x, y, z, Pressure_grid):
    cell_len = box_len/num_cells
    for n in range(len(x)):
        if Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] != 0:
            Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*press_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
        else:
            Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = 0
    return Pressure_grid #Mass weighted mean pressure for each cell

WHIM_Pressure_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Pressure_grid)

@njit
def WHIM_Temp_func(box_len, num_cells, x, y, z, Temp_grid):
    cell_len = box_len/num_cells
    for n in range(len(x)):
        if Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] != 0:
            Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*temp_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
        else:
            Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = 0
    return Temp_grid #Mass weighted mean temperature for each cell

@njit
def WHIM_OverDen_func(box_len, num_cells, x, y, z, WHIM_Overden):
    cell_len = box_len/num_cells
    for n in range(len(x)):
        WHIM_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] / (box_len/num_cells)**3 * 400. / mean_baryon_den #Volume_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] /mean_baryon_den  #matching luminosity values with binned coordinates
    return WHIM_Overden

print('length of WHIM temp dataset:',len(temp_WHIM), len(mass_WHIM), len(den_WHIM))
#print('range of temperature:',min(temp_WHIM),max(temp_WHIM))
#print('distance range in x coordinate:',min(x_pos_WHIM),max(x_pos_WHIM))
#quit()
WHIM_Temp_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Temp_grid)
WHIM_OverDen_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, WHIM_Overden)

print('WHIM density range for cells:',min(WHIM_Overden.flatten()),max(WHIM_Overden.flatten()))
print('WHIM pressure range for cells:',min(Pressure_grid.flatten()),max(Pressure_grid.flatten()))
print('WHIM Temp range for cells:',min(Temp_grid.flatten()),max(Temp_grid.flatten()))
#print('min and max volume grid:',min(Volume_grid.flatten()),max(Volume_grid.flatten()))
#print('sum of the gas cell volume',sum(Volume_grid.flatten()))
#print('sum of gas  cell volume times 400',sum(Volume_grid.flatten()) * 400)
print("number of bins with WHIM (den values):",np.size(np.nonzero(WHIM_Overden.flatten())))
vol_WHIM = np.size(np.nonzero(WHIM_Overden.flatten())) / len(WHIM_Overden.flatten()) #* 303.**3
print("fraction of volume occupied by WHIM",vol_WHIM)
#some of the non-selected WHIM data may be in bins with no WHIM. Also try binning according to which gridpoit is closer to positions.
#quit()
############## Plotting ##########################################
###############  Filaments #######################################

filefils = 'filament_coords_and_lengths.npy'

# Read filaments
seg_z, seg_y, seg_x, lengths = np.load(filefils, allow_pickle=True)

print('Number of filaments:', len(seg_x))

#########################plotting the Whim Slices3#################################################
for i in range(20):
    z_in = 15*i #in Mpc
    z_fin = 15*(i+1) #in Mpc
    cmap_LD = mpl.cm.get_cmap('Oranges')
    cmap_WHIM = mpl.cm.get_cmap('Purples')
    fig, ax = plt.subplots(1, figsize = (8,8))

    zslice = (z > z_in) * (z < z_fin)
    zslice_halo = (z_halo > z_in) * (z_halo < z_fin)
    zslice_WHIMden = (z_pos_WHIM > z_in) * (z_pos_WHIM < z_fin)
    gridIndexMin = int((num_cells*z_in)/box_len)
    gridIndexMax = int((num_cells*z_fin)/box_len)
    imageSlice = np.mean(LD_Overden[gridIndexMin:gridIndexMax, :, :],axis=0)
    #imageVolumeSlice=np.mean(Volume_grid[gridIndexMin:gridIndexMax,:,:],axis=0)
    imageWHIMSlice=np.mean(WHIM_Overden[gridIndexMin:gridIndexMax,:,:],axis=0)
    vmin = 1e-6 #min(imageSlice.flatten())
    vmax = 1e3 #max(imageSlice.flatten())
    vmin_WHIM = min(imageWHIMSlice.flatten())
    vmax_WHIM = max(imageWHIMSlice.flatten())
    vmin_nonzero = imageSlice.flatten()
    vmin_nonzero = vmin_nonzero[np.nonzero(vmin_nonzero)]
    print("min and max LD for image slice:",vmin, vmax)
    print("min nonzero LD for image slice:",min(vmin_nonzero))
    ####### plotting filaments. Filaments given by Daniela ##########
    #for j in range(len(seg_x)):
        #if ( np.mean(seg_z[j]) > z_in ) & ( np.mean(seg_z[j]) <= z_fin ):
            #fil_scat = ax.plot(seg_x[j], seg_y[j], color='k')
    #gal_scat = ax.scatter(x[zslice], y[zslice], marker='o', s=1.0, color='red', alpha=0.7, label='galaxies')
    #cluster_scat = ax.scatter(x_halo[zslice_halo], y_halo[zslice_halo], marker='o', s=1.0, color='red', alpha=0.7, label='clusters')
    #halo_scat = ax.scatter(halo_coord_x[zslice_halo], halo_coord_y[zslice_halo], marker='o', s=1.0, color='red', alpha=0.7, label='halos')
    #WHIM_locations = ax.scatter(x_pos_WHIM[zslice_WHIMden], y_pos_WHIM[zslice_WHIMden], marker='o', s=1.0, color='red', alpha=0.7, label='WHIMpart')
    #LD_plot = ax.imshow(imageSlice, norm=LogNorm(vmin=1e-5, vmax=500.), extent=[0.0,303.0,0.0,303.0], aspect='auto', origin="lower", cmap=cmap_LD) #vmax=2000, vmin=1e-3, just like for EAGLE, only visual purposes
    #Volume_plot = ax.imshow(imageWHIMSlice, cmap=cmap_WHIM, extent= [0.,303.,0.,303.0], origin='lower',alpha=0.6, label="Volume")
    WHIMden_plot = ax.imshow(imageWHIMSlice, cmap=cmap_WHIM, norm=LogNorm(vmin=vmin , vmax=vmax), extent= [0.,303.,0.,303.0], origin='lower',alpha=0.6, label="WHIM") #vmin = 1e-06, vmax = 1e05
    plt.xlabel('X [Mpc]',fontsize=10)
    plt.ylabel('Y [Mpc]',fontsize=10)
    ax.legend(loc=1,fontsize=12)
    divider = make_axes_locatable(ax)
    #cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
    #fig.add_axes(cax)
    #cbar = fig.colorbar(LD_plot, cax=cax, orientation = "horizontal", format=ticker.LogFormatter())
    #cbar.set_label(r'$\delta_{LD}$')#Luminosity Overdensity') #($10^{10} L_{\odot} Mpc^{-3}$)')
    #cbar.ax.tick_params(labelsize=10, width=0.7, size=8)
    #cbar.solids.set_edgecolor("face")
    #cbar.ax.xaxis.set_label_position('bottom')
    ####### colorbar for WHIM #########################
    cax2 = divider.new_vertical(size='5%', pad=0.6)
    fig.add_axes(cax2)
    cbar2 = fig.colorbar(WHIMden_plot, cax=cax2, orientation = "horizontal") #, format=ticker.LogFormatter())
    cbar2.set_label(r'$\delta_{\rho}$')#WHIM Overdensity') #($10^{10} M_{\odot} Mpc^{-3}$)')
    cbar2.ax.tick_params(labelsize=10, width=0.7, size=8)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.xaxis.set_label_position('bottom')

    #plt.savefig('WHIM_high_den_10to12_part_position_slice{i}.pdf'.format(i=i))
    #plt.savefig('WHIM_within_clusters_slice{i}.pdf'.format(i=i))
    plt.savefig('WHIM_within_halos_slice{i}.pdf'.format(i=i))
    #plt.savefig('halo_position_slice{i}.pdf'.format(i=i))
    #plt.savefig('Volume_grid_slice{i}.pdf'.format(i=i))
    #plt.savefig('WHIM_density_filaments_galaxies_slice{i}.pdf'.format(i=i))
    #plt.savefig('WHIM_density_slice{i}.pdf'.format(i=i))
    #plt.savefig('galaxy_positions_LD_slice{i}.pdf'.format(i=i))
    #plt.savefig('galaxies_and_fils{i}.pdf'.format(i=i))
    #plt.savefig('Select_galaxy_positions_LD_slice{i}.pdf'.format(i=i))
    #print('num of galaxies in slice:',np.size(x[zslice]))
    plt.show()
    plt.close()


'''
# The halo and whim information
limiters = 'luminosity, dust'
print('Getting halo and whim info')
WHIM_Data = pd.read_csv('dataset_WHIMgas__tng3001_z0__ratio400.csv')
WHIM_Data = np.asarray(WHIM_Data)
########################################################################################
# ask about original document has x nd z swapped
z_pos_WHIM = WHIM_Data[:, 1]  # z coordinate for WHIM in Mpc
y_pos_WHIM = WHIM_Data[:, 2]  # y ...
x_pos_WHIM = WHIM_Data[:, 3]  # x ...
###############################################################
# WHIM_position = sorted(list(zip([i for i in range(0, len(x_pos_WHIM))], x_pos_WHIM)), key=lambda x: x[1])

halo_rs = []  # holder for the radii of the halos
halos = np.array(np.loadtxt('halo_positions.txt'))  # txt file of all halo pos x,y,z
mass_WHIM = WHIM_Data[:, 4]  # units in M_sun
den_WHIM = WHIM_Data[:, 5]  # units in M_sun * Mpc^-3
temp_WHIM = WHIM_Data[:, 8]  # units in Kelvin (K)
press_WHIM = WHIM_Data[:, 9]  # pressure in units of KeV*cm^-3

print("Reading halo_r200.txt")
with open('halo_r200.txt') as file:
    while line := file.readline().rstrip():
        halo_rs.append(float(line))  # add each radius to the list

# This is where the filtering/sorting/magic happens
# This loads a KD-Tree (K-dimensional Tree) (here it is 3 for x,y,z) which basically lets you quickly find the points that are within a certain distance of another point
# This means that we can give a query with the coordinates of a halo and it's radius and it will return to us all the points we need
# This is what was attempted ealier where searching was just the x-axis, this does all three at the same time
# Here is the link for the KDtree page: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html#scipy.spatial.KDTree.query_ball_point
print("loading tree...")

# scipy.spatial.KDTree.query_ball_point(halos, halo_rs, p=2.0, eps=0, workers=1, return_sorted=None, return_length=False)
points = [(x, y, z) for x, y, z in zip(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM)]
tree = scipy.spatial.KDTree(points)
print("tree loaded")

# The filtering
total = 0
whim_in_halo_index = []
for i in tqdm.trange(len(halos)):
    points_in_radius = tree.query_ball_point(halos[i], halo_rs[i],
                                             workers=-1)  # setting workers to -1 maximizes parallelization
    total += len(
        points_in_radius)  # to keep track of the total number of points that got zeroed out (although this could double count)
    for point in points_in_radius:
        den_WHIM[point] = 0
        temp_WHIM[point] = 0
        press_WHIM[point] = 0
        mass_WHIM[point] = 0
        whim_in_halo_index.append([point, i])
    points_in_radius = np.array(points_in_radius)
    for j in range(len(points_in_radius)):
        whim_in_halo_index.append(points_in_radius[j], i)

# making the folder the data will be saved to
working_path = '/home/benjamin/Thesis/Illustris_TNG'
path = os.path.join(working_path, 'Limited_Data')
print('Saving indices of WHIM with the halos it is in...' + path)
if not os.path.exists(path):
    os.mkdir(path)
np.savetxt(os.path.join(path, 'whim_in_halo_index.txt'), whim_in_halo_index, fmt='%s')
np.savetxt(os.path.join(path, 'den_WHIM_' + limiters + '.txt'), den_WHIM, fmt='%s')
np.savetxt(os.path.join(path, 'temp_WHIM_' + limiters + '.txt'), temp_WHIM, fmt='%s')
np.savetxt(os.path.join(path, 'press_WHIM_' + limiters + '.txt'), press_WHIM, fmt='%s')
np.savetxt(os.path.join(path, 'mass_WHIM_' + limiters + '.txt'), mass_WHIM, fmt='%s')

print('limited data has been saved to: ' + path)

working_path = '/home/benjamin/Thesis/Illustris_TNG'
path = os.path.join(working_path, 'Limited_Data')
whim_in_halo_index = (np.loadtxt(os.path.join(path, 'whim_in_halo_index.txt')))

whim_in_halo_index = np.loadtxt('whim_in_halo_index.txt')

print('Total number of Whim in halos:', total)
# whim_in_halo_index = np.asarray(whim_in_halo_index)
# filtering for repeated points

# whim_in_one_halo = numpy.unique(whim_in_halo_index, axis=0)
# ssss, index_repeated_whim = np.unique(whim_in_halo_index, axis=1, return_inverse=True)
# whim_in_halo_index.sort()

# whim_in_one_halo, index_repeated_whim= numpy.unique(whim_in_halo_index, axis=0, return_inverse=True) # is an array of unique entries, focusing on the first index
# Whim in one halo is the unique entries, sorted, but still keep correlation between the Whim index and Halo index
# index_repeated_whim is the index of whim_in_one_halo of the repeated whim value
# Whim_in_one_halo[index_repeated_whim] will reproduce the original array
# print('Total number of unique,(Whim is only in one halo) Whim in halos:',len(whim_in_one_halo))
# print('Number of overlapping whim (Whim in multiple halos):',len(index_repeated_whim))'''


'''def compute(halo, hr, x_sorted, y_pos_WHIM, z_pos_WHIM, den_WHIM, temp_WHIM, press_WHIM, mass_WHIM):
    try:
        haloxyz = np.array(halo)
        points_in_sphere = []
        start = bisect.bisect_left(x_sorted, haloxyz[0] - hr, key=lambda p: p[1])
        while start < len(x_sorted) and x_sorted[start][1] <= haloxyz[0] + hr:
            x_coord = x_sorted[start][1]
            y_coord = y_pos_WHIM[x_sorted[start][0]]
            z_coord = z_pos_WHIM[x_sorted[start][0]]
            distance = np.sum((np.array((x_coord, y_coord, z_coord)) - haloxyz) ** 2) ** .5
            if distance <= hr:
                den_WHIM[start] = 0
                temp_WHIM[start] = 0
                press_WHIM[start] = 0
                mass_WHIM[start] = 0
                points_in_sphere.append(start)
                # x_sorted.remove(start)
            start = start + 1
        if len(points_in_sphere) > 0:
            print(f"There are {len(points_in_sphere)} points in the sphere")
    except Exception as e:
        print(f"Failed to compute: {e}")

def main():
    with Manager() as manager:
        z_pos_WHIM = WHIM_Data[:, 3]  # x coordinate for WHIM in Mpc
        y_pos_WHIM = WHIM_Data[:, 2]  # y ...
        mass_WHIM = WHIM_Data[:, 4]  # units in M_sun
        den_WHIM = WHIM_Data[:, 5]  # units in M_sun * Mpc^-3
        temp_WHIM = WHIM_Data[:, 8]  # units in Kelvin (K)
        press_WHIM = WHIM_Data[:, 9]  # pressure in units of KeV*cm^-3
        halos=np.loadtxt('halo_positions.txt')
        halo_rad_crit200 = np.loadtxt('halo_r200.txt')

        halos       = manager.list(halos)
        halo_rs     = manager.list(halo_rad_crit200)
        x_sorted    = manager.list(WHIM_position)
        y_pos_WHIM  = manager.list(y_pos_WHIM)
        z_pos_WHIM  = manager.list(z_pos_WHIM)
        den_WHIM    = manager.list(den_WHIM)
        temp_WHIM   = manager.list(temp_WHIM)
        press_WHIM  = manager.list(press_WHIM)
        mass_WHIM   = manager.list(mass_WHIM)
        with Pool(processes=24) as pool:
            for i in tqdm.trange(0, len(halos)):
                pool.map_async(compute, ((halos[i], halo_rs[i], x_sorted, y_pos_WHIM,
                                          z_pos_WHIM, den_WHIM, temp_WHIM, press_WHIM, mass_WHIM)))


if __name__ == '__main__':
    main()'''
