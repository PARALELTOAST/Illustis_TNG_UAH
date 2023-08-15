import numpy as np
import numba as nb
from numba import njit
import pandas as pd
#import cython
#from distutils.cor import setup
#from Cython.Build import cythonize
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter as gf
import math as mth
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
#import random
from functools import partial
#from matplotlib.cm import register_cmap, cmap_d
import h5py
import argparse #add options and arguments when running script in command line
import sys
import illustris_python as il
#import plot_web as cweb #Python script by Chris Duckworth. cite papers

#Reading in zero redshift galaxies with dust correction
dust_corrected_file = "/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5"
with h5py.File(dust_corrected_file, "r") as partData:
    subhalo_mag_r_dust = np.asarray(partData["/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:,2,0],dtype='f4')

min_mag_r = min(subhalo_mag_r_dust)
max_mag_r = max(subhalo_mag_r_dust)
print("min and max of mag_r, dust corrected:",min_mag_r, max_mag_r)
#quit()
#path = '/home/pat/Physics/illustris_python-master/tng_300'
path = '/home/benjamin/Thesis/Illustris_TNG/tng_300'
halo_fields = ['GroupPos', 'GroupMass', 'GroupMassType', 'Group_M_Crit200', 'Group_M_Crit500', 'Group_R_Crit200', 'Group_R_Crit500', 'GroupNsubs', 'GroupLenType', 'GroupLen'] #check how much of gas lies within R200, R500. See ratio of gas mass to total mass. Compare to EAGLE Toni's paper.
subhalo_fields = ['SubhaloFlag','SubhaloMass','SubhaloMassType', 'SubhaloPos', 'SubhaloStellarPhotometrics']
halos = il.groupcat.loadHalos(path,99,fields=halo_fields)
subhalos = il.groupcat.loadSubhalos(path,99,fields=subhalo_fields)

halo_coord_x = halos['GroupPos'][:,2]
halo_coord_y = halos['GroupPos'][:,1]
halo_coord_z = halos['GroupPos'][:,0]
halo_rad_crit200 = halos['Group_R_Crit200'][:] / (0.6774*10**3)
halo_mass = halos['GroupMass'] / 0.6774 * 10**10 #cluster mass, includes all particle types
halo_mass_gas = halos['GroupMassType'][:,0] /0.6774 *10**10#gas mass in cluster
halo_gal_count = halos['GroupNsubs'][:]
print('number of cluster positions and number of r200 radii',len(halo_coord_x), len(halo_rad_crit200))
print('checking the total number of subhalos in all halos, no filtering of halo or subhalos:',sum(halo_gal_count))
print('radii range for clusters',min(halo_rad_crit200), max(halo_rad_crit200))
#quit()
print('halo(cluster) mass range:',min(halo_mass),max(halo_mass))
#subhalomass = subhalos['SubhaloMassType']
Flag_mask = subhalos['SubhaloFlag'] #array with value 1 or 0. 1 meaning, it's an actual galaxy, 0 values should be excluded.
subhalo_mass = subhalos['SubhaloMass'] #galaxy mass, includes all particle types in simulation
subhalo_mass_stellar = subhalos['SubhaloMassType'][:,4] #galaxy mass, only stellar particles.
subhalo_coord = subhalos['SubhaloPos'] #subhalo (galaxy) position
subhalo_mag_r = subhalos['SubhaloStellarPhotometrics'][:,5] #galaxy magnitude r-band. Convert to luminosity
print("min and max of mag_r, no filter:",min(subhalo_mag_r), max(subhalo_mag_r))
subhalo_mass = subhalo_mass /0.6774 *10**10 #subhalo_mass in solar masses
subhalo_mass_stellar = subhalo_mass_stellar /0.6774 *10**10 #subhalo stellar mass in solar masses

#Creating a mask to choose magnitude range for r-band to match SDSS data. Refer to Toni's paper.
mask = np.ones(len(subhalo_mag_r),dtype=np.bool)
Lum_mask = mask*(subhalo_mag_r_dust < -18.014) #-18.4 for EAGLE...-18.014 for TNG300, according to Daniela. Provides the same number density as EAGLE and SDSS.

####### masks for groups/halos (clusters) ##############################
mask_halo = np.ones(len(halo_gal_count),dtype=np.bool)
halo_gal_count_mask = mask_halo*(halo_gal_count > 0)
print('number of cluster positions after applying subhalo count filter',len(halo_coord_x[halo_gal_count_mask]))
print('halo(cluster) mass range after applying subhalo count filter:',min(halo_mass[halo_gal_count]),max(halo_mass[halo_gal_count]))
galaxy_mass_mask = mask*(subhalo_mass_stellar > 0.0) #(subhalo_mass_stellar <= 10**12)*(subhalo_mass_stellar >= 10**9) #mask for the galaxy stellar masses. Some seem unusual
galaxy_lum_r = 10.0**((subhalo_mag_r_dust - 4.42)/-2.5)
lum_massflag_only = galaxy_lum_r[Flag_mask*galaxy_mass_mask]
galaxy_lum_r = galaxy_lum_r[Flag_mask*Lum_mask]

lum_avg_den = sum(galaxy_lum_r.flatten()) / 303.**3
print("Average luminosity density of whole volume, with only flagged bad galaxies excluded and lum filter applied:",lum_avg_den)
print("min and max of galaxy masses, no filtering:",min(subhalo_mass),max(subhalo_mass))
print("min and max of galaxy masses, only flag and lum filtering:",min(subhalo_mass[Lum_mask*Flag_mask]),max(subhalo_mass[Lum_mask*Flag_mask]))
print("min and max of galaxy stellar masses, only lum and flag filtering:", min(subhalo_mass_stellar[Lum_mask*Flag_mask]), max(subhalo_mass_stellar[Lum_mask*Flag_mask]))
print("min and max of galaxy stellar masses,filtering:", min(subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]), max(subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]))
print("number of subhaloes, no filtering at all:",len(subhalo_mass))
print("number of flagged subhaloes:",len(subhalo_mass) - len(subhalo_mass[Flag_mask]))
#print("number of galaxies only mass filtering:",np.shape(lum_massflag_only))
print("number of galaxies with only lumfiltering and flagged filter:",len(subhalo_mass[Flag_mask*Lum_mask]))
subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask*Flag_mask*galaxy_mass_mask]
subhalo_mass = subhalo_mass[Lum_mask*Flag_mask*galaxy_mass_mask]
subhalo_mass_stellar = subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]

np.savetxt("subhalo_mass_total.txt",subhalo_mass,fmt="%s")
np.savetxt("subhalo_stellar_mass.txt",subhalo_mass_stellar,fmt="%s")
np.savetxt("subhalo_r_magnitude_dust.txt",subhalo_mag_r_dust,fmt="%s")

#subhalo_mag_r = subhalo_mag_r[Flag_mask] checking number of flagged galaxies
#subhalo_mass = subhalo_mass[Flag_mask] 
#subhalo_mag_r = subhalo_mag_r[Lum_mask*Flag_mask]
#subhalo_mass = subhalo_mass[Lum_mask*Flag_mask]
print("number of galaxies, filtered:",len(subhalo_mass))
#quit()
##### checking max and min of r-band magnitude Also converting magnitude to luminosity ############
mag_sun_r = 4.42 #absolute magnitude of sun in r-band.
#print("min and max of magnitude, filtered:",min(subhalo_mag_r_dust), max(subhalo_mag_r_dust))
#print("min and max of masses, filtered:",min(subhalo_mass), max(subhalo_mass))
#quit()
lum_r = 10.0**((subhalo_mag_r_dust - mag_sun_r)/-2.5)
lum_avg_den = sum(lum_r.flatten()) / 303.**3
print("average luminosity density of full volume, filtered:",lum_avg_den)
#print("min and max of luminosity, filtered:",min(lum_r),max(lum_r))
#quit()
#histogram plot of masses after filtering
gal_mass = np.log10(subhalo_mass_stellar)
plt.hist(gal_mass,bins=np.linspace(min(gal_mass),max(gal_mass),50), histtype='bar',alpha=0.5, density=False)
plt.xlabel(r'$\log(M*)$',fontsize=12)
plt.savefig("subhalo_mass_histogram_dust.pdf")
plt.close()
#quit()
###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600 #1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303. #Units of Mpc
#cell_len = box_len/num_cells
LD_Overden = np.zeros([num_cells,num_cells,num_cells])
x = subhalo_coord[Lum_mask*Flag_mask*galaxy_mass_mask,2] / (0.6774*10**3)
y = subhalo_coord[Lum_mask*Flag_mask*galaxy_mass_mask,1] / (0.6774*10**3)
z = subhalo_coord[Lum_mask*Flag_mask*galaxy_mass_mask,0] / (0.6774*10**3)

np.savetxt("subhalo_positions.txt", np.transpose([x,y,z]),fmt="%s")

x_bins = np.linspace(0.0,303.0,num_cells + 1)
y_bins = np.linspace(0.0,303.0,num_cells + 1)
z_bins = np.linspace(0.0,303.0,num_cells + 1)

x_idx = np.digitize(x,x_bins)
y_idx = np.digitize(y,y_bins)
z_idx = np.digitize(z,z_bins)

######## re-binning according to which gridpoint is closest ########
#@njit
#def rebin_positions():
#    for n in range(len(x)):
#        dist = np.sqrt((x[n] - (float(x_idx[n]) - 1./2.)*box_len/num_cells)**2 + (y[n] - (float(y_idx[n]) - 1./2.)*box_len/num_cells)**2 + (z[n] - (float(z_idx[n]) - 1./2.)*box_len/num_cells)**2)
#        dist_x1 = np.sqrt((x[n] - (float(x_idx[n] - 1) - 1./2.)*box_len/num_cells)**2 + (y[n] - (float(y_idx[n]) - 1./2.)*box_len/num_cells)**2 + (z[n] - (float(z_idx[n]) - 1./2.)*box_len/num_cells)**2)
#        dist_x2 = np.sqrt((x[n] - (float(x_idx[n] + 1) - 1./2.)*box_len/num_cells)**2 + (y[n] - (float(y_idx[n]) - 1./2.)*box_len/num_cells)**2 + (z[n] - (float(z_idx[n]) - 1./2.)*box_len/num_cells)**2)
#        dist_y1 = np.sqrt((x[n] - (float(x_idx[n]) - 1./2.)*box_len/num_cells)**2 + (y[n] - (float(y_idx[n] - 1) - 1./2.)*box_len/num_cells)**2 + (z[n] - (float(z_idx[n]) - 1./2.)*box_len/num_cells)**2)
#        dist_y2 = np.sqrt((x[n] - (float(x_idx[n]) - 1./2.)*box_len/num_cells)**2 + (y[n] - (float(y_idx[n] + 1) - 1./2.)*box_len/num_cells)**2 + (z[n] - (float(z_idx[n]) - 1./2.)*box_len/num_cells)**2)
#        dist_z1 = np.sqrt((x[n] - (float(x_idx[n]) - 1./2.)*box_len/num_cells)**2 + (y[n] - (float(y_idx[n]) - 1./2.)*box_len/num_cells)**2 + (z[n] - (float(z_idx[n] - 1) - 1./2.)*box_len/num_cells)**2)
#        dist_z2 = np.sqrt((x[n] - (float(x_idx[n]) - 1./2.)*box_len/num_cells)**2 + (y[n] - (float(y_idx[n]) - 1./2.)*box_len/num_cells)**2 + (z[n] - (float(z_idx[n] + 1) - 1./2.)*box_len/num_cells)**2)
#        dist_min = min([dist, dist_x1, dist_x2, dist_y1, dist_y2, dist_z1, dist_z2])
#        if dist_min == dist_x1:

#if 1==1:
#@nb.jit(nopython=True)
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
smooth_param = 1.2 #Units of Mpc
LD_Overden = gf(LD_Overden, sigma=0.6*smooth_param)

#print('max and min lum overden',max(LD_Overden.flatten()),min(LD_Overden.flatten()))
#quit()
########### Smoothing using B3 Spline #########################################
#### maybe use scipy.interpolate.CubicSpline for each coordinate x,y,z ########

####################### Reading in WHIM dataset from Daniela ###################

WHIM_Data = pd.read_csv('dataset_WHIMgas__tng3001_z0__ratio400.csv')
WHIM_Data = np.asarray(WHIM_Data) 

z_pos_WHIM = WHIM_Data[:,1] #x coordinate for WHIM in Mpc
y_pos_WHIM = WHIM_Data[:,2] #y ...
x_pos_WHIM = WHIM_Data[:,3] #z ...
mass_WHIM = WHIM_Data[:,4] #units in M_sun
den_WHIM = WHIM_Data[:,5] #units in M_sun * Mpc^-3
temp_WHIM = WHIM_Data[:,8] #units in Kelvin (K)
press_WHIM = WHIM_Data[:,9] #pressure in units of KeV*cm^-3           
volume = mass_WHIM / den_WHIM #Originally commented out...

#boltz_const = 1.380649 * 10**-23
#boltz_const * temp_WHIM * (6.242*10^15) #converting Joules to keV
##########t plotting distribution of WHIM ##################
#plt.hist(np.log10(den_WHIM),np.linspace(min(np.log10(den_WHIM)),max(np.log10(den_WHIM)),50),density=False)
#plt.xlabel(r'$\log(\rho)$',fontsize=15)
#plt.savefig('WHIMden_dist.pdf')
#plt.close()

#plt.hist(np.log10(volume),np.linspace(min(np.log10(volume)),max(np.log10(volume)),50),density=False)
#plt.xlabel(r'$\log(V)$',fontsize=15)
#plt.savefig('volume_dist.pdf')
#plt.close()

#plt.hist(np.log10(mass_WHIM),np.linspace(min(np.log10(mass_WHIM)),max(np.log10(mass_WHIM)),50),density=False)
#plt.xlabel(r'$\log(M)$',fontsize=15)
#plt.savefig('WHIMmass_dist.pdf')
#plt.close()

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
        Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*press_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
    return Pressure_grid #Mass weighted mean pressure for each cell

#WHIM_Pressure_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Pressure_grid)

@njit
def WHIM_Temp_func(box_len, num_cells, x, y, z, Temp_grid):
    cell_len = box_len/num_cells
    for n in range(len(x)):
        Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*temp_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
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
#WHIM_Temp_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Temp_grid)
WHIM_OverDen_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, WHIM_Overden)

print('The max and min values of WHIM', max(WHIM_Overden.flatten()), min(WHIM_Overden.flatten()))


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

fil_x = np.array([x for filament in seg_x for x in filament])
fil_y = np.array([y for filament in seg_y for y in filament])
fil_z = np.array([z for filament in seg_z for z in filament])
# sorted based on x
fil_cords = sorted([(x, y, z) for x, y, z in zip(fil_x, fil_y, fil_z)])
fil_cords = np.array(fil_cords)
fil_r = []
for i in range(len(seg_x)):
    r = (seg_x[i] ** 2 + seg_y[i] ** 2 + seg_z[i] ** 2) ** .5
    fil_r.append(r)

num_cells = 600 #1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303. #Units of Mpc
cell_len = box_len/num_cells
x_bins = np.linspace(0.0, 303.0, num_cells + 1)
y_bins = np.linspace(0.0, 303.0, num_cells + 1)
z_bins = np.linspace(0.0, 303.0, num_cells + 1)

fil_grid = np.zeros([num_cells, num_cells, num_cells])
for i in range(len(seg_x)):
    x_idx = np.digitize(seg_x[:][i],x_bins)
    y_idx = np.digitize(seg_y[:][i],y_bins)
    z_idx = np.digitize(seg_z[:][i],z_bins)
print('length of seg_x',len(seg_x[0][:]))
print('length of _index', x_idx, y_idx, z_idx)

seg_x_new = seg_x
seg_y_new = seg_y
seg_z_new = seg_z
########################### making a fillament grid #############################################

# Adding the points inbetween fillament locations to "smooth" the fillaments

def seg_points_func(seg_x_array, seg_y_array, seg_z_array, seg_x_array_new, seg_y_array_new, seg_z_array_new):
    steps = 20
    for n in range(len(seg_x_new)):
        for m in range(len(seg_x_new[n])):
            if m == len(seg_x_new[n] - 1):
                continue
            x_pos = (seg_x_new[n][m+1] - seg_x_new[n][m]) / steps
            y_pos = (seg_y_new[n][m+1] - seg_y_new[n][m]) / steps
            z_pos = (seg_z_new[n][m+1] - seg_z_new[n][m]) / steps

            seg_x_add = np.linespace(seg_x_new[n][m] + x_pos, seg_x_new[n][m+1] - x_pos, steps -1)
            seg_y_add = np.linespace(seg_y_new[n][m] + y_pos, seg_y_new[n][m+1] - y_pos, steps -1)
            seg_z_add = np.linespace(seg_z_new[n][m] + z_pos, seg_z_new[n][m+1] - z_pos, steps -1)

            seg_x_new[n] = np.insert(seg_x_new[n], m, seg_x_add)
            seg_y_new[n] = np.insert(seg_y_new[n], m, seg_y_add)
            seg_z_new[n] = np.insert(seg_z_new[n], m, seg_z_add)
    return seg_z_new, seg_x_new, seg_y_new

# digitizing and making this binned
seg_x_bin = seg_y_bin = seg_z_bin = []
for i in range(len(seg_x_new)):
    for j in range(len(seg_x_new[i])):
        seg_x_bin.append(seg_x_new[i][j])
        seg_y_bin.append(seg_y_new[i][j])
        seg_z_bin.append(seg_z_new[i][j])

x_idx = np.digitize(seg_x_bin, x_bins)
y_idx = np.digitize(seg_y_bin, y_bins)
z_idx = np.digitize(seg_z_bin, z_bins)

#def filament_func(box_len, num_cells, x, y, z, x_idx, y_idx, z_idx, fil_grid):
#    for n in range(len(x_idx)):
#        fil_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = fil_grid[z_idx[n] - 1, y_idx[n] - 1, z_idx[n] - 1] + fil_cords[n]
#    return fil_grid
# Patricks Fil grid construction
#@njit
def fillament_func(box_len, num_cells, x_idx, y_idx, z_idx, fil_grid):
    fil_rad = 1.0 #1MPC radius
    for n in range(len(seg_x_new)):
        k_max = int(fil_rad/(box_len/num_cells))
        for k in range(-1 * k_max, k_max + 1):
            for l in range(-1 * k_max, k_max + 1):
                for m in range(-1 * k_max, k_max + 1):
                    if z_idx[n] - 1 + k > num_cells - 1 or z_idx[n] -1 + k < 0 or y_idx[n] - 1 + l > num_cells - 1 or y_idx[n] -1 + l < 0 or x_idx[n] - 1 + m > num_cells - 1 or x_idx[n] -1 + m < 0:
                        continue
                    fil_grid[z_idx[n] - 1 + k, y_idx[n] - 1 + l, x_idx[n] - 1 + m] = True
    return fil_grid

fillament_func(box_len,num_cells,x_idx,y_idx,z_idx,fil_grid)

print('max value of fil grid', np.max(fil_grid.flatten()))

######################################################################################################################
fil_grid = fil_grid.astype(int)

print(WHIM_Overden.shape)
print(LD_Overden.shape)
print('The max and min values of WHIM', max(WHIM_Overden.flatten()), min(WHIM_Overden.flatten()))
print('The max and min values of Lum', max(LD_Overden.flatten()), min(LD_Overden.flatten()))

WHIM_scat = WHIM_Overden * fil_grid
LD_scat = LD_Overden * fil_grid

WHIM_scat = WHIM_scat.flatten()
LD_scat = LD_scat.flatten()

# Removing zeros so the log10 function will work
LD_scat = LD_scat[np.nonzero(WHIM_scat)]
WHIM_scat = WHIM_scat[np.nonzero(WHIM_scat)]
WHIM_scat = WHIM_scat[np.nonzero(LD_scat)]
LD_scat = LD_scat[np.nonzero(LD_scat)]
logden = np.log10(WHIM_scat)
ldlog = np.log10(LD_scat)

#####################################################################################################################
fig, ax=plt.subplots(1, figsize = (8,8))
nbins = 40
bins = [np.linspace(min(ldlog),max(ldlog),nbins+1), np.linspace(min(logden),max(logden),nbins+1)]
h, xedge, yedge = np.histogram2d(ldlog, logden, bins=bins)
cmap = plt.get_cmap('Blues')
vmin = min(h.flatten()) + 1  #added a one so I can make the colormap logarithmic.
vmax = max(h.flatten())
X, Y = np.meshgrid(xedge,yedge)
im = plt.pcolormesh(X,Y,h,cmap=cmap,edgecolors='black',norm=LogNorm(vmin=vmin,vmax=vmax),linewidth=0.3)
plt.xlabel('Luminosity')
plt.ylabel('Whim Density')
plt.title('Plot for 1.2 smoothing factor')
plt.show()
