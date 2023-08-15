import bisect
import pickle
from multiprocessing import Pool, Manager
import numpy as np
import numba as nb
from numba import njit
import pandas as pd
import tqdm
import json
from multiprocessing import Process, Queue
# import cython
# from distutils.cor import setup
# from Cython.Build import cythonize
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter as gf
import math as mth
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
# import random
from functools import partial
# from matplotlib.cm import register_cmap, cmap_d
import illustris_python as il
import h5py
import argparse  # add options and arguments when running script in command line
import sys

# NOTE IF YOUR GENERATED TEXT FILES INCLUDE THE MASS CUT UNCOMMENT THE TXT FINE GENERATION

# Reading in zero redshift galaxies with dust correction
# dust_corrected_file = "/home/pat/Physics/illustris_python-master/tng_300_subhalo_dust_corrected/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5"
# dust_corrected_file = '/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5'
dust_corrected_file = '/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5'
with h5py.File(dust_corrected_file, "r") as partData:
    subhalo_mag_r_dust = np.asarray(partData["/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:, 2, 0],
                                    dtype='f4')

min_mag_r = min(subhalo_mag_r_dust)
max_mag_r = max(subhalo_mag_r_dust)
print("min and max of mag_r, dust corrected:", min_mag_r, max_mag_r)

# path = '/home/pat/Physics/illustris_python-master/tng_300'
# path = '/home/benjamin/Thesis/Illustris_TNG/tng_300'
path = '/home/benjamin/Thesis/Illustris_TNG/tng_300'
# Only care about r200, leave the mass in but cut the gas, first parse in the 'mask' to get rid of the Flag halos and subhalos
halo_fields = ['GroupPos', 'GroupMass', 'GroupMassType', 'Group_M_Crit200', 'Group_M_Crit500', 'Group_R_Crit200',
               'GroupNsubs', 'GroupLenType',
               'GroupLen']  # check how much of gas lies within R200, R500. See ratio of gas mass to total mass. Compare to EAGLE Toni's paper.
subhalo_fields = ['SubhaloFlag', 'SubhaloMass', 'SubhaloMassType', 'SubhaloPos', 'SubhaloStellarPhotometrics']
halos = il.groupcat.loadHalos(path, 99, fields=halo_fields)
subhalos = il.groupcat.loadSubhalos(path, 99, fields=subhalo_fields)

halo_coord_x = halos['GroupPos'][:, 2] / (0.6774 * 10 ** 3)
halo_coord_y = halos['GroupPos'][:, 1] / (0.6774 * 10 ** 3)
halo_coord_z = halos['GroupPos'][:, 0] / (0.6774 * 10 ** 3)
halo_rad_crit200 = halos['Group_R_Crit200'][:] / (0.6774 * 10 ** 3)
halo_mass = halos['GroupMass'] / 0.6774 * 10 ** 10  # cluster mass, includes all particle types
halo_mass_gas = halos['GroupMassType'][:, 0] / 0.6774 * 10 ** 10  # gas mass in cluster
halo_gal_count = halos['GroupNsubs'][:]
x = halo_coord_x
y = halo_coord_y
z = halo_coord_z

#np.savetxt("halo_positions.txt", np.transpose([x,y,z]),fmt="%s")


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

####### masks for groups/halos (clusters) ############################## (This is also the Flag exclusion)
mask_halo = np.ones(len(halo_gal_count),dtype=np.bool)
halo_gal_count_mask = mask_halo*(halo_gal_count > 0)
print('number of cluster positions after applying subhalo count filter',len(halo_coord_x[halo_gal_count_mask]))
print('halo(cluster) mass range after applying subhalo count filter:',min(halo_mass[halo_gal_count]),max(halo_mass[halo_gal_count]))
galaxy_mass_mask = mask*(subhalo_mass_stellar > 0.0) #(subhalo_mass_stellar <= 10**12)*(subhalo_mass_stellar >= 10**9) #mask for the galaxy stellar masses. Some seem unusual
galaxy_lum_r = 10.0**((subhalo_mag_r_dust - 4.42)/-2.5)
lum_massflag_only = galaxy_lum_r[Flag_mask*galaxy_mass_mask]
galaxy_lum_r = galaxy_lum_r[Flag_mask*Lum_mask]

###### Trying to make the gas mask #####
#mask_halo = np.ones(len(subhalo_mag_r),dtype=np.bool)

lum_avg_den = sum(galaxy_lum_r.flatten()) / 303.**3
print("Average luminosity density of whole volume, with only flagged bad galaxies excluded and lum filter applied:",lum_avg_den)
print("min and max of galaxy masses, no filtering:",min(subhalo_mass),max(subhalo_mass))
print("min and max of galaxy masses, only flag and lum filtering:",min(subhalo_mass[Lum_mask*Flag_mask]),max(subhalo_mass[Lum_mask*Flag_mask]))
print("min and max of galaxy stellar masses, only lum and flag filtering:", min(subhalo_mass_stellar[Lum_mask*Flag_mask]), max(subhalo_mass_stellar[Lum_mask*Flag_mask]))
print("min and max of galaxy stellar masses,filtering:", min(subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]), max(subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]))
print("number of subhaloes, no filtering at all:",len(subhalo_mass))
print("number of flagged subhaloes:",len(subhalo_mass) - len(subhalo_mass[Flag_mask]))
# print("number of galaxies only mass filtering:",np.shape(lum_massflag_only))
print("number of galaxies with only lumfiltering and flagged filter:",len(subhalo_mass[Flag_mask*Lum_mask]))
subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask*Flag_mask] # removed mass mask
subhalo_mass = subhalo_mass[Lum_mask*Flag_mask] # removed mass mask
subhalo_mass_stellar = subhalo_mass_stellar[Lum_mask*Flag_mask] # removed mass mask

np.savetxt("subhalo_mass_total_wmass.txt",subhalo_mass,fmt="%s")
np.savetxt("subhalo_stellar_mass_wmass.txt",subhalo_mass_stellar,fmt="%s")
np.savetxt("subhalo_r_magnitude_dust_wmass.txt",subhalo_mag_r_dust,fmt="%s")

###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600 #1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303. #Units of Mpc
LD_Overden = np.zeros([num_cells,num_cells,num_cells])
x = subhalo_coord[Lum_mask*Flag_mask,2] / (0.6774*10**3)
y = subhalo_coord[Lum_mask*Flag_mask,1] / (0.6774*10**3)
z = subhalo_coord[Lum_mask*Flag_mask,0] / (0.6774*10**3)

np.savetxt("subhalo_positions_wmass.txt", np.transpose([x,y,z]),fmt="%s")

np.savetxt('halo_positions_r200_masscut.txt',subhalo_gasin_r200,fmt='%s')


# The .txt files generated are filtered by luminosity and flags open the txt files to save run spce
# IMPORTANT NOTE, THESE TXT FILES HAVE THEIR MASS MASK FILTER REMOVED FROM THE ORIGINAL GENERATION CODE.
#subhalo_mass_total = np.loadtxt('subhalo_mass_total_wmass.txt', dtype=float)
#subhalo_positions = np.loadtxt(
#    'subhalo_positions_wmass.txt')  # This is a list of xyz corrds, having nested arrays may be wacky may need to define as special dtype
#subhalo_r_magnitude_dust = np.loadtxt('subhalo_r_magnitude_dust_wmass.txt', dtype=float)
#subhalo_stellar_mass = np.loadtxt('subhalo_stellar_mass_wmass.txt', dtype=float)

#print("number of galaxies, filtered:", len(subhalo_mass_total))

# Need to remove the gas from r200 in the WHIM
# the r200 virial radius

# Getting the WHIM information
WHIM_Data = pd.read_csv('dataset_WHIMgas__tng3001_z0__ratio400.csv')
WHIM_Data = np.asarray(WHIM_Data)

x_pos_WHIM = WHIM_Data[:, 1]  # x coordinate for WHIM in Mpc
y_pos_WHIM = WHIM_Data[:, 2]  # y ...
z_pos_WHIM = WHIM_Data[:, 3]  # z ...
mass_WHIM = WHIM_Data[:, 4]  # units in M_sun
den_WHIM = WHIM_Data[:, 5]  # units in M_sun * Mpc^-3
temp_WHIM = WHIM_Data[:, 8]  # units in Kelvin (K)
press_WHIM = WHIM_Data[:, 9]  # pressure in units of KeV*cm^-3

# boltz_const = 1.380649 * 10**-23
# boltz_const * temp_WHIM * (6.242*10^15) #converting Joules to keV

# repeat for all data indices
# if halo pos inside r200 and WHIM pos inside r200
# remove gas
halo_r = 0
halo_theata = 0
halo_phi = 0
WHIM_r = 0
WHIM_theata = 0
WHIM_phi = 0
matching_locations = {}

# trying from the perspective of the halo, going to build a cube that contains the r200 sphere,
# creating sorted dictionary of positions where the key is the index and the values is the xyz coord
# (Including density, pressure, temp, pressure and mass, and r200 value can tie the index to the original thing)
# first try to include them in the dictionary
#unsorted_halo_ix = ([(index, hx) for index, hx in zip([i for i in range(0 , len(halo_coord_x))], halo_coord_x)])
#unsorted_WHIM_ix = ([(index, wx) for index, wx in zip([i for i in range(0,len(x_pos_WHIM))],x_pos_WHIM)]) # np arrays dont have key attribute needed for lambda sorting

# sorting the xyz values by the x coord, based on what I have read this speeds things up significantly
# sorting by the x-values, the way the lists were created and method of sorting preserve the xyz relation by keeping track of index
#unsorted_WHIM_ix.sort(key=lambda entry: entry[1])
#unsorted_halo_ix.sort(key=lambda entry: entry[1])
#halo_ix = np.array(unsorted_halo_ix)
#WHIM_ix = np.array(unsorted_WHIM_ix)
# this does the same as a function same as
# def get_x_value(entry): return entry[1]
# list.sort(key=get_x_value)

# Turning the sorted xyz positions into dictionaries
#WHIM_position={}
#halo_position={}
#for i in tqdm.tqdm(range(len(x_pos_WHIM))):
#    WHIM_position[i] = WHIM_ix[i]
#    halo_position[i] = halo_ix[i]
WHIM_position = sorted(list(zip([i for i in range(0, len(x_pos_WHIM))], x_pos_WHIM)), key=lambda x: x[1])
#saving sorted pos as file to increase run speed
#print(len(zip([i for i in range(0, len(x_pos_WHIM))],x_pos_WHIM)))
print(len(list(zip([i for i in range(0, len(x_pos_WHIM))], x_pos_WHIM))))

# creating the loop that will check the points inside a halos r200
# cubeV = 2*r200 * 2*r200 * 2*r200
# sphereV = 4/3 pi r200**3
matching_locations = []
#whilly=np.array([np.array((x, y_pos_WHIM[i], z_pos_WHIM[i])) for i, x in WHIM_position])

def compute(halo, hr, x_sorted, y_pos_WHIM, z_pos_WHIM, den_WHIM, temp_WHIM, press_WHIM, mass_WHIM):
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
        z_pos_WHIM = WHIM_Data[:, 1]  # x coordinate for WHIM in Mpc
        y_pos_WHIM = WHIM_Data[:, 2]  # y ...
        mass_WHIM = WHIM_Data[:, 4]  # units in M_sun
        den_WHIM = WHIM_Data[:, 5]  # units in M_sun * Mpc^-3
        temp_WHIM = WHIM_Data[:, 8]  # units in Kelvin (K)
        press_WHIM = WHIM_Data[:, 9]  # pressure in units of KeV*cm^-3
        halos=np.loadtxt('halo_positions.txt')

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
    main()

















'''for j in tqdm.tqdm(range(len(halo_coord_x))):
    #This halos cube and sphere
    sphere_center_x = halo_coord_x[j]
    sphere_center_y = halo_coord_y[j]
    sphere_center_z = halo_coord_z[j]
    points_in_sphere = []
    # defining the start of the search, left side of cube
    # This will return an index, where the x value of the whim falls within 1 radius from the "left side"
    start = bisect.bisect_left(WHIM_position, sphere_center_x - halo_rad_crit200[j], key=lambda x: x[1])

    # looping while the x position of the whim is within the halo
    while start < len(WHIM_position) and WHIM_position[start][1] <= sphere_center_x + halo_rad_crit200[j]:
        WHIM_x = WHIM_position[start][1]
        WHIM_y = y_pos_WHIM[WHIM_position[start][0]] # WHIM_position[start][0] is the index of the starting x pos
        WHIM_z = z_pos_WHIM[WHIM_position[start][0]]
        distance = ((sphere_center_x - WHIM_x) ** 2 + (sphere_center_y-WHIM_y) ** 2 + (sphere_center_z - WHIM_z) ** 2) ** .5
        if distance <= halo_rad_crit200[j]:
            den_WHIM[WHIM_position[start][0]] = 0
            temp_WHIM[WHIM_position[start][0]] = 0
            press_WHIM[WHIM_position[start][0]] = 0
            mass_WHIM[WHIM_position[start][0]] = 0
            points_in_sphere.append(start)
            print('We got a hit')
        start += 1'''

'''distance = np.sum((np.array((sphere_center_x, sphere_center_y, sphere_center_z)) - whilly[start:]) ** 2) ** .5
       dist_comparison = distance <= halo_rad_crit200
       if dist_comparison.any(): 
           #den_WHIM[i] = 0
           #temp_WHIM[i] = 0
           #press_WHIM[i] = 0
           #mass_WHIM[i] = 0
           # matching_locations.append([i, [dist_comparison.nonzero()]])
           matching_locations = dist_comparison.non_zero()'''

#print('Number of WHIM particles in halos', len(points_in_sphere))


volume = mass_WHIM / den_WHIM  # Originally commented out...

# boltz_const = 1.380649 * 10**-23
# boltz_const * temp_WHIM * (6.242*10^15) #converting Joules to keV
##########t plotting distribution of WHIM ##################
plt.hist(np.log10(den_WHIM), np.linspace(min(np.log10(den_WHIM)), max(np.log10(den_WHIM)), 50), density=False)
plt.xlabel(r'$\log(\rho)$', fontsize=15)
plt.savefig('WHIMden_dist.pdf')
plt.close()

plt.hist(np.log10(volume), np.linspace(min(np.log10(volume)), max(np.log10(volume)), 50), density=False)
plt.xlabel(r'$\log(V)$', fontsize=15)
plt.savefig('volume_dist.pdf')
plt.close()

plt.hist(np.log10(mass_WHIM), np.linspace(min(np.log10(mass_WHIM)), max(np.log10(mass_WHIM)), 50), density=False)
plt.xlabel(r'$\log(M)$', fontsize=15)
plt.savefig('WHIMmass_dist.pdf')
plt.close()

###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600  # 1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303.  # Units of Mpc
# cell_len = box_len/num_cells
LD_Overden = np.zeros([num_cells, num_cells, num_cells])

x_bins = np.linspace(0.0, 303.0, num_cells + 1)
y_bins = np.linspace(0.0, 303.0, num_cells + 1)
z_bins = np.linspace(0.0, 303.0, num_cells + 1)

x_idx = np.digitize(x_pos_WHIM, x_bins)
y_idx = np.digitize(y_pos_WHIM, y_bins)
z_idx = np.digitize(z_pos_WHIM, z_bins)

mean_baryon_den = 0.618 * 10 ** 10  # mean baryon density of universe in units of M_sun * Mpc^-3
WHIM_Den_avg = sum(mass_WHIM) * 400. / 303. ** 3  # WHIM gas sampled selecting 1 out of 400.
WHIM_Den_avg_2 = sum(mass_WHIM) / sum(volume)
print("WHIM density average of whole box:", WHIM_Den_avg, WHIM_Den_avg_2)
print("volume range and cell volume:", min(volume), max(volume), (box_len / num_cells) ** 3.)
# quit()
WHIM_Overden = np.zeros([num_cells, num_cells, num_cells])
Temp_grid = np.zeros([num_cells, num_cells, num_cells])  # mass weighted mean temperature of each cell
Mass_grid = np.zeros([num_cells, num_cells, num_cells])
# Volume_grid = np.zeros([num_cells,num_cells,num_cells])
Pressure_grid = np.zeros([num_cells, num_cells, num_cells])


@njit
def WHIM_mass_func(box_len, num_cells, x, y, z, Mass_grid):
    cell_len = box_len / num_cells
    for n in range(len(x)):
        Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + \
                                                              mass_WHIM[n]
    return Mass_grid


WHIM_mass_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Mass_grid)


@njit
def WHIM_Pressure_func(box_len, num_cells, x, y, z, Pressure_grid):
    cell_len = box_len / num_cells
    for n in range(len(x)):
        Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Pressure_grid[
                                                                      z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + \
                                                                  mass_WHIM[n] * press_WHIM[n] / Mass_grid[
                                                                      z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
    return Pressure_grid  # Mass weighted mean pressure for each cell


WHIM_Pressure_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Pressure_grid)


@njit
def WHIM_Temp_func(box_len, num_cells, x, y, z, Temp_grid):
    cell_len = box_len / num_cells
    for n in range(len(x)):
        Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + \
                                                              mass_WHIM[n] * temp_WHIM[n] / Mass_grid[
                                                                  z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
    return Temp_grid  # Mass weighted mean temperature for each cell


@njit
def WHIM_OverDen_func(box_len, num_cells, x, y, z, WHIM_Overden):
    cell_len = box_len / num_cells
    for n in range(len(x)):
        WHIM_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] / (
                box_len / num_cells) ** 3 * 400. / mean_baryon_den  # Volume_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] /mean_baryon_den  #matching luminosity values with binned coordinates
    return WHIM_Overden


print('length of WHIM temp dataset:', len(temp_WHIM), len(mass_WHIM), len(den_WHIM))
# print('range of temperature:',min(temp_WHIM),max(temp_WHIM))
# print('distance range in x coordinate:',min(x_pos_WHIM),max(x_pos_WHIM))
# quit()
WHIM_Temp_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Temp_grid)
WHIM_OverDen_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, WHIM_Overden)

print('WHIM density range for cells:', min(WHIM_Overden.flatten()), max(WHIM_Overden.flatten()))
print('WHIM pressure range for cells:', min(Pressure_grid.flatten()), max(Pressure_grid.flatten()))
print('WHIM Temp range for cells:', min(Temp_grid.flatten()), max(Temp_grid.flatten()))
# print('min and max volume grid:',min(Volume_grid.flatten()),max(Volume_grid.flatten()))
# print('sum of the gas cell volume',sum(Volume_grid.flatten()))
# print('sum of gas  cell volume times 400',sum(Volume_grid.flatten()) * 400)
print("number of bins with WHIM (den values):", np.size(np.nonzero(WHIM_Overden.flatten())))
vol_WHIM = np.size(np.nonzero(WHIM_Overden.flatten())) / len(WHIM_Overden.flatten())  # * 303.**3
print("fraction of volume occupied by WHIM", vol_WHIM)
# some of the non-selected WHIM data may be in bins with no WHIM. Also try binning according to which gridpoit is closer to positions.
# quit()
############## Plotting ##########################################
###############  Filaments #######################################

filefils = 'filament_coords_and_lengths.npy'

# Read filaments
seg_z, seg_y, seg_x, lengths = np.load(filefils, allow_pickle=True)

print('Number of filaments:', len(seg_x))

for i in range(20):
    z_in = 15 * i  # in Mpc
    z_fin = 15 * (i + 1)  # in Mpc
    cmap_LD = mpl.cm.get_cmap('Oranges')
    cmap_WHIM = mpl.cm.get_cmap('Purples')
    fig, ax = plt.subplots(1, figsize=(8, 8))

    # zslice = (coord[:,0] > z_in) * (coord[:,0] < z_fin) #right now its the whole z axis
    # zslice = (subhalo_coord[:,0] > z_in) * (subhalo_coord[:,0] < z_fin) #convert to Mpc?
    zslice = (z > z_in) * (z < z_fin)  # redo zslice with conversion?
    gridIndexMin = int((num_cells * z_in) / box_len)
    gridIndexMax = int((num_cells * z_fin) / box_len)
    imageSlice = np.mean(LD_Overden[gridIndexMin:gridIndexMax, :, :], axis=0)
    # imageVolumeSlice=np.mean(Volume_grid[gridIndexMin:gridIndexMax,:,:],axis=0)
    imageWHIMSlice = np.mean(WHIM_Overden[gridIndexMin:gridIndexMax, :, :], axis=0)
    vmin = 1e-6  # min(imageSlice.flatten())
    vmax = 1e3  # max(imageSlice.flatten())
    vmin_WHIM = min(imageWHIMSlice.flatten())
    vmax_WHIM = max(imageWHIMSlice.flatten())
    vmin_nonzero = imageSlice.flatten()
    vmin_nonzero = vmin_nonzero[np.nonzero(vmin_nonzero)]
    print("min and max LD for image slice:", vmin, vmax)
    #print("min nonzero LD for image slice:", min(vmin_nonzero))
    ####### plotting filaments. Filaments given by Daniela ##########
    for j in range(len(seg_x)):
        if (np.mean(seg_z[j]) > z_in) & (np.mean(seg_z[j]) <= z_fin):
            fil_scat = ax.plot(seg_x[j], seg_y[j], color='k')
    gal_scat = ax.scatter(x[zslice], y[zslice], marker='o', s=1.0, color='red', alpha=0.7, label='galaxies')
    # LD_plot = ax.imshow(imageSlice, norm=LogNorm(vmin=1e-5, vmax=500.), extent=[0.0,303.0,0.0,303.0], aspect='auto', origin="lower", cmap=cmap_LD) #vmax=2000, vmin=1e-3, just like for EAGLE, only visual purposes
    # WHIMden_plot = ax.imshow(imageWHIMSlice, cmap=cmap_WHIM, extent= [0.,303.,0.,303.0], origin='lower',alpha=0.6, label="Volume")
    WHIMden_plot = ax.imshow(imageWHIMSlice, cmap=cmap_WHIM, norm=LogNorm(vmin=vmin, vmax=vmax),
                             extent=[0., 303., 0., 303.0], origin='lower', alpha=0.6,
                             label="WHIM")  # vmin = 1e-06, vmax = 1e05
    plt.xlabel('X [Mpc]', fontsize=10)
    plt.ylabel('Y [Mpc]', fontsize=10)
    ax.legend(loc=1, fontsize=12)
    divider = make_axes_locatable(ax)
    # cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
    # fig.add_axes(cax)
    # cbar = fig.colorbar(LD_plot, cax=cax, orientation = "horizontal", format=ticker.LogFormatter())
    # cbar.set_label(r'$\delta_{LD}$')#Luminosity Overdensity') #($10^{10} L_{\odot} Mpc^{-3}$)')
    # cbar.ax.tick_params(labelsize=10, width=0.7, size=8)
    # cbar.solids.set_edgecolor("face")
    # cbar.ax.xaxis.set_label_position('bottom')
    ####### colorbar for WHIM #########################
    cax2 = divider.new_vertical(size='5%', pad=0.6)
    fig.add_axes(cax2)
    cbar2 = fig.colorbar(WHIMden_plot, cax=cax2, orientation="horizontal")  # , format=ticker.LogFormatter())
    cbar2.set_label(r'$\delta_{\rho}$')  # WHIM Overdensity') #($10^{10} M_{\odot} Mpc^{-3}$)')
    cbar2.ax.tick_params(labelsize=10, width=0.7, size=8)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.xaxis.set_label_position('bottom')

    # plt.savefig('Volume_grid_slice{i}.pdf'.format(i=i))
    plt.savefig('WHIM_density_filaments_galaxies_slice{i}.pdf'.format(i=i))
    # plt.savefig('WHIM_density_slice{i}.pdf'.format(i=i))
    # plt.savefig('galaxy_positions_LD_slice{i}.pdf'.format(i=i))
    # plt.savefig('galaxies_and_fils{i}.pdf'.format(i=i))
    # plt.savefig('Select_galaxy_positions_LD_slice{i}.pdf'.format(i=i))
    print('num of galaxies in slice:', np.size(x[zslice]))
    plt.show()
    plt.close()
