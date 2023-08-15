import numpy as np
import tqdm
import scipy
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.cm import register_cmap, cmap_d
import h5py
from matplotlib import ticker
from matplotlib.colors import LogNorm
from numba import njit
from scipy.ndimage import gaussian_filter as gf
import illustris_python as il
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# plt.ion()

filefils = 'filament_coords_and_lengths.npy'

# Read filaments
seg_z, seg_y, seg_x, lengths = np.load(filefils, allow_pickle=True)

print('Number of filaments and value at seg_x[0][1] and seg_y[0]:', len(seg_x[0]), seg_x[0][1], seg_y[0])

# fil_x = np.array([x for filament in seg_x for x in filament])
# fil_y = np.array([y for filament in seg_y for y in filament])
# fil_z = np.array([z for filament in seg_z for z in filament])
# sorted based on x
# fil_cords = sorted([(x, y, z) for x, y, z in zip(fil_x, fil_y, fil_z)])
# fil_cords = np.array(fil_cords)
# fil_r = []
# for i in range(len(seg_x)):
#    r = (seg_x[i] ** 2 + seg_y[i] ** 2 + seg_z[i] ** 2) ** .5
#    fil_r.append(r)
box_len = 303.  # Units of Mpc
num_cells = 300 #600  # 1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
# cell_len = box_len/num_cells
fil_grid = np.zeros([num_cells, num_cells, num_cells], dtype=np.bool)
x_bins = np.linspace(0.0, 303.0, num_cells + 1)
y_bins = np.linspace(0.0, 303.0, num_cells + 1)
z_bins = np.linspace(0.0, 303.0, num_cells + 1)

seg_tot_points = 0
for i in range(len(seg_x)):
    seg_tot_points = seg_tot_points + len(seg_x[i])

seg_x_new = np.array(seg_x)
seg_y_new = np.array(seg_y)
seg_z_new = np.array(seg_z)
print('length of new seg_x[0]', len(seg_x_new[0]))
########################### making a fillament grid #############################################

# Adding the points inbetween fillament locations to "smooth" the fillaments

if 1 == 1:  # added to avoid fixing tab space of defined function
    steps = 20
    for n in range(len(seg_x_new)):
        for m in range(len(seg_x_new[n])):
            if m == len(seg_x_new[n]) - 1:
                continue
            x_step = (seg_x_new[n][m + 1] - seg_x_new[n][m]) / steps
            y_step = (seg_y_new[n][m + 1] - seg_y_new[n][m]) / steps
            z_step = (seg_z_new[n][m + 1] - seg_z_new[n][m]) / steps
            seg_x_add = np.linspace(seg_x_new[n][m] + x_step, seg_x_new[n][m + 1] - x_step, steps - 1)
            seg_y_add = np.linspace(seg_y_new[n][m] + y_step, seg_y_new[n][m + 1] - y_step, steps - 1)
            seg_z_add = np.linspace(seg_z_new[n][m] + z_step, seg_z_new[n][m + 1] - z_step, steps - 1)
            seg_x_new[n] = np.insert(seg_x_new[n], m, seg_x_add)
            seg_y_new[n] = np.insert(seg_y_new[n], m, seg_y_add)
            seg_z_new[n] = np.insert(seg_z_new[n], m, seg_z_add)

print('length of new seg_x[0]', len(seg_x_new[0]))

# digitizing and making this binned
# seg_x_bin = seg_y_bin = seg_z_bin = []
seg_x_bin = []
seg_y_bin = []
seg_z_bin = []
for i in range(len(seg_x_new)):
    for j in range(len(seg_x_new[i])):
        seg_x_bin.append(seg_x_new[i][j])
        seg_y_bin.append(seg_y_new[i][j])
        seg_z_bin.append(seg_z_new[i][j])

x_idx = np.digitize(seg_x_bin, x_bins)
y_idx = np.digitize(seg_y_bin, y_bins)
z_idx = np.digitize(seg_z_bin, z_bins)

print('length of new segment arrays', len(seg_x_bin), len(seg_y_bin), len(seg_z_bin))
print('length of indices', len(z_idx))

# Patricks Fil grid construction
# @njit
def fillament_func(box_len, num_cells, x_idx, y_idx, z_idx, fil_grid):
    fil_rad = 1.0  # 1MPC radius
    for n in range(len(seg_x_new)):
        k_max = int(fil_rad / (box_len / num_cells))
        for k in range(-1 * k_max, k_max + 1):
            for l in range(-1 * k_max, k_max + 1):
                for m in range(-1 * k_max, k_max + 1):
                    if z_idx[n] - 1 + k > num_cells - 1 or z_idx[n] - 1 + k < 0 or y_idx[n] - 1 + l > num_cells - 1 or \
                            y_idx[n] - 1 + l < 0 or x_idx[n] - 1 + m > num_cells - 1 or x_idx[n] - 1 + m < 0:
                        continue
                    fil_grid[z_idx[n] - 1 + k, y_idx[n] - 1 + l, x_idx[n] - 1 + m] = True
    return fil_grid


fillament_func(box_len, num_cells, x_idx, y_idx, z_idx, fil_grid)

print('max value of fil grid', np.max(fil_grid.flatten()))
print('The length of the fillament grid, no flat and flat.', len(fil_grid), len(fil_grid.flatten()))

############################ START OF HALO INFORMATION ###############################################3
dust_corrected_file = '/home/benjamin/Thesis/Illustris_TNG/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5'

with h5py.File(dust_corrected_file, "r") as partData:subhalo_mag_r_dust = np.asarray(partData["/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:, 2, 0], dtype='f4')

#min_mag_r = min(subhalo_mag_r_dust)
#max_mag_r = max(subhalo_mag_r_dust)
#print("min and max of mag_r, dust corrected:", min_mag_r, max_mag_r)
path = '/home/benjamin/Thesis/Illustris_TNG/tng_300'

halo_fields = ['GroupPos', 'GroupMass', 'Group_R_Crit200', 'GroupNsubs', 'GroupLenType']  # check how much of gas lies within R200, R500. See ratio of gas mass to total mass. Compare to EAGLE Toni's paper.
subhalo_fields = ['SubhaloFlag', 'SubhaloMass', 'SubhaloMassType', 'SubhaloPos', 'SubhaloStellarPhotometrics', 'SubhaloGrNr']
halos = il.groupcat.loadHalos(path, 99, fields=halo_fields)
subhalos = il.groupcat.loadSubhalos(path, 99, fields=subhalo_fields)

halo_coord_x = halos['GroupPos'][:, 2] / (0.6774 * 10 ** 3)
halo_coord_y = halos['GroupPos'][:, 1] / (0.6774 * 10 ** 3)
halo_coord_z = halos['GroupPos'][:, 0] / (0.6774 * 10 ** 3)
halo_rad_crit200 = halos['Group_R_Crit200'][:] / (0.6774 * 10 ** 3)
halo_mass = halos['GroupMass'] / 0.6774 * 10 ** 10  # cluster mass, includes all particle types
#halo_mass_gas = halos['GroupMassType'][:, 0] / 0.6774 * 10 ** 10  # gas mass in cluster
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
# subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask*Flag_mask]

##### Halo Masks ########
mask_halo = np.ones(len(halo_gal_count), dtype=np.bool)
halo_gal_count_mask = mask_halo * (halo_gal_count > 0)
galaxy_mass_mask = mask_halo #* (halo_mass > 10 ** 14)  # applying 10^14 mass cut

print('length of halo cord array pre mask:', len(halo_coord_x), len(halo_coord_y), len(halo_coord_z))

halo_coord_x = halo_coord_x[halo_gal_count_mask * galaxy_mass_mask]
halo_coord_y = halo_coord_y[halo_gal_count_mask * galaxy_mass_mask]
halo_coord_z = halo_coord_z[halo_gal_count_mask * galaxy_mass_mask]
#halo_rad_mass_filtered = halo_rad_crit200[halo_gal_count_mask * galaxy_mass_mask]
halo_rad_mass_filtered = halo_rad_crit200
print('Length of halo cord array x,y,z after masks', len(halo_coord_x), len(halo_coord_y), len(halo_coord_z))

# mag_sun_r = 4.42 #absolute magnitude of sun in r-band.
# lum_r = 10.0**((subhalo_mag_r_dust - mag_sun_r)/-2.5)

############################################ LUMINOSITY R200 FILTERING ##############################################
# Needs to be based off of lum r
print('The length of subhalo_mag_r pre kd tree', len(subhalo_mag_r_dust))
# points = [(x, y, z) for x, y, z in zip(halo_coord_x, halo_coord_y, halo_coord_z)]
points = subhalo_coord  # having the subhalo cords be the tree
tree = scipy.spatial.KDTree(points)
print("tree loaded")
halos = [(x, y, z) for x, y, z in zip(halo_coord_x, halo_coord_y, halo_coord_z)]
# #######################################The filtering interior to r200#######################################33

# total = 0
# halo_in_halo_index = []
# for i in tqdm.trange(len(halos)):
#     points_in_radius = tree.query_ball_point(halos[i], halo_rad_mass_filtered[i], workers=-1)  # setting workers to -1 maximizes parallelization
#     total += len(points_in_radius)  # to keep track of the total number of points that got zeroed out
#     for point in points_in_radius:
#         subhalo_mag_r_dust[point] = 0  # Changed from lum r so that we can exclude before the mask and before the lum_r creation
#         halo_in_halo_index.append([point, i])
#     points_in_radius = np.array(points_in_radius)
#     for j in range(len(points_in_radius)):
#         halo_in_halo_index.append([points_in_radius[j], i]) # array that has first entry as every index that the i'th subhalo is interior to

######################################33 filtering for exterior to r200 #############################################
'''
total = 0
whim_in_halo_index = []
indices=[]
for i in tqdm.trange(len(halos)):
    points_in_radius = tree.query_ball_point(halos[i], halo_rad_crit200[i], workers=-1)  # setting workers to -1 maximizes parallelization
    #points_in_radius_set = set(points_in_radius)
    total += len(points_in_radius)  # to keep track of the total number of points that got zeroed out (although this could double count, but there are no dual halo inhabitants for r200)
    indices.extend(points_in_radius)

print('Total number of Whim in halos:', total)
indices_set = set(indices)
for i in tqdm.tqdm(range(len(subhalo_mag_r_dust))): # Zero exterior points
    if i in indices_set:
        continue
    else:
        subhalo_mag_r_dust[i] = 0
'''
#print('Number of lum values excluded', total)
print('The length of subhalo_mag_r post kd tree', len(subhalo_mag_r_dust))
# lum_avg_den = sum(lum_r.flatten()) / 303.**3

# Applying the mask after filtering to maintain the length
subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask * Flag_mask]
print('The length of subhalo_mag_r post kd and masks (lum + flag)', len(subhalo_mag_r_dust))
mag_sun_r = 4.42  # absolute magnitude of sun in r-band.
lum_r = 10.0 ** ((subhalo_mag_r_dust - mag_sun_r) / -2.5)

#lum_r = lum_r[Flag_mask*Lum_mask]

lum_avg_den = sum(lum_r.flatten()) / 303. ** 3

###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 300  # 1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303.  # Units of Mpc
cell_len = box_len / num_cells
LD_Overden = np.zeros([num_cells, num_cells, num_cells])
x = subhalo_coord[Lum_mask * Flag_mask, 2]  #/ (0.6774*10**3)
y = subhalo_coord[Lum_mask * Flag_mask, 1]  #/ (0.6774*10**3)
z = subhalo_coord[Lum_mask * Flag_mask, 0]  #/ (0.6774*10**3)

del subhalo_coord

x_bins = np.linspace(0.0, 303.0, num_cells + 1)
y_bins = np.linspace(0.0, 303.0, num_cells + 1)
z_bins = np.linspace(0.0, 303.0, num_cells + 1)

x_idx = np.digitize(x, x_bins)
y_idx = np.digitize(y, y_bins)
z_idx = np.digitize(z, z_bins)


@njit
def Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden):
    LD_smooth_param = 1.2 / cell_len  # smoothing 1.2 Mpc converted to Mpc/cell_len (dimensionless), Units of Mpc
    for n in range(len(x)):
        LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + \
                                                               lum_r[n] / lum_avg_den / (
                                                                           box_len / num_cells) ** 3  # matching luminosity values with binned coordinates
    return LD_Overden


Lum_OverDen_func(box_len, num_cells, x, y, z, LD_Overden)  # running LD function

print("number of galaxies for smoothing:", np.size(x))
print("dimensions of LD grid", np.shape(LD_Overden))
print("number of bins with galaxies (lum values):", np.size(np.nonzero(LD_Overden.flatten())))

########### smoothing using Gaussian kernel ####################################
smooth_param = 1.2 / (box_len/cell_len) # 2 #Units of Mpc
LD_Overden = gf(LD_Overden, sigma=0.6 * smooth_param)

########################## WHIM OVERDENSITY GRID ################################################
####################### Reading in WHIM dataset from Daniela ###################

WHIM_Data = pd.read_csv('dataset_WHIMgas__tng3001_z0__ratio400.csv')
WHIM_Data = np.asarray(WHIM_Data)

z_pos_WHIM = WHIM_Data[:, 1]  # x coordinate for WHIM in Mpc
y_pos_WHIM = WHIM_Data[:, 2]  # y ...
x_pos_WHIM = WHIM_Data[:, 3]  # z ...
mass_WHIM = WHIM_Data[:, 4]  # units in M_sun
den_WHIM = WHIM_Data[:, 5]  # units in M_sun * Mpc^-3
temp_WHIM = WHIM_Data[:, 8]  # units in Kelvin (K)
press_WHIM = WHIM_Data[:, 9]  # pressure in units of KeV*cm^-3
volume = mass_WHIM / den_WHIM  # Originally commented out...

# boltz_const = 1.380649 * 10**-23
# boltz_const * temp_WHIM * (6.242*10^15) #converting Joules to keV

x_idx = np.digitize(x_pos_WHIM, x_bins)
y_idx = np.digitize(y_pos_WHIM, y_bins)
z_idx = np.digitize(z_pos_WHIM, z_bins)

mean_baryon_den = 0.618 * 10 ** 10  # mean baryon density of universe in units of M_sun * Mpc^-3
WHIM_Den_avg = sum(mass_WHIM) * 400. / 303. ** 3  # WHIM gas sampled selecting 1 out of 400.
WHIM_Den_avg_2 = sum(mass_WHIM) / sum(volume)

##################################### WHIM R200 Filtering ########################################3

print("loading tree...")
halos = [(x, y, z) for x, y, z in zip(halo_coord_x, halo_coord_y, halo_coord_z)]

# scipy.spatial.KDTree.query_ball_point(halos, halo_rs, p=2.0, eps=0, workers=1, return_sorted=None, return_length=False)
points = [(x, y, z) for x, y, z in zip(x_pos_WHIM, y_pos_WHIM, z_pos_WHIM)]
tree = scipy.spatial.KDTree(points)
print("tree loaded")

############################## The filtering for interior to r200##########################################

total = 0
whim_in_halo_index = []
for i in tqdm.trange(len(halos)):
    points_in_radius = tree.query_ball_point(halos[i], halo_rad_mass_filtered[i], workers=-1)  # setting workers to -1 maximizes parallelization
    total += len(points_in_radius)  # to keep track of the total number of points that got zeroed out (although this could double count, but there are no dual halo inhabitants for r200)
    for point in points_in_radius:
        den_WHIM[point] = 0
        mass_WHIM[point] = 0
        whim_in_halo_index.append([point, i])
    points_in_radius = np.array(points_in_radius)
    for j in range(len(points_in_radius)):
        whim_in_halo_index.append([points_in_radius[j], i])

############################################ Filtering for extioror r200 ######################################3
'''
total = 0
whim_in_halo_index = []
indices=[]
for i in tqdm.trange(len(halos)):
    points_in_radius = tree.query_ball_point(halos[i], halo_rad_crit200[i], workers=-1)  # setting workers to -1 maximizes parallelization
    #points_in_radius_set = set(points_in_radius)
    total += len(points_in_radius)  # to keep track of the total number of points that got zeroed out (although this could double count, but there are no dual halo inhabitants for r200)
    indices.extend(points_in_radius)

print('Total number of Whim in halos:', total)
indices_set = set(indices)
for i in tqdm.tqdm(range(len(den_WHIM))): # Zero exterior points
    if i in indices_set:
        continue
    else:
        den_WHIM[i] = 0
        mass_WHIM[i] = 0
'''

print('I climbed the tree (hopefully as intended)')
print('Total number of Whim in halos set to zero:', total)

del (tree)

WHIM_Overden = np.zeros([num_cells, num_cells, num_cells])
Mass_grid = np.zeros([num_cells, num_cells, num_cells])


@njit
def WHIM_mass_func(box_len, num_cells, x, y, z, Mass_grid):
    cell_len = box_len / num_cells
    for n in range(len(x)):
        Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + \
                                                              mass_WHIM[n]
    return Mass_grid


WHIM_mass_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, Mass_grid)


@njit
def WHIM_OverDen_func(box_len, num_cells, x, y, z, WHIM_Overden):
    cell_len = box_len / num_cells
    for n in range(len(x)):
        WHIM_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] / (
                    box_len / num_cells) ** 3 / mean_baryon_den * 400.  # Volume_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] /mean_baryon_den  #matching luminosity values with binned coordinates
    return WHIM_Overden


WHIM_OverDen_func(box_len, num_cells, x_pos_WHIM, y_pos_WHIM, z_pos_WHIM, WHIM_Overden)

print(WHIM_Overden.shape)
print(LD_Overden.shape)
print('The max and min values of WHIM', max(WHIM_Overden.flatten()), min(WHIM_Overden.flatten()))
print('The max and min values of Lum', max(LD_Overden.flatten()), min(LD_Overden.flatten()))
print('length of whim and lum before applying the fillaments', len(WHIM_Overden.flatten()), len(LD_Overden.flatten()))
# WHIM_scat = WHIM_Overden * fil_grid
# LD_scat = LD_Overden * fil_grid
WHIM_scat = WHIM_Overden[fil_grid].flatten()
LD_scat = LD_Overden[fil_grid].flatten()

WHIM_scat = WHIM_scat.flatten()
LD_scat = LD_scat.flatten()
print('length of the whim and ld after the fillaments applied',len(WHIM_scat), len(LD_scat))
print('The max and min values of WHIM after fil_grid', max(WHIM_scat.flatten()), min(WHIM_scat.flatten()))
print('The max and min values of Lum after fil_grid', max(LD_scat.flatten()), min(LD_scat.flatten()))

# Removing zeros so the log10 function will work
LD_scat = LD_scat[np.nonzero(WHIM_scat)]
WHIM_scat = WHIM_scat[np.nonzero(WHIM_scat)]
WHIM_scat = WHIM_scat[np.nonzero(LD_scat)]
LD_scat = LD_scat[np.nonzero(LD_scat)]
print('The max and min values of WHIM after fil_grid', max(WHIM_scat.flatten()), min(WHIM_scat.flatten()))
print('The max and min values of Lum after fil_grid', max(LD_scat.flatten()), min(LD_scat.flatten()))
logden = np.log10(WHIM_scat)
ldlog = np.log10(LD_scat)

print(min(logden), max(logden))
print(min(ldlog), max(ldlog))
print('Length of logden and ldlog: ', len(logden), len(ldlog))

# np.save('filtered3xr200_log_luminosity.npy', ldlog, True)
# np.save('filtered3xr200_WHIM_Density.npy', logden, True)


######################################WORKING##########################################################################

fig, ax = plt.subplots(1, figsize=(8, 8))
nbins = 40
divider = make_axes_locatable(ax)
# cax = divider.new_vertical(size='5%', pad = .6, pack_start=True)
# fig.add_axes(cax)
bins = [np.linspace(min(ldlog), max(ldlog), nbins + 1), np.linspace(min(logden), max(logden), nbins + 1)]
h, xedge, yedge = np.histogram2d(ldlog, logden, bins=bins)
h = h.transpose()

cmap = plt.get_cmap('Blues')
vmin = min(h.flatten()) + 1  # added a one so I can make the colormap logarithmic.
vmax = max(h.flatten())
X, Y = np.meshgrid(xedge, yedge)  # X_dim and Y_dim same as xedge and yedge 41,41
im = plt.pcolormesh(X, Y, h, cmap=cmap, edgecolors='black', norm=LogNorm(vmin=vmin, vmax=vmax), linewidth=0.3)

# values,  pos_count = np.unique(h, return_counts = True)
# print(values)
# print(pos_count)
# h = h[values]
par, cov = np.polyfit(h[0], h[1], deg=1, cov=True)
plt.plot(ldlog, ldlog * par[0] + par[1], color='red', label='Polyfit')
print('The fit parameters', par[0], par[1])

plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
plt.title('Plot for 1.2 smoothing factor excluding exterior to r200')
# ticks = np.linspace(min(ldlog), max(ldlog), 5)
cbar = plt.colorbar(mappable=im)
cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())
plt.ylim(min(yedge), max(yedge))
# cbar = fig.colorbar(im, cax = cax, orientation = 'horizontal', format = ticker.LogFormatter())
# cbar.ax.tick_params(labelsize=10, width = .7, size = 8)
# cbar.set_label('lumden')
# cbar.solids.set_edgecolor('face')
# cbar.ax.xaxis.set_label_position('bottom')
plt.legend()
plt.show()
###################################################################################################################
############################## Making a weighted fit based on counts ##############################################
counts = []
h_count = h.tolist()
for i in range(len(h_count)):
    counts.append(h_count.count(h_count[i]))
cmap = plt.get_cmap('Blues')
vmin = min(h.flatten()) + 1  # added a one so I can make the colormap logarithmic.
vmax = max(h.flatten())
X, Y = np.meshgrid(xedge, yedge)  # X_dim and Y_dim same as xedge and yedge 41,41
im = plt.pcolormesh(X, Y, h, cmap=cmap, edgecolors='black', norm=LogNorm(vmin=vmin, vmax=vmax), linewidth=0.3)

par, cov = np.polyfit(h[0], h[1], deg=1, cov=True)
plt.plot(ldlog, ldlog * par[0] + par[1], color='red', label='Polyfit')
print('The fit parameters', par[0], par[1])

par, cov = np.polyfit(h[0], h[1], deg = 1, cov = True, w = counts)
plt.plot(ldlog, ldlog * par[0] + par[1], color='Cyan', label='Weighted Polyfit')
print('The weighted fit parameters', par[0], par[1])

xy = np.column_stack((ldlog, logden))
xy_count = xy.tolist()
counts_raw = []
for i in range(len(xy)):
    counts_raw.append(xy_count.count(xy_count[i]))
print (len(ldlog), len(logden), len(counts_raw))
counts_raw = np.sqrt(np.array(counts_raw))
par, cov = np.polyfit(ldlog, logden, deg = 1, cov = True, w = counts_raw)
plt.plot(ldlog, ldlog * par[0] + par[1], color='m', label='Weighted Polyfit raw data')
print('The weighted fit parameters raw data', par[0], par[1])

plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
plt.title('Plot for 1.2 smoothing factor excluding exterior to r200')
# ticks = np.linspace(min(ldlog), max(ldlog), 5)
cbar = plt.colorbar(mappable=im)
cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())
plt.ylim(min(yedge), max(yedge))
# cbar = fig.colorbar(im, cax = cax, orientation = 'horizontal', format = ticker.LogFormatter())
# cbar.ax.tick_params(labelsize=10, width = .7, size = 8)
# cbar.set_label('lumden')
# cbar.solids.set_edgecolor('face')
# cbar.ax.xaxis.set_label_position('bottom')
plt.legend()
plt.show()







########################################################### DOING IT BUT FOR LUM MORE THAN -.5 BROKEN######################
lim_ldlog = []
lim_logden = []
for i in range(len(ldlog)):
    if ldlog[i] > -.5:
    #   continue
    #else:
        lim_ldlog.append(ldlog[i])
        lim_logden.append(logden[i])
lim_logden = np.array(lim_logden)
lim_ldlog = np.array(lim_ldlog)

fig, ax = plt.subplots(1, figsize=(8, 8))
nbins = 40
divider = make_axes_locatable(ax)
bins = [np.linspace(min(lim_ldlog), max(lim_ldlog), nbins + 1), np.linspace(min(lim_logden), max(lim_logden), nbins + 1)]
h, xedge, yedge = np.histogram2d(lim_ldlog, lim_logden, bins=bins)
h = h.transpose()

cmap = plt.get_cmap('Blues')
vmin = min(h.flatten()) + 1  # added a one so I can make the colormap logarithmic.
vmax = max(h.flatten())
X, Y = np.meshgrid(xedge, yedge)  # X_dim and Y_dim same as xedge and yedge 41,41
im = plt.pcolormesh(X, Y, h, cmap=cmap, edgecolors='black', norm=LogNorm(vmin=vmin, vmax=vmax), linewidth=0.3)


par, cov = np.polyfit(h[0], h[1], deg=1, cov=True)
plt.plot(lim_ldlog, lim_ldlog * par[0] + par[1], color='red', label='Polyfit')
print('The fit parameters', par[0], par[1])

plt.plot(ldlog, ldlog * .7443409 + .6830892, color='Cyan', label='uncut Weighted Polyfit')

xy = np.column_stack((lim_ldlog, lim_logden))
xy_count = xy.tolist()
counts_raw = []
for i in range(len(xy)):
    counts_raw.append(xy_count.count(xy_count[i]))
print (len(lim_ldlog), len(lim_logden), len(counts_raw))
counts_raw = np.sqrt(np.array(counts_raw))
par, cov = np.polyfit(lim_ldlog, lim_logden, deg = 1, cov = True, w = counts_raw)
plt.plot(lim_ldlog, lim_ldlog * par[0] + par[1], color='m', label='Weighted Polyfit raw data')
print('The weighted fit parameters raw data', par[0], par[1])

counts = []
h_count = h.tolist()
for i in range(len(h_count)):
    counts.append(h_count.count(h_count[i]))
par, cov = np.polyfit(h[0], h[1], deg = 1, cov = True, w = counts)
plt.plot(ldlog, ldlog * par[0] + par[1], color='m', linestyle = '--' , label='Weighted Polyfit')
print('The weighted fit parameters', par[0], par[1])

plt.xlim(-.6, 4)

plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
plt.title('limiting x-axis to > -.5 and y-axis to > .5')
cbar = plt.colorbar(mappable=im)
cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())
plt.ylim(min(yedge), max(yedge))
plt.legend()
plt.show()










# ####################################################################################
# fig, ax=plt.subplots(1, figsize = (8,8))
# nbins = 40
# divider = make_axes_locatable(ax)
# cax = divider.new_vertical(size='5%', pad = .6, pack_start=True)
# fig.add_axes(cax)
# bins = [np.linspace(min(ldlog),max(ldlog),nbins+1), np.linspace(min(logden),max(logden),nbins+1)]
# h, xedge, yedge = np.histogram2d(ldlog, logden, bins=bins)
# cmap = plt.get_cmap('Blues')
# vmin = min(h.flatten()) + 1  #added a one so I can make the colormap logarithmic.
# vmax = max(h.flatten())
# X, Y = np.meshgrid(xedge,yedge)
# im = ax.pcolormesh(X, Y, h, cmap=cmap, edgecolors='black', norm=LogNorm(vmin=vmin, vmax=vmax), linewidth=0.3)
#
# xc , yc = np.where(h > 0)
# a, b = np.polyfit(xedge[xc],yedge[yc], 1)
# print(a, b)
# print(type(a), type(b))
# linear_fit = np.poly1d([a, b])
#
# ax.plot(xedge, a*xedge + b, color = 'red', label = 'polyfit')
# #range = np.linspace(min(ldlog), max(ldlog), 100)
# #ax.plot(range, linear_fit(range), color = 'RED', label = 'polyfit')
# #plt.plot   (xedge,a*xedge + b, c='red', label = 'np.polyfit')
# #ax.plot(xedge,(a)*xedge + b, c='red', label = 'np.polyfit')
# #plt.plot(xedge, parameters_edge.intercept +  parameters_edge.slope * xedge, color = 'yellow', label = 'lineregress based on edge')
# #plt.plot(ldlog, parameters_log.intercept +  parameters_log.slope * ldlog, color = 'Green', label = 'based on log data')
# #ax.plot(xedge, parameters_log.intercept + parameters_log.slope * xedge, color = 'Green', label = 'based on log data')
# #plt.plot(h[0], parameters_h.intercept + h[0] * parameters_h.slope, color = 'red')
# #plt.xlabel(r'LD [$\log \delta_{LD}$]',fontsize=15)
# #plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]',fontsize=15)
# #plt.title('Plot for 1.2 smoothing factor excluding exterior to r200')
# #cbar = ax.colorbar(mappable=im)
# #cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())
# #plt.legend()
# plt.show()
# # ax.plot()
#
# ###############################################################################################33
# #plt.scatter(ldlog,logden)
# #plt.plot(xedge, parameters_log.intercept +  parameters_log.slope * xedge, color = 'Green', label = 'based on log data')
# #plt.show()
# ####################################################################################################
# # Im-show coloar plot, it looks upside down
# #fig, ax=plt.subplots(1, figsize = (8,8))
# #nbins = 40
# #divider = make_axes_locatable(ax)
# #bins = [np.linspace(min(ldlog),max(ldlog),nbins+1), np.linspace(min(logden),max(logden),nbins+1)]
# #h, xedge, yedge = np.histogram2d(ldlog, logden, bins=bins)
# #cmap = plt.get_cmap('Blues')
# #vmin = min(h.flatten()) + 1  #added a one so I can make the colormap logarithmic.
# #vmax = max(h.flatten())
# #X, Y = np.meshgrid(xedge,yedge)
# #im = ax.pcolormesh(X,Y,h,cmap=cmap,edgecolors='black',norm=LogNorm(vmin=vmin,vmax=vmax),linewidth=0.3)
#
#
# #z = np.array([i*i+j*j for j in logden for i in ldlog])
# #Z = z.reshape(len(ldlog), len(logden))
# #plt.imshow(h, interpolation='bilinear')
# #plt.show()
# ####################################################################################################
# #fig, ax=plt.subplots(1, figsize = (8,8))
# #nbins = 40
# #divider = make_axes_locatable(ax)
# #bins = [np.linspace(min(ldlog),max(ldlog),nbins+1), np.linspace(min(logden),max(logden),nbins+1)]
# #H, xedges, yedges = np.histogram2d(ldlog, logden, bins = bins)
# #cmap_color = plt.get_cmap('Blues')
# #plt.plot(xedge, parameters_log.intercept +  parameters_log.slope * xedge, color = 'Green', label = 'based on log data')
# #x, y = np.meshgrid(xedge, yedge)
#
# #im = ax.pcolormesh(x , y, H, norm = 'log', edgecolors = 'black', shading = 'flat', cmap = 'Blues')
# #plt.show()
# ####################################################################################
# #plt.hist2d(ldlog,logden, bins = 150)
# #plt.plot(xedge, parameters_log.intercept +  parameters_log.slope * xedge, color = 'Green', label = 'based on log data')
# #plt.show()
# ########################################################################################
