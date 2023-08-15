import ill_functions as fun
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gf
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Goal of this script is to plot different cuts of WHIM on Patrick's Filament plots

# doing the limiting for luminosity and dust only, need to do only once
# fun.limiter('lum_dust', True, True,[0],[0], '/home/benjamin/Thesis/Illustris_TNG')

den_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/den_WHIM_lum_dust.txt')
mass_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/mass_WHIM_lum_dust.txt')
press_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/press_WHIM_lum_dust.txt')
temp_WHIM = np.loadtxt('/home/benjamin/Thesis/Illustris_TNG/Limited_Data/temp_WHIM_lum_dust.txt')

# same parameters as above but zeros are removed, it is for the histograms
den_whim = den_WHIM[den_WHIM != 0]
mass_whim = mass_WHIM[mass_WHIM != 0]
press_whim = press_WHIM[press_WHIM != 0]
temp_whim = temp_WHIM[temp_WHIM != 0]

# subhalo_mag_r_dust = subhalo_mag_r_dust[Lum_mask*Flag_mask*galaxy_mass_mask]
# subhalo_mass = subhalo_mass[Lum_mask*Flag_mask*galaxy_mass_mask]
# subhalo_mass_stellar = subhalo_mass_stellar[Lum_mask*Flag_mask*galaxy_mass_mask]
# np.savetxt("subhalo_mass_total.txt",subhalo_mass,fmt="%s")
# np.savetxt("subhalo_stellar_mass.txt",subhalo_mass_stellar,fmt="%s")
# np.savetxt("subhalo_r_magnitude_dust.txt",subhalo_mag_r_dust,fmt="%s")
subhalo_mag_r_dust = np.loadtxt('subhalo_r_magnitude_dust.txt')
mag_sun_r = 4.42
lum_r = 10.0**((subhalo_mag_r_dust - mag_sun_r)/-2.5)
lum_avg_den = sum(lum_r.flatten()) / 303.**3
print("average luminosity density of full volume, filtered:",lum_avg_den)

###################### Constructing Luminosity Overdensity grid ###############################
num_cells = 600 #1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
box_len = 303. #Units of Mpc
#cell_len = box_len/num_cells
LD_Overden = np.zeros([num_cells,num_cells,num_cells])
#x = subhalo_coord[Lum_mask*Flag_mask*galaxy_mass_mask,2] / (0.6774*10**3)
#y = subhalo_coord[Lum_mask*Flag_mask*galaxy_mass_mask,1] / (0.6774*10**3)
#z = subhalo_coord[Lum_mask*Flag_mask*galaxy_mass_mask,0] / (0.6774*10**3)
#np.savetxt("subhalo_positions.txt", np.transpose([x,y,z]),fmt="%s")

x, y, z = np.loadtxt('subhalo_positions.txt', unpack=True)

x_bins = np.linspace(0.0,303.0,num_cells + 1)
y_bins = np.linspace(0.0,303.0,num_cells + 1)
z_bins = np.linspace(0.0,303.0,num_cells + 1)

x_idx = np.digitize(x,x_bins)
y_idx = np.digitize(y,y_bins)
z_idx = np.digitize(z,z_bins)

# The lum overden func
fun.Lum_OverDen_func(box_len, num_cells, x, y, z, lum_r, lum_avg_den, LD_Overden)

print("number of galaxies for smoothing:",np.size(x))
print("dimensions of LD grid",np.shape(LD_Overden))
print("number of bins with galaxies (lum values):",np.size(np.nonzero(LD_Overden.flatten())))
########### smoothing using Gaussian kernel ####################################
smooth_param = 2.0 #Units of Mpc
LD_Overden = gf(LD_Overden, sigma=0.6*smooth_param)

# Reading the WHIM into the file
WHIM_Data = pd.read_csv('dataset_WHIMgas__tng3001_z0__ratio400.csv')
WHIM_Data = np.asarray(WHIM_Data)

z_pos_WHIM = WHIM_Data[:,1] #x coordinate for WHIM in Mpc
y_pos_WHIM = WHIM_Data[:,2] #y ...
x_pos_WHIM = WHIM_Data[:,3] #z ...

# These values should come from the fun.functions
# mass_WHIM = WHIM_Data[:,4] #units in M_sun
# den_WHIM = WHIM_Data[:,5] #units in M_sun * Mpc^-3
# temp_WHIM = WHIM_Data[:,8] #units in Kelvin (K)
# press_WHIM = WHIM_Data[:,9] #pressure in units of KeV*cm^-3
volume = []
for i in range(len(mass_WHIM)):
    if den_WHIM[i] != 0:
        volume.append(mass_WHIM[i] / den_WHIM[i])
    else:
        volume.append(0)
vol = (mass_whim / den_whim)

#boltz_const = 1.380649 * 10**-23
#boltz_const * temp_WHIM * (6.242*10^15) #converting Joules to keV

########## plotting distribution of WHIM with cuts based on pos applied ##################

plt.hist(np.log10(den_whim),np.linspace(min(np.log10(den_whim)),max(np.log10(den_whim)),50),density=False)
plt.xlabel(r'$\log(\rho)$',fontsize=15)
plt.savefig('WHIMden_dist.pdf')
plt.close()

plt.hist(np.log10(vol),np.linspace(min(np.log10(vol)),max(np.log10(vol)),50),density=False)
plt.xlabel(r'$\log(V)$',fontsize=15)
plt.savefig('volume_dist.pdf')
plt.close()

plt.hist(np.log10(mass_whim),np.linspace(min(np.log10(mass_whim)),max(np.log10(mass_whim)),50),density=False)
plt.xlabel(r'$\log(M)$',fontsize=15)
plt.savefig('WHIMmass_dist.pdf')
plt.close()

x_idx = np.digitize(x_pos_WHIM,x_bins)
y_idx = np.digitize(y_pos_WHIM,y_bins)
z_idx = np.digitize(z_pos_WHIM,z_bins)

mean_baryon_den = 0.618 * 10**10 #mean baryon density of universe in units of M_sun * Mpc^-3
WHIM_Den_avg = sum(mass_WHIM) * 400. / 303.**3 #WHIM gas sampled selecting 1 out of 400.
WHIM_Den_avg_2 = sum(mass_WHIM)/ sum(volume)

print("WHIM density average of whole box:",WHIM_Den_avg, WHIM_Den_avg_2)
print("volume range and cell volume:",min(volume), max(volume), (box_len/num_cells)**3.)

WHIM_Overden = np.zeros([num_cells,num_cells,num_cells])
Temp_grid = np.zeros([num_cells,num_cells,num_cells]) #mass weighted mean temperature of each cell
Mass_grid = np.zeros([num_cells,num_cells,num_cells])
#Volume_grid = np.zeros([num_cells,num_cells,num_cells])
Pressure_grid = np.zeros([num_cells, num_cells, num_cells])

#CALL ALL OVERDEN FUNCS
fun.WHIM_mass_func(box_len, num_cells, x, y, z, mass_WHIM, Mass_grid)
fun.WHIM_Pressure_func(box_len, num_cells, x, y, z, mass_WHIM, press_WHIM, mass_WHIM, Pressure_grid)
fun.WHIM_Temp_func(box_len, num_cells, x, y, x, mass_WHIM, temp_WHIM, Mass_grid,Temp_grid)
fun.WHIM_OverDen_func(box_len, num_cells, x, y, z, Mass_grid, mean_baryon_den, WHIM_Overden)

print('length of WHIM temp dataset:', len(temp_WHIM), len(mass_WHIM), len(den_WHIM))
print('WHIM density range for cells:', min(WHIM_Overden.flatten()), max(WHIM_Overden.flatten()))
print('WHIM pressure range for cells:', min(Pressure_grid.flatten()), max(Pressure_grid.flatten()))
print('WHIM Temp range for cells:', min(Temp_grid.flatten()), max(Temp_grid.flatten()))
print("number of bins with WHIM (den values):", np.size(np.nonzero(WHIM_Overden.flatten())))
vol_WHIM = np.size(np.nonzero(WHIM_Overden.flatten())) / len(WHIM_Overden.flatten())  # * 303.**3
print("fraction of volume occupied by WHIM", vol_WHIM)
# some of the non-selected WHIM data may be in bins with no WHIM. Also try binning according to which gridpoit is closer to positions.
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
    print("min nonzero LD for image slice:", min(vmin_nonzero))
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