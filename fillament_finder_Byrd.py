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

############################## Filling in the fillamets#########################################################
filefils = 'filament_coords_and_lengths.npy'

# Read filaments
seg_x, seg_y, seg_z, lengths = np.load(filefils, allow_pickle=True)

print('Number of filaments and value at seg_x[0][1] and seg_y[0]:', len(seg_x[0]), seg_x[0][1], seg_y[0])
box_len = 303.  # Units of Mpc
num_cells =  600  # 1500 to match 0.2 Mpc cell length of EAGLE is too large. Maybe try bladerunner.
# cell_len = box_len/num_cells
fil_grid = np.zeros([num_cells, num_cells, num_cells])#, dtype=np.bool)
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


fil_grid = np.zeros([num_cells, num_cells, num_cells])#, dtype=np.bool)

x = np.concatenate(seg_x_new)
y = np.concatenate(seg_y_new)
z = np.concatenate(seg_z_new)


grid_fil, edges = np.histogramdd((x,y,z), bins = (x_bins, y_bins, z_bins))
bool_fil = np.bool_(grid_fil)
fil_grid[bool_fil] += 1

########################### Slice if whim################################################
# plt.imshow(fil_grid[0:50,:,:].sum(0).T, origin = 'lower', extent = (0,303, 0, 303), norm = LogNorm())
# plt.ylabel('Y [Mpc]')
# plt.xlabel('X [Mpc]')
# plt.title('Slice of The Filament Grid')
# #plt.colorbar()
# #plt.set_cmap('hot')
# plt.show()


# for i in tqdm.trange(len(seg_x_new)):
#     grid_fil, edges  = np.histogramdd((seg_x_new[i], seg_y_new[i], seg_z_new[i]) , bins = (x_bins, y_bins, z_bins))
#     bool_fil = np.bool_(grid_fil)
#     fil_grid[bool_fil] += 1

np.save('fil_gri.npy', fil_grid)