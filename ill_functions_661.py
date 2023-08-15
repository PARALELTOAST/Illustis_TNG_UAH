import scipy
from numba import njit
import numpy as np
import tqdm
from scipy import spatial
import os
import pandas as pd
# This is an attempt to organize / collect all functions and useful functions that can be made so that
# primary documents are neater and visually simple

# TO IMPORT
# drop in run dir
# import ill_functions as fun
    # fun.limiter
    # fun.lum_OverDen_func
    # etc

# Limiting the WHIM that is inside the halos, This will return the limited arrays for:
# density, temperature, pressure, and mass, as well as an array that contains the indices for the
# whim that is inside a halo
def limiter(Masks_applied_halos: str, Load_halo_pos_txt: bool, Load_rxxx_txt: bool, halo_positions: object, halo_rxxx: object, working_path: str) -> object:
    # Masks_applied_halos = string of masks that have been applied to halo information
        # EG: 'Lum_dust_mass14'
    # Load_halo_pos_txt = bool for if halo position is a text file load in or an array, True is for if input is a txt file.
        #  positions file name MUST BE halo_positions.txt
    # Load_rxxx_txt = bool for is rxxx is a text file or an array, True is for txt file
        # radius file must be halor200.txt
    # halo_positions = array of halo x,y,z positions [IF LOADING FROM TEXT FILES SET AS EMPTY ARRAYS, []]object
        # EG: arr = ([x,y,z],[x,y,z]................)
    # halo_rxxx = array of halo radii [IF LOADING FROM TEXT FILES SET AS EMPTY ARRAYS, []]
        # EG: arr = (r,r,r,r...............)
    # working_ path = string for the location of the current working directory
        # EG: '/path/to/working/directory'

    # The halo and whim information
    limiters = Masks_applied_halos
    print('Getting halo and whim info')
    WHIM_Data = pd.read_csv('/home/benjamin/Thesis/Illustris_TNG/dataset_WHIMgas__tng3001_z0__ratio400.csv')
    WHIM_Data = np.asarray(WHIM_Data)
    z_pos_WHIM = WHIM_Data[:, 1]  # z coordinate for WHIM in Mpc
    y_pos_WHIM = WHIM_Data[:, 2]  # y ...
    x_pos_WHIM = WHIM_Data[:, 3]  # x ...
    mass_WHIM = WHIM_Data[:, 4]  # units in M_sun
    den_WHIM = WHIM_Data[:, 5]  # units in M_sun * Mpc^-3
    temp_WHIM = WHIM_Data[:, 8]  # units in Kelvin (K)
    press_WHIM = WHIM_Data[:, 9]  # pressure in units of KeV*cm^-3
    ###############################################################
    halo_rs = []  # holder for the radii of the halos
    if Load_halo_pos_txt:
        halos = np.array(np.loadtxt('halo_positions.txt'))  # txt file of all halo pos x,y,z
    else:
        halos = halo_positions

    print("Reading halo_rxxx")
    if Load_rxxx_txt:
        with open('halo_r200.txt') as file:
            while line := file.readline().rstrip():
                halo_rs.append(float(line))  # add each radius to the list
    else:
        halo_rs = halo_rxxx

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
    indices=[]
    for i in tqdm.trange(len(halos)):
        points_in_radius = tree.query_ball_point(halos[i], halo_rs[i], workers=-1)  # setting workers to -1 maximizes parallelization
        #points_in_radius_set = set(points_in_radius)
        total += len(points_in_radius)  # to keep track of the total number of points that got zeroed out (although this could double count, but there are no dual halo inhabitants for r200)
        indices.extend(points_in_radius)
        #for point in points_in_radius:
        #    den_WHIM[i] = 0
        #    temp_WHIM[i] = 0
        #    press_WHIM[i] = 0
        #    mass_WHIM[i] = 0
        #    whim_in_halo_index.append([point, i])
    print('Total number of Whim in halos:', total)
    indices_set = set(indices)
    for i in tqdm.tqdm(range(len(den_WHIM))):
        if i in indices_set:
            continue
        else:
            den_WHIM[i] = 0
            temp_WHIM[i] = 0
            press_WHIM[i] = 0
            mass_WHIM[i] = 0
    print(den_WHIM)
    # making the folder the data will be saved to
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
    return den_WHIM, mass_WHIM, press_WHIM, temp_WHIM, whim_in_halo_index

# Luminosity over density function, creates a lumminosity grid based on how many 'cells' are in the box
@njit
def Lum_OverDen_func(box_len, num_cells, x, y, z, lum_r, lum_avg_den, LD_Overden):
    cell_len = box_len/num_cells
    x_bins = np.linspace(0.0, 303.0, num_cells + 1)
    y_bins = np.linspace(0.0, 303.0, num_cells + 1)
    z_bins = np.linspace(0.0, 303.0, num_cells + 1)
    x_idx = np.digitize(x, x_bins)
    y_idx = np.digitize(y, y_bins)
    z_idx = np.digitize(z, z_bins)
    LD_smooth_param = 1.2/cell_len #smoothing 1.2 Mpc converted to Mpc/cell_len (dimensionless), Units of Mpc
    for n in range(len(x)):
        LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = LD_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + lum_r[n]/lum_avg_den/(box_len/num_cells)**3 #matching luminosity values with binned coordinates
    return LD_Overden

# whim mass function, same as lum/den function
@njit
def WHIM_mass_func(box_len, num_cells, x, y, z, mass_WHIM, Mass_grid):
    cell_len = box_len/num_cells
    x_bins = np.linspace(0.0, 303.0, num_cells + 1)
    y_bins = np.linspace(0.0, 303.0, num_cells + 1)
    z_bins = np.linspace(0.0, 303.0, num_cells + 1)
    x_idx = np.digitize(x, x_bins)
    y_idx = np.digitize(y, y_bins)
    z_idx = np.digitize(z, z_bins)
    for n in range(len(x)):
        Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]
    return Mass_grid

# whim pressure function, same as lum/den function
@njit
def WHIM_Pressure_func(box_len, num_cells, x, y, z, mass_WHIM, press_WHIM, Mass_grid, Pressure_grid):
    cell_len = box_len/num_cells
    x_bins = np.linspace(0.0, 303.0, num_cells + 1)
    y_bins = np.linspace(0.0, 303.0, num_cells + 1)
    z_bins = np.linspace(0.0, 303.0, num_cells + 1)
    x_idx = np.digitize(x, x_bins)
    y_idx = np.digitize(y, y_bins)
    z_idx = np.digitize(z, z_bins)
    for n in range(len(x)):
        Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Pressure_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*press_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
    return Pressure_grid #Mass weighted mean pressure for each cell

# whim temperature function, same as lum/den function
@njit
def WHIM_Temp_func(box_len, num_cells, x, y, z, mass_WHIM, temp_WHIM, Mass_grid, Temp_grid):
    cell_len = box_len/num_cells
    x_bins = np.linspace(0.0, 303.0, num_cells + 1)
    y_bins = np.linspace(0.0, 303.0, num_cells + 1)
    z_bins = np.linspace(0.0, 303.0, num_cells + 1)
    x_idx = np.digitize(x, x_bins)
    y_idx = np.digitize(y, y_bins)
    z_idx = np.digitize(z, z_bins)
    for n in range(len(x)):
        Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Temp_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] + mass_WHIM[n]*temp_WHIM[n]/Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1]
    return Temp_grid #Mass weighted mean temperature for each cell

# whim over density, same as lum/den function
@njit
def WHIM_OverDen_func(box_len, num_cells, x, y, z, Mass_grid, mean_baryon_den, WHIM_Overden):
    cell_len = box_len/num_cells
    x_bins = np.linspace(0.0, 303.0, num_cells + 1)
    y_bins = np.linspace(0.0, 303.0, num_cells + 1)
    z_bins = np.linspace(0.0, 303.0, num_cells + 1)
    x_idx = np.digitize(x, x_bins)
    y_idx = np.digitize(y, y_bins)
    z_idx = np.digitize(z, z_bins)
    for n in range(len(x)):
        WHIM_Overden[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] = Mass_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] / (box_len/num_cells)**3 * 400. / mean_baryon_den #Volume_grid[z_idx[n] - 1, y_idx[n] - 1, x_idx[n] - 1] /mean_baryon_den  #matching luminosity values with binned coordinates
    return WHIM_Overden