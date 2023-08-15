import numpy as np

#plt.ion()

filefils = 'filament_coords_and_lengths.npy'

# Read filaments
seg_x, seg_y, seg_z, lengths = np.load(filefils, allow_pickle=True)

#seg_x = np.asarray(seg_x)
print('Number of filaments and value at seg_x[0][1] and seg_y[0]:', len(seg_x[0]), seg_x[0][1], seg_y[0])
#print('length of seg_x, y, and z at specific array place',len(seg_x[len(seg_x) - 10]), len(seg_y[len(seg_y) - 10]), len(seg_z[len(seg_z) - 10]))
#print('total number of points in segments',len(seg_x[:][0]))
box_len = 303.
num_cells = 600
# I changed this
fil_grid = np.zeros([num_cells, num_cells, num_cells],dtype=np.bool)

x_bins = np.linspace(0.0,303.0,num_cells + 1)
y_bins = np.linspace(0.0,303.0,num_cells + 1)
z_bins = np.linspace(0.0,303.0,num_cells + 1)

seg_tot_points=0
for i in range(len(seg_x)):
    seg_tot_points = seg_tot_points + len(seg_x[i])

print('total number of filament points',seg_tot_points)    

#seg_x_array = np.zeros(seg_tot_points)
#seg_y_array = np.zeros(seg_tot_points)
#seg_z_array = np.zeros(seg_tot_points)

############ adding more seg points in between the already existing seg points, to create better filaments #################
seg_x_new = np.array(seg_x)
seg_y_new = np.array(seg_y)
seg_z_new = np.array(seg_z)


print('length of new seg_x[0]',len(seg_x_new[0]))
#quit()
'''
for i in range(20):
    z_in = 15*i
    z_fin = 15*(i+1)
    for j in range(len(seg_x)):
        if ( np.mean(seg_z_new[j]) > z_in ) & ( np.mean(seg_z_new[j]) <= z_fin ):
            fil_scat = plt.plot(seg_x_new[j], seg_y_new[j],color='k')
    plt.show()
    plt.close()
 1'''
#quit()
if 1==1: #added to avoid fixing tab space of defined function
#@njit
#def seg_points_func(seg_x_array, seg_y_array, seg_z_array, seg_x_array_new, seg_y_array_new, seg_z_array_new):

    steps = 40
    for n in range(len(seg_x_new)):
        for m in range(len(seg_x_new[n])):
            if m==len(seg_x_new[n]) - 1:
                continue
            x_step = (seg_x_new[n][m+1] - seg_x_new[n][m]) / steps
            y_step = (seg_y_new[n][m+1] - seg_y_new[n][m]) / steps
            z_step = (seg_z_new[n][m+1] - seg_z_new[n][m]) / steps
            seg_x_add = np.linspace(seg_x_new[n][m] + x_step, seg_x_new[n][m+1] - x_step, steps - 1) 
            seg_y_add = np.linspace(seg_y_new[n][m] + y_step, seg_y_new[n][m+1] - y_step, steps - 1)
            seg_z_add = np.linspace(seg_z_new[n][m] + z_step, seg_z_new[n][m+1] - z_step, steps - 1)
            seg_x_new[n] = np.insert(seg_x_new[n],m,seg_x_add)
            seg_y_new[n] = np.insert(seg_y_new[n],m,seg_y_add)
            seg_z_new[n] = np.insert(seg_z_new[n],m,seg_z_add)
            #seg_x_new[n] = np.append(seg_x_new[n], seg_x_add)
            #seg_y_new[n] = np.append(seg_y_new[n], seg_y_add)
            #seg_z_new[n] = np.append(seg_z_new[n], seg_z_add)
#    return seg_z_array_new, seg_y_array_new, seg_x_array_new    

   ####### plotting filaments. Filaments given by Daniela ##########
'''   
for i in range(20):
    z_in = 15*i
    z_fin = 15*(i+1)
    for j in range(len(seg_x_new)):
        if ( np.mean(seg_z_new[j]) > z_in ) & ( np.mean(seg_z_new[j]) <= z_fin ):
            fil_scat = plt.plot(seg_x_new[j], seg_y_new[j],color='k')
    plt.show()
    plt.close()
'''
#quit()
#seg_points_func(seg_x_array, seg_y_array, seg_z_array, seg_x_array_new, seg_y_array_new, seg_z_array_new)

#del seg_x
#del seg_y
#del seg_z
print('length of new seg_x[0]',len(seg_x_new[0]))
seg_x_array = []
seg_y_array = []
seg_z_array = []
for i in range(len(seg_x_new)):
    for j in range(len(seg_x_new[i])):
        seg_x_array.append(seg_x_new[i][j])
        seg_y_array.append(seg_y_new[i][j])
        seg_z_array.append(seg_z_new[i][j])

x_idx = np.digitize(seg_x_array,x_bins)
y_idx = np.digitize(seg_y_array,y_bins)
z_idx = np.digitize(seg_z_array,z_bins)

print('length of new segment arrays',len(seg_x_array),len(seg_y_array),len(seg_z_array))
print('length of indices',len(z_idx))


# #### attempting to match the grib from Byrd Script
#
# x = np.concatenate(seg_x_new)
# y = np.concatenate(seg_y_new)
# z = np.concatenate(seg_z_new)
#
# grid_fil, edges = np.histogramdd((x,y,z), bins = (x_bins, y_bins, z_bins))
# bool_fil = np.bool_(grid_fil)
# fil_grid[bool_fil] += 1
#
# np.save('fil_grid_viapat.npy', fil_grid)










#quit()
'''
X_mesh = np.linspace(0.5*box_len/num_cells, box_len - 0.5*box_len/num_cells , num_cells)
Y_mesh = np.linspace(0.5*box_len/num_cells, box_len - 0.5*box_len/num_cells , num_cells)
Z_mesh = np.linspace(0.5*box_len/num_cells, box_len - 0.5*box_len/num_cells , num_cells)

X_mesh , Y_mesh, Z_mesh = np.meshgrid(X_mesh,Y_mesh,Z_mesh)
'''

#fil_grid = np.load('fil_grid_Daniela.npy',allow_pickle=True)
#@njit
# def filament_func(box_len, num_cells, seg_x_array_new, seg_y_array_new, seg_z_array_new, fil_grid):
#     fil_rad = 3.0 #1.0Mpc radius
#     for n in range(len(seg_x_array_new)):
#         #if n==len(seg_x_array) - 1:
#             #continue
#         k_max = int(fil_rad/(box_len/num_cells))
#         for k in range(-1*k_max,k_max+1):
#             for l in range(-1*k_max, k_max+1):
#                 for m in range(-1*k_max, k_max+1):
#                     if z_idx[n] - 1 + k > num_cells - 1 or z_idx[n] - 1 + k < 0 or y_idx[n] - 1 + l > num_cells - 1 or y_idx[n] - 1  + l < 0 or x_idx[n] - 1 + m > num_cells - 1 or x_idx[n] - 1 + m < 0:
#                         continue
#                     fil_grid[z_idx[n] -1 + k, y_idx[n] -1 + l, x_idx[n] -1 + m] = True
#             '''
#             r_ab_sq = (seg_x_array[n+1] - seg_x_array[n])**2 + (seg_y_array[n+1] - seg_y_array[n])**2 + (seg_z_array[n+1] - seg_z_array[n])**2
#             cross_mag_sq = ((Y_mesh[y_idx[n] - 1 + k] - seg_y_array[n])*(seg_z_array[n+1] - seg_z_array[n]) - (Z_mesh[z_idx[n] - 1 + k] - seg_z_array[n])*(seg_y_array[n+1] - seg_y_array[n]))**2 + ((Z_mesh[z_idx[n] - 1 + k] - seg_z_array[n])*(seg_x_array[n+1] - seg_x_array[n]) - (X_mesh[x_idx[n] - 1 + k] - seg_x_array[n])*(seg_z_array[n+1] - seg_z_array[n]))**2 + ((X_mesh[x_idx[n] - 1 + k] - seg_x_array[n])*(seg_y_array[n+1] - seg_y_array[n]) - (Y_mesh[y_idx[n] - 1 + k] - seg_y_array[n])*(seg_x_array[n+1] - seg_x_array[n]))**2
#             if abs(fil_rad**2 * r_ab_sq - cross_mag_sq) < 1.0e-15:
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1 - k, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1 + k, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1 - k, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1 + k, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1 - k, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1 + k, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1 - k, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1 - k, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1 + k, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1 - k, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1 - k, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1 + k, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1 + k, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1 - k, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1 + k, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1 + k, y_idx[n] -1, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1 + k, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1, x_idx[n] -1 + k] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1, x_idx[n] -1 - k] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1 - k, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1 - k, y_idx[n] -1, x_idx[n] -1] = 10000.
#             fil_grid[z_idx[n] -1, y_idx[n] -1, x_idx[n] -1] = 10000.
#             '''
#     return fil_grid

#filament_func(box_len, num_cells, seg_x_array, seg_y_array, seg_z_array, fil_grid)
#np.save('fil_grid_Patrick_rad3Mpc',fil_grid)
#fil_grid = np.load('fil_grid_Daniela.npy',allow_pickle = True)
#'''
# for i in range(20):
#     fig, ax = plt.subplots(1, figsize = (8,8))
#     cmap_WHIM = mpl.cm.get_cmap('Purples')
#     z_in = 15*i #in Mpc
#     z_fin = 15*(i+1) #in Mpc
#     gridIndexMin = int((num_cells*z_in)/box_len)
#     gridIndexMax = int((num_cells*z_fin)/box_len)
#     imageFilSlice = np.mean(fil_grid[gridIndexMin:gridIndexMax, :, :]*1000,axis=0)
#     ####### plotting filaments. Filaments given by Daniela ##########
#     for j in range(len(seg_x)):
#         if ( np.mean(seg_z[j]) > z_in ) & ( np.mean(seg_z[j]) <= z_fin ):
#             fil_scat = ax.plot(seg_x[j], seg_y[j],'-',color='k')
#
#     Fil_plot = ax.imshow(imageFilSlice, cmap=cmap_WHIM, norm=LogNorm(vmin=1e-2 , vmax=1000), extent= [0.,303.,0.,303.0], origin='lower',alpha=0.6, label="filament")
#     divider = make_axes_locatable(ax)
#     cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
#     fig.add_axes(cax)
#     cbar = fig.colorbar(Fil_plot, cax=cax, orientation = "horizontal", format=ticker.LogFormatter())
#     cbar.solids.set_edgecolor("face")
#     cbar.ax.xaxis.set_label_position('bottom')
#     plt.xlabel('X [Mpc]',fontsize=10)
#     plt.ylabel('Y [Mpc]',fontsize=10)
#     #plt.savefig('filament_volume_location_slice{i}.pdf'.format(i=i))
#     plt.show()
#     plt.close()
#'''
