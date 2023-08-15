import numpy as np


def create_frequency_map_adding_sampling_points(xbins,ybins,zbins, Nsampling, seg_x, seg_y, seg_z):
    seg_x_full = []
    seg_y_full = []
    seg_z_full = []
    for f in range(len(seg_x)):
        # 1) add sampling points to filament
        seg_x_interp = []
        seg_y_interp = []
        seg_z_interp = []
        for s in range(len(seg_x[f])-1): # loop over each segment
            seg = (seg_x[f][s:s+2], seg_y[f][s:s+2], seg_z[f][s:s+2])
            dx =  np.diff(seg[0]) / (Nsampling-1)
            dy =  np.diff(seg[1]) / (Nsampling-1)
            dz =  np.diff(seg[2]) / (Nsampling-1)
            sampling_points_x = np.array([ float(seg[0][0]+N*dx) for N in range(Nsampling)])
            sampling_points_y = np.array([ float(seg[1][0]+N*dy) for N in range(Nsampling)])
            sampling_points_z = np.array([ float(seg[2][0]+N*dz) for N in range(Nsampling)])
            #append sampling points to filament
            if s < len(seg_x[f])-2: #do not append the last point, which would be repeated in the next s value
                seg_x_interp.extend(sampling_points_x[:-1])
                seg_y_interp.extend(sampling_points_y[:-1])
                seg_z_interp.extend(sampling_points_z[:-1])
            else:
                seg_x_interp.extend(sampling_points_x)
                seg_y_interp.extend(sampling_points_y)
                seg_z_interp.extend(sampling_points_z)
        # save filament with more sampling points
        seg_x_full.append(seg_x_interp)
        seg_y_full.append(seg_y_interp)
        seg_z_full.append(seg_z_interp)
    seg_x_full = np.array(seg_x_full)
    seg_y_full = np.array(seg_y_full)
    seg_z_full = np.array(seg_z_full)
    # 2) compute frequency map
    H, edges = np.histogramdd((np.concatenate(seg_x_full), np.concatenate(seg_y_full), np.concatenate(seg_z_full)) , bins=(xbins,ybins,zbins), density=False)
    boolH = np.bool_(H)
    return boolH

def create_frequency_map_adding_sampling_points_with_fil_struct(Np, Nsampling, seg_x, seg_y, seg_z):
    HHH = np.ndarray(shape=(Np,Np,Np)) #create
    _, edges = np.histogramdd( (np.concatenate(seg_x), np.concatenate(seg_y), np.concatenate(seg_z)), (Np, Np, Np), density=False) #get the edges 
    #HHHLENGTHS = np.array([ [ [] for j in range(len(seg_x))] for i in range(len(seg_x)) ] ])
    for f in range(len(seg_x)):
        # 1) add sampling points to filament
        seg_x_interp = []
        seg_y_interp = []
        seg_z_interp = []
        for s in range(len(seg_x[f])-1): # loop over each segment
            seg = (seg_x[f][s:s+2], seg_y[f][s:s+2], seg_z[f][s:s+2])
            dx =  np.diff(seg[0]) / (Nsampling-1)
            dy =  np.diff(seg[1]) / (Nsampling-1)
            dz =  np.diff(seg[2]) / (Nsampling-1)
            sampling_points_x = np.array([ float(seg[0][0]+N*dx) for N in range(Nsampling)])
            sampling_points_y = np.array([ float(seg[1][0]+N*dy) for N in range(Nsampling)])
            sampling_points_z = np.array([ float(seg[2][0]+N*dz) for N in range(Nsampling)])
            #append sampling points to filament
            if s < len(seg_x[f])-2: #do not append the last point, which would be repeated in the next s value
                seg_x_interp.extend(sampling_points_x[:-1])
                seg_y_interp.extend(sampling_points_y[:-1])
                seg_z_interp.extend(sampling_points_z[:-1])
            else:
                seg_x_interp.extend(sampling_points_x)
                seg_y_interp.extend(sampling_points_y)
                seg_z_interp.extend(sampling_points_z)
        # save filament with more sampling points
        seg_x_interp = np.array(seg_x_interp)
        seg_y_interp = np.array(seg_y_interp)
        seg_z_interp = np.array(seg_z_interp)
        # 2) compute frequency map
        Hf, _ = np.histogramdd( (seg_x_interp, seg_y_interp, seg_z_interp) , edges, density=False)
        #Hf, _ = np.histogramdd( (seg_x[f], seg_y[f], seg_z[f]), edges, density=False) # without interp
        boolHf = np.bool_(Hf)
        HHH[boolHf] += 1
    return HHH, edges

#function export to array
def get_Coord_and_Proba_arrays(Hfils, edges, Np):
    #compute centers of pixels
    centers_x = (edges[0][:-1] + edges[0][1:])*0.5
    centers_y = (edges[1][:-1] + edges[1][1:])*0.5
    centers_z = (edges[2][:-1] + edges[2][1:])*0.5
    #create arrays
    fils_coord = []
    fils_proba = []
    for iz in range(Np):
        for iy in range(Np):
            for ix in range(Np):
                fils_coord.append( np.array([centers_x[ix], centers_y[iy], centers_z[iz]]) )
                fils_proba.append( Hfils[ix,iy,iz] )
    fils_coord = np.array(fils_coord)
    fils_proba = np.array(fils_proba)
    return fils_coord, fils_proba

'''
Nsampling = 200
# Read filaments
num_cells = 600
filefils = 'filament_coords_and_lengths.npy'
seg_z, seg_y, seg_x, lengths = np.load(filefils, allow_pickle=True)
x_bins = np.linspace(0.0,303.0,num_cells + 1)
y_bins = np.linspace(0.0,303.0,num_cells + 1)
z_bins = np.linspace(0.0,303.0,num_cells + 1)
fil_grid = create_frequency_map_adding_sampling_points(z_bins,y_bins,x_bins, Nsampling, seg_z, seg_y, seg_x)

box_len = 303.
for i in range(20):
    fig, ax = plt.subplots(1, figsize = (8,8))
    cmap_WHIM = mpl.cm.get_cmap('Purples')
    z_in = 15*i #in Mpc
    z_fin = 15*(i+1) #in Mpc
    gridIndexMin = int((num_cells*z_in)/box_len)
    gridIndexMax = int((num_cells*z_fin)/box_len)
    imageFilSlice = np.mean(fil_grid[gridIndexMin:gridIndexMax, :, :]*1000.,axis=0)
    ####### plotting filaments. Filaments given by Daniela ##########
    for j in range(len(seg_x)):
        if ( np.mean(seg_z[j]) > z_in ) & ( np.mean(seg_z[j]) <= z_fin ):
            fil_scat = ax.plot(seg_x[j], seg_y[j],'-',color='k')

    Fil_plot = ax.imshow(imageFilSlice, cmap=cmap_WHIM, norm=LogNorm(vmin=1e-2 , vmax=1000), extent= [0.,303.,0.,303.0], origin='lower',alpha=0.6, label="filament")
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
    fig.add_axes(cax)
    cbar = fig.colorbar(Fil_plot, cax=cax, orientation = "horizontal", format=ticker.LogFormatter())
    cbar.set_label(r'$\delta_{LD}$')#Luminosity Overdensity') #($10^{10} L_{\odot} Mpc^{-3}$)')
    cbar.ax.tick_params(labelsize=10, width=0.7, size=8)
    cbar.solids.set_edgecolor("face")
    cbar.ax.xaxis.set_label_position('bottom')
    plt.xlabel('X [Mpc]',fontsize=10)
    plt.ylabel('Y [Mpc]',fontsize=10)
    #plt.savefig('filament_volume_location_slice{i}.pdf'.format(i=i))
    plt.show()
    plt.close()
'''

filefils = 'filament_coords_and_lengths.npy'
num_cells = 600
# Read filaments
seg_z, seg_y, seg_x, lengths = np.load(filefils, allow_pickle=True)
x_bins = np.linspace(0.0,303.0,num_cells + 1)
y_bins = np.linspace(0.0,303.0,num_cells + 1)
z_bins = np.linspace(0.0,303.0,num_cells + 1)

Nsampling = 20
#fil_grid = create_frequency_map_adding_sampling_points(z_bins, y_bins, x_bins, Nsampling, seg_z, seg_y, seg_x)

#np.save('fil_grid_Daniela',fil_grid)

Np = 600
fil_grid, fil_grid_edges = create_frequency_map_adding_sampling_points_with_fil_struct(Np, Nsampling, seg_x, seg_y, seg_z)
np.save('fil_grid_Daniela_otherfunc',fil_grid)
# — main —
#Nsampling = 4
#print(' - NUMBER OF SAMPLING POINTS PER SEGMENT:', Nsampling)

#ALL FILAMENTS!! ****
#HHH, edges = create_frequency_map_adding_sampling_points(Np, Nsampling, seg_x, seg_y, seg_z)
#fils_coord, fils_proba = get_Coord_and_Proba_arrays(HHH, edges, Np)

#HHH_bool = np.bool_(HHH)
