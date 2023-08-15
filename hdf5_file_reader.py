import h5py
import illustris_python as il

dust_corrected_file = "/path/to/file/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5"
with h5py.File(dust_corrected_file, "r") as partData:
    subhalo_mag_r_dust = np.asarray(partData["/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:,2,0],dtype='f4')

path = '/path/to/directory/tng_300'

halo_fields = ['GroupPos', 'GroupMass', 'GroupMassType', 'Group_M_Crit200', 'Group_M_Crit500', 'Group_R_Crit200', 'Group_R_Crit500', 'GroupNsubs', 'GroupLenType', 'GroupLen'] #check how much of gas lies within R200, R500. See ratio of gas mass to total mass. Compare to EAGLE Toni's paper.
subhalo_fields = ['SubhaloFlag','SubhaloMass','SubhaloMassType', 'SubhaloPos', 'SubhaloStellarPhotometrics']

halos = il.groupcat.loadHalos(path,99,fields=halo_fields)
subhalos = il.groupcat.loadSubhalos(path,99,fields=subhalo_fields)    
