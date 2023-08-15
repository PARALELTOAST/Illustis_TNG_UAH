import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd

#plotting the Cumulative WHIM mass distribution as a function of LD
ax = plt.subplot(111)

######### loading in filament grid #############################
#fil_grid = np.load('fil_grid_Patrick_rad3Mpc.npy')
#fil_grid = np.load('fil_grid_Daniela.npy')
#lum_avg_den = 48923580.28441872

fil_grid = np.load('fil_gri.npy')
#fil_grid = np.load('fil_grid_viapat.npy')
#fil_grid = np.load('fil_grid_broke.npy')
fil_grid = np.bool_(fil_grid)

WHIM_den = np.load('GRID_WHIM_DENSITY_NOMASKhaloes.npy')
LD_Overden = np.load('ld_overden.npy')
MASK_halo_Mlim1e10 = np.load('GRID_MASKHALOES_bools_Mlim1e10_fac3R200.npy') #Apply masks to LD?
#MASK_halo_Mlim1e12 = np.load('GRID_MASKHALOES_bools_Mlim1e12_fac3R200.npy')

WHIM_den[ np.logical_not(MASK_halo_Mlim1e10) ] = 0

print(len(WHIM_den.flatten()))
print(len(LD_Overden.flatten()))
#WHIM_den = WHIM_den[fil_grid]
#LD_Overden = LD_Overden[fil_grid]

LD_Overden = LD_Overden[np.nonzero(WHIM_den)]
WHIM_den = WHIM_den[np.nonzero(WHIM_den)]
WHIM_den = WHIM_den[np.nonzero(LD_Overden)]
LD_Overden = LD_Overden[np.nonzero(LD_Overden)]
LD_Overden = np.log10(LD_Overden) #taking the log of LD

LD_Overden = np.sort(LD_Overden)
LD_index = np.argsort(LD_Overden)
#print(LD_index_sorted)
#print(len(LD_Overden))
#LD_Overden = LD_Overden[LD_index_sorted]
#WHIM_den = WHIM_den[LD_index_sorted]



WHIM_mass_total = np.sum(WHIM_den) * (303./600)**3

whim_mass = WHIM_den * ((303. / 600) **3)

lower_LD_bound = 0.025 #* WHIM_mass_total
upper_LD_bound = 1.0 - lower_LD_bound #WHIM_mass_total - lower_LD_bound
Cumulative_WHIM_mass = np.cumsum(WHIM_den[LD_index] * (303./600)**3) / WHIM_mass_total

#Cumulative_WHIM_volume = np.cumsum(WHIM_den[LD_index] * WHIM_mass_total) / ((300./600) **3)
#Cumulative_WHIM_volume = np.cumsum(whim_mass[LD_index_sorted] / WHIM_den[LD_index_sorted] ) / np.sum(whim_mass[LD_index_sorted] / WHIM_den[LD_index_sorted]) #3616631.8579419437#((303./600) **3)
#print(Cumulative_WHIM_volume)

#quit()

# np.save('LD_vales_full', LD_Overden)
# np.save('LD_index_full', LD_index)
# quit()
#saving cumulative distribution
#np.save('Cumulative_WHIM_mass_Dany',Cumulative_WHIM_mass)
#np.save('Cumulative_WHIM_mass_rad30',Cumulative_WHIM_mass)
#np.save('LD_Overden_rad30',LD_Overden)
#np.save('LD_Overden_Dany', LD_Overden)

#np.save('cumsum_WHIM_fillim',Cumulative_WHIM_mass)
#np.save('LD_fil_lim',LD_Overden)
np.save('cumsum_WHIM_all',Cumulative_WHIM_mass)
np.save('LD_all',LD_Overden)

#np.save('vol_cumulative_fil',Cumulative_WHIM_volume)
#np.save('val_cumulative', Cumulative_WHIM_volume)

#quit()

#'''
#loading other saved cumulutaive distribution values for other filament radii. Other distributions created using this script.
cumsum_WHIM_fil = np.load('cumsum_WHIM_fillim.npy')
LD_fil_lim = np.load('LD_fil_lim.npy')
LD_all = np.load('LD_all.npy')
cumsum_WHIM_all = np.load('cumsum_WHIM_all.npy')



Low_LD_vals = np.asarray([i for i in range(len(LD_Overden)) if Cumulative_WHIM_mass[i] < lower_LD_bound])
High_LD_vals = np.asarray([i for i in range(len(LD_Overden)) if Cumulative_WHIM_mass[i] > upper_LD_bound])
LD_low = LD_Overden[np.size(Low_LD_vals) - 1]
LD_high = LD_Overden[min(High_LD_vals)]

print('low and high bounds of 95 percent mass range:',LD_low, LD_high)
del Low_LD_vals
del High_LD_vals
gc.collect()

ax.plot(LD_all, cumsum_WHIM_all, 'Black',linewidth=2, label='Full Simulation')
#ax.plot(LD_Overden_Dany, Cumulative_WHIM_mass_Dany, 'm', linewidth=2, label='Danys fils')
ax.plot(LD_fil_lim, cumsum_WHIM_fil, 'magenta', linewidth=2, label='Inside Filaments r=1Mpc')
#ax.plot(LD_blue, cumsum_WHIM_blue, 'blue', linewidth=2, label='All Blue')
#ax.plot(LD_blue_fil, cumsum_WHIM_blue_fil, 'green', linewidth=2, label='Blue Inside Filaments r=1Mpc')
#ax.plot(LD_red, cumsum_WHIM_red, 'red', linewidth=2, label='All red')
#ax.plot(LD_red_fil, cumsum_WHIM_red_fil, 'yellow', linewidth=2, label='Red Inside Filaments r=1Mpc')

ax.axvline(x=LD_low,color='r', linestyle='--', label = '95% Mass Range (Fil)')
ax.axvline(x=LD_high, color='r', linestyle='--')
plt.xlabel(r'$\log \delta_{LD}$',fontsize=15)
plt.ylabel('WHIM Cumulative Mass Distribution',fontsize=15)
plt.legend(loc=0,fontsize=12)
plt.grid(True)
#plt.savefig('Cumulative_WHIMmass_dist_within_fil_3Mpc.pdf')
#plt.savefig('Cumulative_WHIMmass_dist_within_fil_Daniela.pdf')
#plt.savefig('Cumulative_WHIMmass_dist_overplots.pdf')
plt.show()





# # ################################################################################################
# # ###############################Voloume CD#######################################################
# # #######################################################################################
# ax = plt.subplot(111)
# # #Low_LD_vals = np.asarray([i for i in range(len(LD_Overden)) if Cumulative_WHIM_volume[i] < lower_LD_bound])
# # #High_LD_vals = np.asarray([i for i in range(len(LD_Overden)) if Cumulative_WHIM_volume[i] > upper_LD_bound])
# # #LD_low = LD_Overden[np.size(Low_LD_vals) - 1]
# # #LD_high = LD_Overden[min(High_LD_vals)]
# #
# # print('low and high bounds of 95 percent mass range:',LD_low, LD_high)
# # #del Low_LD_vals
# # #del High_LD_vals
# # gc.collect()
# #
# #ax.plot(LD_all, vol_cumu, 'Black',linewidth=2, label='Full Simulation')
# #ax.plot(LD_Overden_Dany, Cumulative_WHIM_mass_Dany, 'm', linewidth=2, label='Danys fils')
# ax.plot(LD_Overden, Cumulative_WHIM_volume, 'red', linewidth=2, label='Inside Filaments r=1Mpc')
# #ax.plot(LD_blue, cumsum_WHIM_blue, 'blue', linewidth=2, label='All Blue')
# #ax.plot(LD_blue_fil, cumsum_WHIM_blue_fil, 'green', linewidth=2, label='Blue Inside Filaments r=1Mpc')
# #ax.plot(LD_red, cumsum_WHIM_red, 'red', linewidth=2, label='All red')
# #ax.plot(LD_red_fil, cumsum_WHIM_red_fil, 'yellow', linewidth=2, label='Red Inside Filaments r=1Mpc')
#
# ax.axvline(x=LD_low,color='r', linestyle='--', label = '95% Vol Range (Fil)')
# ax.axvline(x=LD_high, color='r', linestyle='--')
# plt.xlabel(r'$\log \delta_{LD}$',fontsize=15)
# plt.ylabel('WHIM Cumulative Volume Distribution',fontsize=15)
# plt.legend(loc=0,fontsize=12)
# plt.grid(True)
# #plt.savefig('Cumulative_WHIMmass_dist_within_fil_3Mpc.pdf')
# #plt.savefig('Cumulative_WHIMmass_dist_within_fil_Daniela.pdf')
# #plt.savefig('Cumulative_WHIMmass_dist_overplots.pdf')
# plt.show()
#
# LD_Overden_full = np.load('LD_vales_full.npy')
# LD_index_full = np.load('LD_index_full.npy')
# LD_index = np.sort(LD_index_sorted)
# #plt.plot(LD_Overden_full,  LD_index_full/ len(LD_index_full), color = 'black', label = 'Full Sim')
# plt.plot(LD_Overden, LD_index / len(LD_index), color = 'red', label = 'Fil')
# plt.legend()
# plt.xlim(0,500)
# plt.grid()
# plt.xlabel(r'$\log \delta_{LD}$',fontsize=15)
# plt.ylabel('WHIM Cumulative Volume Distribution',fontsize=15)
# plt.show()