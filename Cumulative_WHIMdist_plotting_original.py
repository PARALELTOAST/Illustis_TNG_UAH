#plotting the Cumulative WHIM mass distribution as a function of LD
ax = plt.subplot(111)

######### loading in filament grid #############################
#fil_grid = np.load('fil_grid_Patrick_rad3Mpc.npy')
#fil_grid = np.load('fil_grid_Daniela.npy')
#lum_avg_den = 48923580.28441872

WHIM_den = np.load('GRID_WHIM_DENSITY_NOMASKhaloes.npy')
LD_Overden = np.load('LD_Overden_grid_smooth=1.2.npy')
MASK_halo_Mlim1e10 = np.load('GRID_MASKHALOES_bools_Mlim1e10_fac3R200.npy') #Apply masks to LD?
#MASK_halo_Mlim1e12 = np.load('GRID_MASKHALOES_bools_Mlim1e12_fac3R200.npy')

WHIM_den[ np.logical_not(MASK_halo_Mlim1e10) ] = 0
WHIM_den = np.transpose(WHIM_den, (2,1,0))
#WHIM_den = WHIM_den[fil_grid]
#LD_Overden = LD_Overden[fil_grid]
LD_Overden = LD_Overden[np.nonzero(WHIM_den)]
WHIM_den = WHIM_den[np.nonzero(WHIM_den)]
WHIM_den = WHIM_den[np.nonzero(LD_Overden)]
LD_Overden = LD_Overden[np.nonzero(LD_Overden)]
LD_Overden = np.log10(LD_Overden) #taking the log of LD

LD_Overden = np.sort(LD_Overden)
LD_index = np.argsort(LD_Overden)
WHIM_mass_total = np.sum(WHIM_den) * (303./600)**3

lower_LD_bound = 0.025 #* WHIM_mass_total
upper_LD_bound = 1.0 - lower_LD_bound #WHIM_mass_total - lower_LD_bound
Cumulative_WHIM_mass = np.cumsum(WHIM_den[LD_index] * (303./600)**3) / WHIM_mass_total

#saving cumulative distribution 
#np.save('Cumulative_WHIM_mass_Dany',Cumulative_WHIM_mass)
#np.save('Cumulative_WHIM_mass_rad30',Cumulative_WHIM_mass)
#np.save('LD_Overden_rad30',LD_Overden)
#np.save('LD_Overden_Dany', LD_Overden)

#'''
#loading other saved cumulutaive distribution values for other filament radii. Other distributions created using this script.
Cumulative_WHIM_mass_Dany = np.load('Cumulative_WHIM_mass_Dany.npy')
Cumulative_WHIM_mass_rad05 = np.load('Cumulative_WHIM_mass_rad05.npy')
Cumulative_WHIM_mass_rad10 = np.load('Cumulative_WHIM_mass_rad10.npy')
Cumulative_WHIM_mass_rad20 = np.load('Cumulative_WHIM_mass_rad20.npy')
Cumulative_WHIM_mass_rad30 = np.load('Cumulative_WHIM_mass_rad30.npy')
LD_Overden_rad05 = np.load('LD_Overden_rad05.npy')
LD_Overden_rad10 = np.load('LD_Overden_rad10.npy')
LD_Overden_rad20 = np.load('LD_Overden_rad20.npy')
LD_Overden_rad30 = np.load('LD_Overden_rad30.npy')
LD_Overden_Dany = np.load('LD_Overden_Dany.npy')
#'''

Low_LD_vals = np.asarray([i for i in range(len(LD_Overden)) if Cumulative_WHIM_mass[i] < lower_LD_bound])
High_LD_vals = np.asarray([i for i in range(len(LD_Overden)) if Cumulative_WHIM_mass[i] > upper_LD_bound])
LD_low = LD_Overden[np.size(Low_LD_vals) - 1]
LD_high = LD_Overden[min(High_LD_vals)]

print('low and high bounds of 95 percent mass range:',LD_low, LD_high)
del Low_LD_vals
del High_LD_vals
gc.collect()

ax.plot(LD_Overden, Cumulative_WHIM_mass, 'k',linewidth=2, label='no fil mask')
ax.plot(LD_Overden_Dany, Cumulative_WHIM_mass_Dany, 'm', linewidth=2, label='Danys fils')
ax.plot(LD_Overden_rad05, Cumulative_WHIM_mass_rad05, 'r', linewidth=2, label='filrad 0.5 Mpc')
ax.plot(LD_Overden_rad10, Cumulative_WHIM_mass_rad10, 'c', linewidth=2, label='filrad 1 Mpc')
ax.plot(LD_Overden_rad20, Cumulative_WHIM_mass_rad20, 'b', linewidth=2, label='filrad 2 Mpc')
ax.plot(LD_Overden_rad30, Cumulative_WHIM_mass_rad30, 'y', linewidth=2, label='filrad 3 Mpc')
ax.axvline(x=LD_low,color='g', linestyle='--', label = '95 percent')
ax.axvline(x=LD_high, color='g', linestyle='--')
plt.xlabel(r'$\log \delta_{LD}$',fontsize=15)
plt.ylabel('Cumulative WHIM',fontsize=15)
plt.legend(loc=0,fontsize=12)
plt.grid(True)
#plt.savefig('Cumulative_WHIMmass_dist_within_fil_3Mpc.pdf')
#plt.savefig('Cumulative_WHIMmass_dist_within_fil_Daniela.pdf')
plt.savefig('Cumulative_WHIMmass_dist_overplots.pdf')
plt.show()
plt.close()

