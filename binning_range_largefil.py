import numpy as np


###########################Import da grids####################################################
#fil_grid = np.load('fil_grid_5MPC.npy')
#fil_grid = np.load('fil_gri.npy')
fil_grid = np.load('fil_grid_1MPC.npy')
#fil_grid = np.load('fil_grid_2MPC.npy')
#fil_grid = np.load('fil_grid_3MPC.npy')
fil_grid = np.bool_(fil_grid)
if np.any(np.isnan(fil_grid)) is True:
    print('There is nan in the fil grid')

#LD_Overden = np.load('ld_overden_5smoothing.npy')
#LD_Overden = np.load('ld_overden.npy')
LD_Overden = np.load('ld_overden_1_5smoothing.npy')
#LD_Overden = np.load('ld_overden_2smoothing.npy')
#LD_Overden = np.load('ld_overden_2_5smoothing.npy')
#LD_Overden = np.load('ld_overden_3smoothing.npy')
if np.any(np.isnan(LD_Overden)) is True:
    print('There is nan in the LD grid')

# --- load WHIM density grid - No mask ---
WHIM_GRID0 = np.load('GRID_WHIM_DENSITY_NOMASKhaloes.npy')

# New!!!
MASK = np.load('GRID_MASKHALOES_bools_Mlim1e10_fac3R200.npy') #True if is outside halo
print(' -> % of voxels outside haloes (not masked):', MASK.sum() / 600**3 * 100)
import copy
WHIM_GRID = copy.deepcopy(WHIM_GRID0) #create a deepcopy of the unmasked grid
WHIM_GRID[ np.logical_not(MASK) ] = 0 #put 0 values for the pixels inside haloes

WHIM_Overden_nofilter = WHIM_GRID
WHIM_Overden_nofilter = WHIM_Overden_nofilter/ (0.618 * 10 ** 10)

#############################Scattering ######################################
#print(WHIM_Overden.shape)
print(LD_Overden.shape)
print(fil_grid.shape)
# print('The max and min values of WHIM', max(WHIM_Overden.flatten()), min(WHIM_Overden.flatten()))
# print('The max and min values of Lum', max(LD_Overden.flatten()), min(LD_Overden.flatten()))
# print('length of whim and lum before applying the fillaments', len(WHIM_Overden.flatten()), len(LD_Overden.flatten()))

#WHIM_scat = WHIM_Overden[fil_grid].flatten()
LD_scat = LD_Overden[fil_grid].flatten()
WHIM_scat_nofilter = WHIM_Overden_nofilter[fil_grid].flatten()



#WHIM_scat = WHIM_scat.flatten()
WHIM_scat_nofilter = WHIM_scat_nofilter.flatten()
LD_scat = LD_scat.flatten()
ld_scat_nofilter=LD_scat
#print('length of the whim and ld after the fillaments applied',len(WHIM_scat), len(LD_scat))
#print('None zero indicies in the array', len(np.nonzero(WHIM_scat)), len(np.nonzero(LD_scat)))
#print('The max and min values of WHIM after fil_grid', max(WHIM_scat.flatten()), min(WHIM_scat.flatten()))
print('The max and min values of Lum after fil_grid', max(LD_scat.flatten()), min(LD_scat.flatten()))

# Removing zeros so the log10 function will work
#LD_scat = LD_scat[np.nonzero(WHIM_scat)]
#WHIM_scat = WHIM_scat[np.nonzero(WHIM_scat)]
#WHIM_scat = WHIM_scat[np.nonzero(LD_scat)]
LD_scat = LD_scat[np.nonzero(LD_scat)]

ld_scat_nofilter = ld_scat_nofilter[np.nonzero(WHIM_scat_nofilter)]
WHIM_scat_nofilter = WHIM_scat_nofilter[np.nonzero(WHIM_scat_nofilter)]
WHIM_scat_nofilter = WHIM_scat_nofilter[np.nonzero(ld_scat_nofilter)]
ld_scat_nofilter = ld_scat_nofilter[np.nonzero(ld_scat_nofilter)]

logden_nofil = np.log10(WHIM_scat_nofilter)
ldlog_nofil = np.log10(ld_scat_nofilter)

#print(min(logden), max(logden))
print('min and max of ldlog', min(ldlog_nofil), max(ldlog_nofil))
#print('Length of logden and ldlog: ', len(logden), len(ldlog))

maskrange= np.ones(len(ldlog_nofil), dtype = np.bool)
ldlog_nofil_range = maskrange*(ldlog_nofil > -6) #* (ldlog_nofil < 2.3)
ldlog_nofil = ldlog_nofil[ldlog_nofil_range]
logden_nofil = logden_nofil[ldlog_nofil_range]

#print(min(ldlog_nofil), max(ldlog_nofil))

#######################################################################################
################################# Binning the log space values for the blues ##########################
maskrange = np.ones(len(ldlog_nofil), dtype=np.bool)
bin_means = 30
med_LD = []
mu_LD = []
std_LD = []
mean_values = []
std_values = []
meadian_values = []
for i in range(0,bin_means):
    ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.81 / (bin_means)) * (i + 1))  # *(ldlog_nofil<2.3)
    if i > 0:
        ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.81 / (bin_means)) * (i + 1)) * (ldlog_nofil > -6 + (8.81 / (bin_means)) * (i))  # *(ldlog_nofil<2.3)
    ldlog_nofil_temp = ldlog_nofil[ldlog_nofil_range]
    logden_nofil_temp = logden_nofil[ldlog_nofil_range]
    mean_values.append(np.mean(logden_nofil_temp))
    meadian_values.append(np.median(logden_nofil_temp))
    std_values.append(np.std(logden_nofil_temp))
    med_LD.append(np.median(ldlog_nofil_temp))
    mu_LD.append(np.mean(ldlog_nofil_temp))
    std_LD.append(np.std(ldlog_nofil_temp))
print(mu_LD)


#.5mpc#############################
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_5MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a5_5MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a5_5MPC.txt', std_LD)

# #1mpc############################
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_1MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a5_1MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a5_1MPC.txt', std_LD)

# #2##########################################
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_2MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a5_2MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a5_2MPC.txt', std_LD)
#
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a1_2MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a1_2MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a1_2MPC.txt', std_LD)
#
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a15_2MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a15_2MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a15_2MPC.txt', std_LD)
#
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a2_2MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a2_2MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a2_2MPC.txt', std_LD)
#
# #3########################################
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_3MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a5_3MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a5_3MPC.txt', std_LD)
#
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a1_3MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a1_3MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a1_3MPC.txt', std_LD)
#
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a15_3MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a15_3MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a15_3MPC.txt', std_LD)
#
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a2_3MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a2_3MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a2_3MPC.txt', std_LD)
#
# np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a3_3MPC.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a3_3MPC.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a3_3MPC.txt', std_LD)


np.savetxt('scriptforexponentialmodel/largefilinfo/mu_ld_a15_1MPC.txt', mu_LD)
np.savetxt('scriptforexponentialmodel/largefilinfo/med_ld_a15_1MPC.txt', med_LD)
np.savetxt('scriptforexponentialmodel/largefilinfo/std_ld_a15_1PC.txt', std_LD)
print('saved!')