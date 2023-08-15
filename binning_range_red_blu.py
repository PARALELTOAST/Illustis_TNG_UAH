import numpy as np
import tqdm
import scipy
import scipy.stats
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
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import random
from functools import partial
import datashader as ds
from datashader.mpl_ext import dsshow

###########################Import da grids####################################################
fil_grid = np.load('fil_gri.npy')
fil_grid = np.bool_(fil_grid)

#LD_Overden = np.load('ld_overden_reds.npy')
LD_Overden = np.load('ld_overden_blue.npy')
#LD_Overden = np.load('ld_overden_1.5smooth.npy')
#LD_Overden = np.load('ld_overden.npy')

#WHIM_Overden = np.load('GRID_WHIM_DENSITY_maskhaloesMlim1e10.npy')
#WHIM_Overden = WHIM_Overden /  (0.618 * 10 ** 10)

#WHIM_Overden_nofilter = np.load('GRID_WHIM_DENSITY_NOMASKhaloes.npy')
#WHIM_Overden_nofilter = WHIM_Overden_nofilter/ (0.618 * 10 ** 10)
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
#print(min(ldlog), max(ldlog))
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
    ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.5 / (bin_means)) * (i + 1))  # *(ldlog_nofil<2.3)
    if i > 0:
        ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.5 / (bin_means)) * (i + 1)) * (ldlog_nofil > -6 + (8.5 / (bin_means)) * (i))  # *(ldlog_nofil<2.3)
    ldlog_nofil_temp = ldlog_nofil[ldlog_nofil_range]
    logden_nofil_temp = logden_nofil[ldlog_nofil_range]
    mean_values.append(np.mean(logden_nofil_temp))
    meadian_values.append(np.median(logden_nofil_temp))
    std_values.append(np.std(logden_nofil_temp))
    med_LD.append(np.median(ldlog_nofil_temp))
    mu_LD.append(np.mean(ldlog_nofil_temp))
    std_LD.append(np.std(ldlog_nofil_temp))

# np.savetxt('scriptforexponentialmodel/mu_ld_blue.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/med_ld_blue.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/std_ld_blue.txt', std_LD)

# np.savetxt('scriptforexponentialmodel/mu_ld_red.txt', mu_LD)
# np.savetxt('scriptforexponentialmodel/med_ld_red.txt', med_LD)
# np.savetxt('scriptforexponentialmodel/std_ld_red.txt', std_LD)
range_bins = np.linspace(-6, 2.3, bin_means)

################################ Saving the bins ################################################
for i in range(0,bin_means):
    ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.5 / (bin_means)) * (i + 1))  # *(ldlog_nofil<2.3)
    if i > 0:
        ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.5 / (bin_means)) * (i + 1)) * (ldlog_nofil > -6 + (8.5 / (bin_means)) * (i))  # *(ldlog_nofil<2.3)
    ldlog_nofil_temp = ldlog_nofil[ldlog_nofil_range]
    logden_nofil_temp = logden_nofil[ldlog_nofil_range]
    mean = np.mean(logden_nofil_temp)
    std = np.std(logden_nofil_temp)
    median = np.median(logden_nofil_temp)
    np.save('mean_blue_bin' + str(i+1)+'.npy',mean)
    np.save('med_blue_bin' + str(i+1)+'.npy', median)
    np.save('std_blue_bin' + str(i+1)+'.npy', std)
    np.savetxt('logden_blue_bin' + str(i+1) +'.txt', logden_nofil_temp)
    # x = plt.np.linspace(min(logden_nofil_temp), max(logden_nofil_temp), 50)
    # plt.hist(logden_nofil_temp, 25, label = 'WHIM Distribution', density = True)
    # plt.plot(x, scipy.stats.norm.pdf(x, mean, std), label = 'Normal Gaussian')
    # plt.title('Bin Number:'+str(i+1)+'/'+str(bin_means)+' LD Range: ['+str(round(min(ldlog_nofil_temp),3))+','+str(round(max(ldlog_nofil_temp),3))+']')
    # plt.legend()
    # #plt.show()
    # #plt.savefig('blue_dis_bins/bin_'+str(i+1)+'.png')
    # plt.close()

#################### Retreving the 68% confidence interval, by cutting the first and last 16% out#######################
# lower_16 = []
# upper_16 = []
# for i in range(0,bin_means):
#     ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.3 / (bin_means)) * (i + 1))  # *(ldlog_nofil<2.3)
#     if i > 0:
#         ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.3 / (bin_means)) * (i + 1)) * (ldlog_nofil > -6 + (8.3 / (bin_means)) * (i))  # *(ldlog_nofil<2.3)
#     ldlog_nofil_temp = ldlog_nofil[ldlog_nofil_range]
#     logden_nofil_temp = logden_nofil[ldlog_nofil_range]
#     sort = sorted(logden_nofil_temp)
#     range_per_index = len(logden_nofil_temp) * 0.16
#     if range_per_index is not int:
#         print('goofy ah long division')
#         range_per_index = int(range_per_index)
#     #lower_16.append(sort[range_per_index])
#     #upper_16.append(sort[len(logden_nofil_temp)- range_per_index])
#     reduced_range = sort[range_per_index:len(sort)-range_per_index]
#     print(i)
#     lower_16.append(min(reduced_range))
#     upper_16.append(max(reduced_range))
#
# ###############################plotting the confidence interval on the scatter ####################################
# ax = plt.subplots()
# plt.hist2d(ldlog_nofil,logden_nofil, bins = 100, norm = LogNorm(), alpha = .5)
#
# '''confidence interval'''
# plt.plot(range_bins, lower_16, color = 'red', label = '68% Confidence Interval', linestyle = '--')
# plt.plot(range_bins, upper_16, color = 'red', linestyle = '--')
# plt.title('Red Population')
# plt.set_cmap('viridis')
# plt.grid(which='both')
# plt.title('68% confidence interval with '+str(bin_means)+' bins')
# plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
# plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
# plt.legend()
# plt.xlim(-6,2.5)
# #ax.xlim(min(ldlog_nofil),2.5)
# plt.ylim(-1.5)
# plt.colorbar()
# plt.show()

