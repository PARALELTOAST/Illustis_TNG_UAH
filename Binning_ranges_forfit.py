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
#LD_Overden = np.load('ld_overden_blue.npy')
#LD_Overden = np.load('ld_overden_1.5smooth.npy')
LD_Overden = np.load('ld_overden.npy')

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

# plt.imshow(LD_scat)
# plt.imshow(WHIM_scat_nofilter)
# plt.show()
#
# ##################################Grid after fil applied############################
# fig,ax = plt.subplots()
# pa = ax.imshow(LD_scat[0:2,:,:].sum(0).T, origin = 'lower', extent = (0,303, 0, 303), norm = LogNorm())
# cba = plt.colorbar(pa,location = 'bottom', shrink = .65, spacing = 'uniform')
# pb = ax.imshow(WHIM_scat_nofilter[0:2,:,:].sum(0).T, origin= 'lower', extent=(0,303,0,303), norm=LogNorm())
# cbb = plt.colorbar(pb,location = 'top', shrink = .65, spacing = 'uniform')
# plt.xlabel('X [Mpc]')
# plt.ylabel('Y [Mpc]')
# cba.set_label('$\delta_{LD}$')
# cbb.set_label('$\delta_{\u03C1}$')
# pa.set_cmap('Reds')
# pb.set_cmap('PuBu')
# plt.show()
#
# quit()

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
ldlog_nofil_range = maskrange*(ldlog_nofil > -6) * (ldlog_nofil < 2.3)
ldlog_nofil = ldlog_nofil[ldlog_nofil_range]
logden_nofil = logden_nofil[ldlog_nofil_range]

#print(min(ldlog_nofil), max(ldlog_nofil))

#######################################################################################
################################# Binning the log space values ##########################
maskrange = np.ones(len(ldlog_nofil), dtype=np.bool)
bin_means = 30
mean_values = []
std_values = []
meadian_values = []
for i in range(0,bin_means):
    ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.3 / (bin_means)) * (i + 1))  # *(ldlog_nofil<2.3)
    if i > 0:
        ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.3 / (bin_means)) * (i + 1)) * (ldlog_nofil > -6 + (8.3 / (bin_means)) * (i))  # *(ldlog_nofil<2.3)
    ldlog_nofil_temp = ldlog_nofil[ldlog_nofil_range]
    logden_nofil_temp = logden_nofil[ldlog_nofil_range]
    mean_values.append(np.mean(logden_nofil_temp))
    meadian_values.append(np.median(logden_nofil_temp))
    std_values.append(np.std(logden_nofil_temp))

range_bins = np.linspace(-6, 2.3, bin_means)
print('len of mean',len(mean_values))
#print(mean_values)
print('len of range bin', len(range_bins))
#print(range_bins)
################################################Over plotting it all on the graph ############################
# ax = plt.subplots()
# plt.hist2d(ldlog_nofil,logden_nofil, bins = 100, norm = LogNorm(), alpha = .5)
# plt.plot(range_bins, mean_values, label = 'Mean value ', marker='o', color = 'r' )
# plt.errorbar(range_bins, mean_values, yerr = std_values, color = 'r')
# plt.plot(range_bins, meadian_values, label = 'Meadian value ', marker='o', color = 'b' )
# plt.errorbar(range_bins, meadian_values, yerr = std_values, color = 'b')
# xy=np.array((ldlog_nofil,logden_nofil)).T
# #unique, weight = np.unique(xy, return_counts=True)
# weight = []
# for i in range(len(xy)):
#     weight.append(np.count_nonzero(xy == xy[i]))

'''e fit'''
# def func(X,A,B,C):
#     return A * np.exp(X / B) + C
# par, cov = scipy.optimize.curve_fit(func, ldlog_nofil, logden_nofil)
# plt.plot(sorted(ldlog_nofil), func(sorted(ldlog_nofil), *par), color= 'black', label = 'Exponetial Fit')
#
# par, pov = scipy.optimize.curve_fit(func, range_bins,mean_values)
# plt.plot(range_bins, func(range_bins, *par), color = 'black', label = 'ex fit mean based', linestyle = '--')
#
# par, pov = scipy.optimize.curve_fit(func, range_bins,meadian_values)
# plt.plot(range_bins, func(range_bins, *par), color = 'm', label = 'ex fit med based', linestyle = '--')

'''poly fit'''
# x = sorted(ldlog_nofil)
# par, cov = np.polyfit(ldlog_nofil, logden_nofil, deg=2, w=weight, cov = 'unscaled')
# plt.plot(x, np.polyval(par, x), color='black', label = 'Polynomial fit')
# #print('The polynomial fit parameters:', par)
#
# par, cov = np.polyfit(range_bins, mean_values, deg=2, w=weight, cov = 'unscaled')
# plt.plot(range_bins, np.polyval(par, range_bins), color='red', label = 'Mean fit')
#
# par, cov = np.polyfit(range_bins, meadian_values, deg=2, w=weight, cov = 'unscaled')
# plt.plot(range_bins, np.polyval(par, range_bins), color='blue', label = 'polynomial fit')
#
# plt.set_cmap('viridis')
# plt.grid(which='both')
# plt.title('median and mean values with std 50 bins')
# plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
# plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
# plt.legend()
# #ax.xlim(min(ldlog_nofil),2.5)
# plt.ylim(-1.5)
# plt.colorbar()
# #plt.show()
# plt.savefig('Bin_dists/Mean_med_50_Bin')
# plt.close()
#
# quit()
np.savetxt('ldlog_nofil.txt', ldlog_nofil)

quit()
###############################################Determining the distributions of bins##################################
for i in range(0,bin_means):
    ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.5 / (bin_means)) * (i + 1))  # *(ldlog_nofil<2.3)
    if i > 0:
        ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.5 / (bin_means)) * (i + 1)) * (ldlog_nofil > -6 + (8.5 / (bin_means)) * (i))  # *(ldlog_nofil<2.3)
    ldlog_nofil_temp = ldlog_nofil[ldlog_nofil_range]
    logden_nofil_temp = logden_nofil[ldlog_nofil_range]
    mean = np.mean(logden_nofil_temp)
    std = np.std(logden_nofil_temp)
    x = plt.np.linspace(min(logden_nofil_temp), max(logden_nofil_temp), 50)
    plt.hist(logden_nofil_temp, 25, label = 'WHIM Distribution', density = True)
    plt.plot(x, scipy.stats.norm.pdf(x, mean, std), label = 'Normal Gaussian')
    plt.title('Bin Number:'+str(i+1)+'/'+str(bin_means)+' LD Range: ['+str(round(min(ldlog_nofil_temp),3))+','+str(round(max(ldlog_nofil_temp),3))+']')
    plt.legend()
    #plt.show()
    plt.savefig('Bin_dists/bin_'+str(i+1)+'.png')
    plt.close()

quit()

#################### Retreving the 68% confidence interval, by cutting the first and last 16% out#######################
lower_16 = []
upper_16 = []
for i in range(0,bin_means):
    ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.3 / (bin_means)) * (i + 1))  # *(ldlog_nofil<2.3)
    if i > 0:
        ldlog_nofil_range = maskrange * (ldlog_nofil < -6 + (8.3 / (bin_means)) * (i + 1)) * (ldlog_nofil > -6 + (8.3 / (bin_means)) * (i))  # *(ldlog_nofil<2.3)
    ldlog_nofil_temp = ldlog_nofil[ldlog_nofil_range]
    logden_nofil_temp = logden_nofil[ldlog_nofil_range]
    sort = sorted(logden_nofil_temp)
    range_per_index = len(logden_nofil_temp) * 0.16
    if range_per_index is not int:
        print('goofy ah long division')
        range_per_index = int(range_per_index)
    #lower_16.append(sort[range_per_index])
    #upper_16.append(sort[len(logden_nofil_temp)- range_per_index])
    reduced_range = sort[range_per_index:len(sort)-range_per_index]
    print(i)
    lower_16.append(min(reduced_range))
    upper_16.append(max(reduced_range))

    ##############################################Plotting the 68% confidence interval###############################
ax = plt.subplots()
plt.hist2d(ldlog_nofil,logden_nofil, bins = 100, norm = LogNorm(), alpha = .5)
#plt.plot(range_bins, mean_values, label = 'Mean value ', marker='o', color = 'r' )
#plt.errorbar(range_bins, mean_values, yerr = std_values, color = 'r')
#plt.plot(range_bins, meadian_values, label = 'Meadian value ', marker='o', color = 'b' )
#plt.errorbar(range_bins, meadian_values, yerr = std_values, color = 'b')
xy=np.array((ldlog_nofil,logden_nofil)).T
#unique, weight = np.unique(xy, return_counts=True)
weight = []
for i in range(len(xy)):
    weight.append(np.count_nonzero(xy == xy[i]))

'''e fit'''
def func(X,A,B,C):
    return A * np.exp(X / B) + C
#par, cov = scipy.optimize.curve_fit(func, ldlog_nofil, logden_nofil)
#plt.plot(sorted(ldlog_nofil), func(sorted(ldlog_nofil), *par), color= 'black', label = 'Exponetial Fit')

par, pov = scipy.optimize.curve_fit(func, range_bins, mean_values)
plt.plot(range_bins, func(range_bins, *par), color = 'black', label = 'Exponeteal Fit Mean Based')

#par, pov = scipy.optimize.curve_fit(func, range_bins,meadian_values)
#plt.plot(range_bins, func(range_bins, *par), color = 'm', label = 'ex fit med based', linestyle = '--')

'''confidence interval'''
plt.plot(range_bins, lower_16, color = 'red', label = '68% Confidence Interval', linestyle = '--')
plt.plot(range_bins, upper_16, color = 'red', linestyle = '--')

plt.set_cmap('viridis')
plt.grid(which='both')
plt.title('68% confidence interval with '+str(bin_means)+' bins')
plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
plt.legend()
plt.xlim(-6,2.5)
#ax.xlim(min(ldlog_nofil),2.5)
plt.ylim(-1.5)
plt.colorbar()
plt.show()
#plt.savefig('Bin_dists/Mean_med_50_Bin')
#plt.close()

######################################Plotting both 68% range and gthe confidence interval######################
ax = plt.subplots(111)
plt.hist2d(ldlog_nofil,logden_nofil, bins = 100, norm = LogNorm(), alpha = .5)
xy=np.array((ldlog_nofil,logden_nofil)).T
weight = []
for i in range(len(xy)):
    weight.append(np.count_nonzero(xy == xy[i]))

'''parameter con interval'''
from exponential_monte_carlo import mu_alpha
from exponential_monte_carlo import mu_A
from exponential_monte_carlo import mu_beta
from exponential_monte_carlo import conf_interval
from exponential_monte_carlo import u

def expo_fit(x,A,alpha,beta):
    return A * np.exp(alpha * x) + beta

plt.plot(u, expo_fit(u, mu_A, mu_alpha, mu_beta), color='k', label="fit")
dyn = partial(ds.tf.dynspread, max_px=40, threshold=0.5)
def datshader(ax, y, z):
    df = pd.DataFrame(dict(x=conf_interval[:, 0], y=conf_interval[:, 1]))
    da1 = dsshow(df, ds.Point('x', 'y'), aspect='auto', ax=ax)
    plt.colorbar(da1)
scat = datshader(ax, conf_interval[:, 0], conf_interval[:, 1])


'''e fit'''
def func(X,A,B,C):
    return A * np.exp(X / B) + C
par, pov = scipy.optimize.curve_fit(func, range_bins, mean_values)
plt.plot(range_bins, func(range_bins, *par), color = 'black', label = 'Exponeteal Fit Mean Based')


'''confidence interval'''
plt.plot(range_bins, lower_16, color = 'red', label = '68% Confidence Interval', linestyle = '--')
plt.plot(range_bins, upper_16, color = 'red', linestyle = '--')

plt.set_cmap('viridis')
plt.grid(which='both')
plt.title('68% confidence interval with '+str(bin_means)+' bins')
plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
plt.legend()
plt.xlim(-6,2.5)
plt.ylim(-1.5)
plt.colorbar()
plt.show()