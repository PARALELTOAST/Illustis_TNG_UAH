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

###########################Import da grids####################################################
fil_grid = np.load('fil_gri.npy')
fil_grid = np.bool_(fil_grid)

#LD_Overden = np.load('ld_overden_reds.npy')
LD_Overden = np.load('ld_overden_blue.npy')
#LD_Overden = np.load('ld_overden_1.5smooth.npy')
#LD_Overden = np.load('ld_overden.npy')

WHIM_Overden = np.load('GRID_WHIM_DENSITY_NOMASKhaloes.npy')
WHIM_Overden = WHIM_Overden /  (0.618 * 10 ** 10)

#WHIM_Overden_nofilter = np.load('GRID_WHIM_DENSITY_NOMASKhaloes.npy')
#WHIM_Overden_nofilter = WHIM_Overden_nofilter/ (0.618 * 10 ** 10)
# --- load WHIM density grid - No mask ---
WHIM_GRID0 = np.load('GRID_WHIM_DENSITY_NOMASKhaloes.npy')

# New!!!
MASK = np.load('GRID_MASKHALOES_bools_Mlim1e10_fac3R200.npy') #True if is outside halo
print(' -> % of voxels outside haloes (not masked):', MASK.sum() / 600**3 * 100)
print(' -> % of voxels outside haloes (not masked):', (MASK.sum() / 600**3 * 100) - 100)
import copy
WHIM_GRID = copy.deepcopy(WHIM_GRID0) #create a deepcopy of the unmasked grid
WHIM_GRID[ np.logical_not(MASK) ] = 0 #put 0 values for the pixels inside haloes

WHIM_Overden_nofilter = WHIM_GRID
WHIM_Overden_nofilter = WHIM_Overden_nofilter/ (0.618 * 10 ** 10)

# --- Residuals: pixels inside haloes ---
#RES = copy.deepcopy(WHIM_GRID0) #create a deepcopy of the unmasked grid
#RES[ MASK ] = 0 #0 values for the pixels OUTside  haloes
##########################over plotting whim and lum #######################
# plt.imshow(LD_Overden[0:2,:,:].sum(0).T, origin = 'lower', extent = (0,303, 0, 303), norm = LogNorm())
# plt.imshow(WHIM_Overden[0:2,:,:].sum(0).T, origin= 'lower', extent=(0,303,0,303), norm=LogNorm())
# plt.ylabel('Y [Mpc]')
# plt.xlabel('X [Mpc]')
# plt.title('$\delta_{LD}$')
# plt.colorbar()
# plt.show()

# fig,ax = plt.subplots()
# pa = ax.imshow(LD_Overden[0:2,:,:].sum(0).T, origin = 'lower', extent = (0,303, 0, 303), norm = LogNorm())
# cba = plt.colorbar(pa,location = 'left', shrink = .8, spacing = 'uniform', pad = .13)
# pb = ax.imshow(WHIM_Overden_nofilter[0:2,:,:].sum(0).T, origin= 'lower', extent=(0,303,0,303), norm=LogNorm())
# cbb = plt.colorbar(pb,location = 'right', shrink = .8, spacing = 'uniform')
# plt.xlabel('X [Mpc]')
# plt.ylabel('Y [Mpc]')
# cba.set_label('$\delta_{LD}$')
# cbb.set_label('$\delta_{\u03C1}$')
# pa.set_cmap('Reds')
# pb.set_cmap('PuBu')
# plt.show()
#
# quit()

#############################Scattering ######################################
print(WHIM_Overden.shape)
print(LD_Overden.shape)
print(fil_grid.shape)
# print('The max and min values of WHIM', max(WHIM_Overden.flatten()), min(WHIM_Overden.flatten()))
# print('The max and min values of Lum', max(LD_Overden.flatten()), min(LD_Overden.flatten()))
# print('length of whim and lum before applying the fillaments', len(WHIM_Overden.flatten()), len(LD_Overden.flatten()))

WHIM_scat = WHIM_Overden[fil_grid].flatten()
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

WHIM_scat = WHIM_scat.flatten()
WHIM_scat_nofilter = WHIM_scat_nofilter.flatten()
LD_scat = LD_scat.flatten()
ld_scat_nofilter=LD_scat
print('length of the whim and ld after the fillaments applied',len(WHIM_scat), len(LD_scat))
print('None zero indicies in the array', len(np.nonzero(WHIM_scat)), len(np.nonzero(LD_scat)))
print('The max and min values of WHIM after fil_grid', max(WHIM_scat.flatten()), min(WHIM_scat.flatten()))
print('The max and min values of Lum after fil_grid', max(LD_scat.flatten()), min(LD_scat.flatten()))

# Removing zeros so the log10 function will work
LD_scat = LD_scat[np.nonzero(WHIM_scat)]
WHIM_scat = WHIM_scat[np.nonzero(WHIM_scat)]
WHIM_scat = WHIM_scat[np.nonzero(LD_scat)]
LD_scat = LD_scat[np.nonzero(LD_scat)]

ld_scat_nofilter = ld_scat_nofilter[np.nonzero(WHIM_scat_nofilter)]
WHIM_scat_nofilter = WHIM_scat_nofilter[np.nonzero(WHIM_scat_nofilter)]
WHIM_scat_nofilter = WHIM_scat_nofilter[np.nonzero(ld_scat_nofilter)]
ld_scat_nofilter = ld_scat_nofilter[np.nonzero(ld_scat_nofilter)]
print('The max and min values of WHIM after fil_grid', max(WHIM_scat.flatten()), min(WHIM_scat.flatten()))
print('The max and min values of Lum after fil_grid', max(LD_scat.flatten()), min(LD_scat.flatten()))
logden = np.log10(WHIM_scat)
ldlog = np.log10(LD_scat)
logden_nofil = np.log10(WHIM_scat_nofilter)
ldlog_nofil = np.log10(ld_scat_nofilter)

print(min(logden), max(logden))
print(min(ldlog), max(ldlog))
print('Length of logden and ldlog: ', len(logden), len(ldlog))

# np.save('filtered3xr200_log_luminosity.npy', ldlog, True)
# np.save('filtered3xr200_WHIM_Density.npy', logden, True)




######################################LUMINOSITY VERS WHIM DENSITY IN LOG SPACE########################################
#
# fig, ax = plt.subplots(1, figsize=(8, 8))
# nbins = 40
# divider = make_axes_locatable(ax)
# bins = [np.linspace(min(ldlog), max(ldlog), nbins + 1), np.linspace(min(logden), max(logden), nbins + 1)]
# h, xedge, yedge = np.histogram2d(ldlog, logden, bins=bins)
# h = h.transpose()
#
# cmap = plt.get_cmap('Blues')
# vmin = min(h.flatten()) + 1  # added a one so I can make the colormap logarithmic.
# vmax = max(h.flatten())
# X, Y = np.meshgrid(xedge, yedge)  # X_dim and Y_dim same as xedge and yedge 41,41
# im = plt.pcolormesh(X, Y, h, cmap=cmap, edgecolors='black', norm=LogNorm(vmin=vmin, vmax=vmax), linewidth=0.3)
#
# #par, cov = np.polyfit(ldlog, logden, deg=1, cov=True)
# #par, cov = np.polyfit(h[0], h[1], deg=1, cov=True)
# #plt.plot(ldlog, ldlog * par[0] + par[1], color='red', label='Polyfit')
# #####################trying to fit based on weight which is count##############
# xy=np.array((ldlog,logden)).T
# #unique, weight = np.unique(xy, return_counts=True)
# weight = []
# for i in range(len(xy)):
#     weight.append(np.count_nonzero(xy == xy[i]))
#
# par, cov = np.polyfit(ldlog, logden, deg=1, w=weight, cov=True)
# plt.plot(ldlog, ldlog * par[0] + par[1], color='red', label='Polyfit')
# print('The fit parameters', par[0], par[1])
# plt.ylim(min(yedge),max(yedge))
# plt.xlim(min(xedge),max(xedge))
#
# plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
# plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
# plt.title('Plot for 1.2 smoothing factor excluding exterior to r200')
# cbar = plt.colorbar(mappable=im)
# cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())
# plt.legend()
# plt.show()
#################################################################################################################
maskrange= np.ones(len(ldlog_nofil), dtype = np.bool)
ldlog_nofil_range = maskrange*(ldlog_nofil > -6)#*(ldlog_nofil<2.3)
ldlog_nofil = ldlog_nofil[ldlog_nofil_range]
logden_nofil = logden_nofil[ldlog_nofil_range]
#
# maskrange_lod= np.ones(len(logden_nofil), dtype = np.bool)
# logden_nofil_range = maskrange_lod*(logden_nofil > -1.2)
# logden_nofil = logden_nofil[logden_nofil_range]
# ldlog_nofil = ldlog_nofil[logden_nofil_range]

plt.hist2d(ldlog_nofil,logden_nofil, bins = 100, norm = LogNorm())
xy=np.array((ldlog_nofil,logden_nofil)).T
#unique, weight = np.unique(xy, return_counts=True)
weight = []
for i in range(len(xy)):
    weight.append(np.count_nonzero(xy == xy[i]))


'''Linear Fit'''
# par, cov = np.polyfit(ldlog_nofil, logden_nofil, deg=1, w=weight, cov='unscaled')
# plt.plot(ldlog_nofil, ldlog_nofil * par[0] + par[1], color='red', label='Polyfit')
# print('The fit parameters', par[0], par[1])
# print('the covariance', cov)

'''polynomial Fit'''
# x = sorted(ldlog_nofil)
# par, cov = np.polyfit(ldlog_nofil, logden_nofil, deg=2, w=weight, cov = 'unscaled')
# plt.plot(x, np.polyval(par, x), color='red', label = 'polynomial fit')
# print('The polynomial fit parameters:', par, cov)

'''e fit'''
def func(X,A,B,C):
    return A * np.exp(X / B) + C
par, cov = scipy.optimize.curve_fit(func, ldlog_nofil, logden_nofil)
plt.plot(sorted(ldlog_nofil), func(sorted(ldlog_nofil), *par), color= 'black', label = 'Exponetial Fit')
print('Exp params', par, cov)

'''lmFit EXP'''
import lmfit

# def func(X,A,B,C):
#     return A * np.exp(X / B) + C

#def func(X,A,B,C):
#    return A * X ** 2 + B * X + C

emodel = lmfit.Model(func)

print('Parameter names and class:', emodel.param_names, emodel.independent_vars)
params = emodel.make_params()
params = emodel.make_params(A = 0.074, B = .439, C = -0.196, X = ldlog_nofil)

x_vals = np.linspace(min(ldlog_nofil), max(ldlog_nofil), 67872)
results = emodel.fit(logden_nofil, X = ldlog_nofil, A = 0.585, B = 1.536, C = -0.831)
print(results.fit_report())
#plt.plot(sorted(ldlog_nofil), results.best_fit, '--', label='lmfit exp')


'''power law'''
# def func(x,alpha,beta,a,x0):
#     return a + (x / x0) ** alpha + (x / x0) ** beta
# par, cov = scipy.optimize.curve_fit(func, ldlog_nofil, logden_nofil)
# plt.plot(sorted(ldlog_nofil), func(sorted(ldlog_nofil), * par), color= 'black', label = 'Power Law')


plt.set_cmap('viridis')
plt.grid(which='both')
#plt.xticks(np.arange(min(ldlog_nofil), max(ldlog_nofil), .5))
plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
plt.xlim(min(ldlog_nofil),2.5)
plt.ylim(-1.5)
plt.title('Blues')
plt.colorbar()
plt.legend()
plt.show()

####################################################################################################################
############################################NO MASKS GRAPHS################################################
# fig, ax = plt.subplots(1, figsize=(8, 8))
# nbins = 40
# divider = make_axes_locatable(ax)
# bins = [np.linspace(min(ldlog_nofil), max(ldlog_nofil), nbins + 1), np.linspace(min(logden_nofil), max(logden_nofil), nbins + 1)]
# h, xedge, yedge = np.histogram2d(ldlog_nofil, logden_nofil, bins=bins)
# h = h.transpose()
#
# cmap = plt.get_cmap('Blues')
# vmin = min(h.flatten()) + 1  # added a one so I can make the colormap logarithmic.
# vmax = max(h.flatten())
# X, Y = np.meshgrid(xedge, yedge)  # X_dim and Y_dim same as xedge and yedge 41,41
# im = plt.pcolormesh(X, Y, h, cmap=cmap, edgecolors='black', norm=LogNorm(vmin=vmin, vmax=vmax), linewidth=0.3)
#
# xy=np.array((ldlog_nofil,logden_nofil)).T
# #unique, weight = np.unique(xy, return_counts=True)
# weight = []
# for i in range(len(xy)):
#     weight.append(np.count_nonzero(xy == xy[i]))
#
# par, cov = np.polyfit(ldlog_nofil, logden_nofil, deg=1, w=weight, cov=True)
# plt.plot(ldlog_nofil, ldlog_nofil * par[0] + par[1], color='red', label='Polyfit')
# #xy=np.array((ldlog,logden)).T
# #unique, weight = np.unique(xy, return_counts=True)
# #weight = []
# #for i in range(len(xy)):
# #    weight.append(np.count_nonzero(xy == xy[i]))
#
# #par, cov = np.polyfit(ldlog, logden, deg=1, w=weight, cov=True)
# # plt.plot(ldlog, ldlog * par[0] + par[1], color='red', label='Polyfit')
# # xy=np.array((ldlog,logden)).T
# # #unique, weight = np.unique(xy, return_counts=True)
# # weight = []
# # for i in range(len(xy)):
# #     weight.append(np.count_nonzero(xy == xy[i]))
# #
# # par, cov = np.polyfit(ldlog, logden, deg=1, w=weight, cov=True)
# # plt.plot(ldlog, ldlog * par[0] + par[1], color='red', label='Polyfit')
#
# plt.ylim(min(yedge),max(yedge))
# plt.xlim(min(xedge),max(xedge))
# print('The fit parameters', par[0], par[1])
# plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
# plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
# plt.title('Plot for 1.2 smoothing factor excluding exterior to r200')
# cbar = plt.colorbar(mappable=im)
# cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())
# plt.legend()
# plt.show()
#################################################################################################################
# maskrange= np.ones(len(ldlog_nofil), dtype = np.bool)
# #ldlog_nofil_range = maskrange*(ldlog_nofil > 0)*(ldlog_nofil<5)
# ldlog_nofil = ldlog_nofil[ldlog_nofil_range]
# logden_nofil = logden_nofil[ldlog_nofil_range]
#
# #0.8239518405418181 -0.32252274080155335
#
# plt.hist2d(ldlog_nofil,logden_nofil, bins = 80, norm = LogNorm())
# #par, cov = np.polyfit(ldlog_nofil, logden_nofil, deg=1, cov=True)
# #plt.plot(ldlog_nofil, ldlog_nofil * par[0] + par[1], color='red', label='Polyfit')
# xy=np.array((ldlog_nofil,logden_nofil)).T
# #unique, weight = np.unique(xy, return_counts=True)
# weight = []
# for i in range(len(xy)):
#     weight.append(np.count_nonzero(xy == xy[i]))

# par, cov = np.polyfit(ldlog_nofil, logden_nofil, deg=1, w=weight, cov=True)
# plt.plot(ldlog_nofil, ldlog_nofil * par[0] + par[1], color='red', label='Polyfit')
# plt.colorbar()
# print('The fit parameters and covariance', par[0], par[1], cov)
# plt.show()/