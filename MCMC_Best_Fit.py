import math
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import tqdm
import random

# Loading in filtered, log WHIM and Lum data for r200 and 1mpc filament radius
###########################Import da grids####################################################
fil_grid = np.load('fil_gri.npy')
fil_grid = np.bool_(fil_grid)
LD_Overden = np.load('ld_overden.npy')
WHIM_Overden = np.load('GRID_WHIM_DENSITY_maskhaloesMlim1e10.npy')
WHIM_Overden = WHIM_Overden /  (0.618 * 10 ** 10)

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

WHIM_scat = WHIM_Overden[fil_grid].flatten()
LD_scat = LD_Overden[fil_grid].flatten()
WHIM_scat_nofilter = WHIM_Overden_nofilter[fil_grid].flatten()
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

maskrange= np.ones(len(ldlog_nofil), dtype = np.bool)
ldlog_nofil_range = maskrange*(ldlog_nofil > 0)*(ldlog_nofil<2.3)
ldlog_nofil = ldlog_nofil[ldlog_nofil_range]
logden_nofil = logden_nofil[ldlog_nofil_range]

# maskrange_lod= np.ones(len(logden_nofil), dtype = np.bool)
# logden_nofil_range = maskrange_lod*(logden_nofil > -1.2)
# logden_nofil = logden_nofil[logden_nofil_range]
# ldlog_nofil = ldlog_nofil[logden_nofil_range]

x = ldlog_nofil
y = logden_nofil

# Using the scipy linear regression and the polyfit, as to compare to the MCMC
parameters = scipy.stats.linregress(x, y)
print('This is the results of the scipy linear regression:', parameters)

a, b = np.polyfit(x, y, 1)
print('The parameters from the np polyfit:', a, b)
#######################################################################################################################
# Starting the MCMC-MH
iterations = 10000
# These values for a and b come from the attempted fit for info_inside_filaments.py
a_start = 3  # .08103
b_start = 3  # 1.9214
a_step = .01
b_step = .01
A = [a_start]
B = [b_start]
a_current = a_start
b_current = b_start
acpt_count = 0
step_count = [0]
var = np.var(y)
# The MCMC
for i in tqdm.trange(0, iterations):
    # proposals are chosen from a uniform distribution
    a_proposal = random.uniform(a_current - .01, a_current + .01)
    b_proposal = random.uniform(b_current - .01, b_current + .01)
    # calculating the sample variance for the chi squared values, there are 2 free parameters
    # using the sample variance ASSUMES that this model is the CORRECT model
    # s_current = []
    # s_proposal = []
    # for k in range(len(x)):
    #    s_current.append((y[k] - (a_current * x[k] + b_current)) ** 2)
    #    s_proposal.append((y[k] - (a_proposal * x[k] + b_proposal)) ** 2)
    # s_current = (1 / (len(y) - 2)) * np.sum(s_current)
    # s_proposal = (1 / (len(y) - 2)) * np.sum(s_proposal)
    # calculating chi squared values for the current and proposal
    chi_current = []
    chi_proposal = []
    for j in range(len(x)):
        chi_current.append((y[j] - (a_current * x[j] + b_current)) ** 2 / var)
        chi_proposal.append((y[j] - (a_proposal * x[j] + b_proposal)) ** 2 / var)
    # calculating alpha, the acceptance probability
    chi_current = np.sum(chi_current)
    chi_proposal = np.sum(chi_proposal)
    alpha = min(np.exp((chi_current - chi_proposal) / 2), 1)
    # if alpha = 1, 100% acceptance rate
    if alpha == 1:
        a_current = a_proposal
        b_current = b_proposal
        acpt_count += 1
    else:
        # draw a random number from uniform distribution if less than alpha the proposals are accepted
        u = random.uniform(0, 1)
        if u <= alpha:
            a_current = a_proposal
            b_current = b_proposal
            acpt_count += 1
    A.append(a_current)
    B.append(b_current)
    step_count.append(i)
print('Number of accepted values:', acpt_count)
print('percentage of accepted values:', (acpt_count / 10000) * 100)
print(len(step_count), len(A), len(B))

print('\nuntrimmed median A:', np.median(A))
print('untrimmed median B:', np.median(B))
print('Trimmed median A:', np.median(A[2000:]))
print('Trimmed median B:', np.median(B[2000:]))

print('\nUntrimmed std A:', np.std(A))
print('Untrimmed std B:', np.std(B))
print('Trimmed std A:', np.std(A[2000:]))
print('Trimmed std B:', np.std(B[2000:]))

print('trimmed mean a', np.mean(A[2000:]))
print('trimmed mean b', np.mean(B[2000:]))

print('\nUntrimmed confidence level A: ' + str(np.mean(A) - np.std(A)) + '<= a <=' + str(np.mean(A) + np.std(A)))
print('Untrimmed confidence level B: ' + str(np.mean(B) - np.std(B)) + '<= a <=' + str(np.mean(B) + np.std(B)))
print('Trimmed confidence level A: ' + str(np.mean(A[2000:]) - np.std(A[2000:])) + '<= a <=' + str(
    np.mean(A[2000:]) + np.std(A[2000:])))
print('Trimmed confidence level B: ' + str(np.mean(B[2000:]) - np.std(B[2000:])) + '<= b <= ' + str(
    np.mean(B[2000:]) + np.std(B[2000:])))

# Plotting the behavior of both a and b parameters.
# parameter A
plt.scatter(step_count, A, linewidths=.35)
plt.grid()
plt.xlabel('Step Number')
plt.ylabel('Paremeter a Value')
#plt.title('MCMC-MH For Parameter A')
plt.show()
# parameter B
plt.scatter(step_count, B, linewidths=.35)
plt.grid()
plt.xlabel('Step Number')
plt.ylabel('Paremeter b Value')
#plt.title('MCMC-MH For Parameter B')
plt.show()

# histogram of parmeter a
plt.hist(A, 50, histtype='step', color='green', label='Parameter A', density=True)
plt.legend()
plt.xlabel('Parameter a')
plt.ylabel('Distribution')
#plt.title('Parameter a histogram')
plt.grid()
plt.show()

# histogram of parameter B
plt.hist(B, 50, histtype='step', color='red', label='Parameter B', density=True)
plt.legend()
plt.xlabel('Parameter b')
plt.ylabel('Distribution')
#plt.title('Parameter b histogram')
plt.grid()
plt.show()

######### Burnin Figures#################################################
plt.scatter(step_count[2000:], A[2000:], linewidths=.35)
plt.grid()
plt.xlabel('Step Number')
plt.ylabel('Paremeter A Value')
#plt.title('MCMC-MH For Parameter A')
plt.show()
# parameter B
plt.scatter(step_count[2000:], B[2000:], linewidths=.35)
plt.grid()
plt.xlabel('Step Number')
plt.ylabel('Paremeter A Value')
#plt.title('MCMC-MH For Parameter B')
plt.show()

# histogram of parmeter a
plt.hist(A[2000:], 50, histtype='step', color='green', label='Parameter A', density=True)
plt.legend()
plt.xlabel('Parameter a')
plt.ylabel('Distribution')
#plt.title('Parameter a histogram')
plt.grid()
plt.show()

# histogram of parameter B
plt.hist(B[2000:], 50, histtype='step', color='red', label='Parameter B', density=True)
plt.legend()
plt.xlabel('Parameter b')
plt.ylabel('Distribution')
#plt.title('Parameter b histogram')
plt.grid()
plt.show()




# scattering x and y and plotting the MCMC and the linregression and the np.polyfit
plt.hist2d(x, y, bins = 100, norm = LogNorm())
plt.plot(x, parameters.slope * x + parameters.intercept, label='scipy.linregress', c='red')
plt.plot(x, a * x + b, label='np.polyfit', c='blue')
plt.plot(x, np.mean(A[2000:]) * x + np.mean(B[2000:]), label='MCMC-MH', c='Green')
plt.grid()
plt.legend()
plt.xlabel(r'LD [$\log \delta_{LD}$]', fontsize=15)
plt.ylabel(r'WHIM Density  [$\log \delta_{\rho}$]', fontsize=15)
#plt.title('Comparison of all line of best fit')
plt.show()

