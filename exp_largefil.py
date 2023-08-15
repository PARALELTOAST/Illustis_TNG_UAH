import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import random

#################### Loading in WHIMden data for monte-carlo fitting ###############################
#med_LD = np.loadtxt('scriptforexponentialmodel/med_LD_filrad=10_a=1.0.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_5MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_1MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/mu_LD_filrad=10_a=1.0.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_2MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a1_2MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a15_2MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a2_2MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a5_3MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a1_3MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a15_3MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a2_3MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a3_3MPC.txt')

#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a3_2MPC.txt')
mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a15_1MPC.txt')
#mu_LD = np.loadtxt('scriptforexponentialmodel/largefilinfo/mu_ld_a2_1MPC.txt')

    #Remains The Same for the Red and Blue#
mu_WHIM = np.loadtxt('scriptforexponentialmodel/mu_WHIM_filrad=10_a=1.0.txt')
std_WHIM = np.loadtxt('scriptforexponentialmodel/std_WHIM_filrad=10_a=1.0.txt')

iteration = '1.5a 1MPC'
############################# Defining exponential function to fit the mean WHIM values for each bin ########################

def expo_fit(x, A, alpha, beta):
    return A * np.exp(alpha * x) + beta


#################################################################################################
idx = 1.0
# expo_dat = open("scaling_relation_plots/exponential_fit_parameters_smooth={idx}_filrad=1Mpc_MC.txt".format(idx=idx),"w")

ParameterData = open("scriptforexponentialmodel/ParameterData_a={idx}_filrad=10.txt".format(idx=idx), 'w')

################################ Monte-Carlo simulation on the WHIM density For #############################################################################################################################
####################################### BLUE #####################################################
Num_samp = 1
num_bins = 30
for n in range(10000):
    # LD_Over_Data = []
    # WHIM_Over_Data = []
    WHIM_Over_Data = np.zeros(num_bins)
    param_bounds = [[-5, -2, -4], [3, 2, 0]]
    for i in range(num_bins):
        q = i + 1
        WHIMdenbin = np.load('scriptforexponentialmodel/WHIMdenbin{q}_smooth=1.0_filrad=1Mpc.npy'.format(q=q))
        rnd_idx = random.randrange(len(WHIMdenbin))
        # rnd_idx = random.choices(WHIMdenbin, Num_samp)
        # WHIM_Over_Data.append(WHIMdenbin[rnd_idx])
        WHIM_Over_Data[i] = WHIMdenbin[rnd_idx]
    params, pcov = curve_fit(expo_fit, mu_LD, WHIM_Over_Data, p0=[0.5586, 0.6353, -0.7324], bounds=param_bounds,
                             maxfev=8000)
    ParameterData.write(str(params[0]) + " " + str(params[1]) + " " + str(params[2]) + "\n")

ParameterData.close()

params = np.loadtxt('scriptforexponentialmodel/ParameterData_a={idx}_filrad=10.txt'.format(idx=idx))
A = np.asarray(params[:, 0])
alpha = np.asarray(params[:, 1])
beta = np.asarray(params[:, 2])

############################################### a plot############################################
ax1 = plt.subplot(111)
ax1.hist(A, bins=np.linspace(min(A), max(A), 100), color="orange", density=True)
# mu_A, std_A = norm.fit(A)
mu_A = np.mean(A)
std_A = np.std(A)
# t_A=np.linspace(min(A),max(A),100)
# pdf_A = norm.pdf(t_A,mu_A,std_A)
# ax1.plot(t_A,pdf_A,'r',label='normal fit')
# ax1.axvline(x=mu_A)
# print('mean of a=', mu_A)
# print('std of a=',std_A)
# plt.xlim(-1,100)
plt.xlabel(r'$A$ Blue', fontsize=15)
plt.savefig("scriptforexponentialmodel/parameter_A_blue.jpg".format(idx=idx))
plt.ion()
plt.close()

######################################## alpha plot ############################################
ax2 = plt.subplot(111)
ax2.hist(alpha, bins=np.linspace(min(alpha), max(alpha), 100), color="blue", density=True)
# mu_alpha, std_alpha = norm.fit(alpha)
mu_alpha = np.mean(alpha)
std_alpha = np.std(alpha)
# t_alpha = np.linspace(min(alpha),max(alpha),100)
# pdf_alpha = norm.pdf(t_b,mu_alpha,std_alpha)
# ax2.plot(t_alpha ,pdf_alpha,'r',label='normal fit')
# ax2.axvline(x=mu_alpha)
# print('mean of b =',mu_alpha)
# print('std of b =',std_alpha)
plt.xlabel(r'$\alpha$ Blue', fontsize=15)
plt.savefig("scriptforexponentialmodel/parameter_alpha_blue.jpg".format(idx=idx))
plt.ion()
plt.close()

##################################### Beta plot############################################
ax3 = plt.subplot(111)
ax3.hist(beta, bins=np.linspace(min(beta), max(beta), 100), color='red', density=True)
mu_beta = np.mean(beta)
std_beta = np.std(beta)
# plt.xlim(-100,1)
plt.xlabel(r'$\beta$ Blue', fontsize=15)
plt.savefig("scriptforexponentialmodel/parameter_beta_blue".format(idx=idx))
plt.ion()
plt.close()

values = ([mu_A, std_A],[mu_alpha, std_alpha],[mu_beta, std_beta])
print(values)
new_parameters = open('scriptforexponentialmodel/extra_fit_parameters.txt','a')
new_parameters.write('\n' + 'Iteration:'+ iteration + ': ' +  str(values))
