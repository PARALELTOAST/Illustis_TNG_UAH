import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import random
from functools import partial
import datashader as ds
from datashader.mpl_ext import dsshow

#################### Loading in WHIMden data for monte-carlo fitting ###############################
med_LD = np.loadtxt('scriptforexponentialmodel/med_LD_filrad=10_a=1.0.txt')
mu_LD = np.loadtxt('scriptforexponentialmodel/mu_LD_filrad=10_a=1.0.txt')
mu_WHIM = np.loadtxt('scriptforexponentialmodel/mu_WHIM_filrad=10_a=1.0.txt')
std_WHIM = np.loadtxt('scriptforexponentialmodel/std_WHIM_filrad=10_a=1.0.txt')
############################# Defining exponential function to fit the mean WHIM values for each bin ########################

def expo_fit(x,A,alpha,beta):
    return A * np.exp(alpha * x) + beta

#################################################################################################
idx = 1.0
#expo_dat = open("scaling_relation_plots/exponential_fit_parameters_smooth={idx}_filrad=1Mpc_MC.txt".format(idx=idx),"w")
    
ParameterData=open("scriptforexponentialmodel/ParameterData_a={idx}_filrad=10.txt".format(idx=idx),'w')

################################ Monte-Carlo simulation on the WHIM density #############################################################################################################################
Num_samp=1
num_bins = 30
for n in range(10000):
    #LD_Over_Data = []
    #WHIM_Over_Data = []
    WHIM_Over_Data = np.zeros(num_bins)
    param_bounds = [[-5, -2, -1.5],[3, 2, 0]]
    for i in range(num_bins):
        q=i+1
        WHIMdenbin = np.load('scriptforexponentialmodel/WHIMdenbin{q}_smooth=1.0_filrad=1Mpc.npy'.format(q=q))
        rnd_idx = random.randrange(len(WHIMdenbin))
        #rnd_idx = random.choices(WHIMdenbin, Num_samp)
        #WHIM_Over_Data.append(WHIMdenbin[rnd_idx])
        WHIM_Over_Data[i] = WHIMdenbin[rnd_idx]
    params, pcov = curve_fit(expo_fit, mu_LD, WHIM_Over_Data, p0=[0.5586, 0.6353, -0.7324], bounds=param_bounds, maxfev=8000)
    ParameterData.write(str(params[0])+" "+str(params[1])+" "+str(params[2])+"\n")

ParameterData.close()

params = np.loadtxt('scriptforexponentialmodel/ParameterData_a={idx}_filrad=10.txt'.format(idx=idx))
A = np.asarray(params[:,0])
alpha = np.asarray(params[:,1])
beta = np.asarray(params[:,2])

############################################### a plot############################################
ax1=plt.subplot(111)
ax1.hist(A,bins=np.linspace(min(A),max(A),100),color="orange",density=True)
#mu_A, std_A = norm.fit(A)
mu_A = np.mean(A)
std_A = np.std(A)
#t_A=np.linspace(min(A),max(A),100)
#pdf_A = norm.pdf(t_A,mu_A,std_A)
#ax1.plot(t_A,pdf_A,'r',label='normal fit')
#ax1.axvline(x=mu_A)
#print('mean of a=', mu_A)
#print('std of a=',std_A)
#plt.xlim(-1,100)
plt.xlabel(r'$A$',fontsize=15)
plt.savefig("scriptforexponentialmodel/parameter_A_smooth={idx}_filrad=10.jpg".format(idx=idx))
plt.ion()
plt.show()
plt.close()

######################################## alpha plot ############################################
ax2=plt.subplot(111)
ax2.hist(alpha,bins=np.linspace(min(alpha),max(alpha),100),color="blue",density=True)
#mu_alpha, std_alpha = norm.fit(alpha)
mu_alpha = np.mean(alpha)
std_alpha = np.std(alpha)
#t_alpha = np.linspace(min(alpha),max(alpha),100)
#pdf_alpha = norm.pdf(t_b,mu_alpha,std_alpha)
#ax2.plot(t_alpha ,pdf_alpha,'r',label='normal fit')
#ax2.axvline(x=mu_alpha)
#print('mean of b =',mu_alpha)
#print('std of b =',std_alpha)
plt.xlabel(r'$\alpha$',fontsize=15)
plt.savefig("scriptforexponentialmodel/parameter_alpha_smooth={idx}_filrad=10.jpg".format(idx=idx))
plt.ion()
plt.show()
plt.close()

##################################### Beta plot############################################
ax3=plt.subplot(111)
ax3.hist(beta, bins=np.linspace(min(beta), max(beta), 100), color='red', density=True)
mu_beta = np.mean(beta)
std_beta = np.std(beta)
#plt.xlim(-100,1)
plt.xlabel(r'$\beta$',fontsize=15)
plt.savefig("scriptforexponentialmodel/parameter_beta_smooth={idx}_filrad=10.jpg".format(idx=idx))
plt.ion()
plt.show()
plt.close()

#'''
#Store best fit values for parameters of exponential model.............................
BestFit = open("scriptforexponentialmodel/BestFitValues_smooth={idx}_filrad=10.txt".format(idx=idx),"w")
BestFit.write("Mean of A:"+str(mu_A)+"\n"+"standard deviation of A:"+str(std_A)+"\n"+"Mean of alpha:"+str(mu_alpha)+"\n"+"standard deviation of alpha:"+str(std_alpha)+"\n"+"Mean of beta:"+str(mu_beta)+"\n"+"standard deviation of beta:"+str(std_beta))
BestFit.close()
#'''

WHIM_Data=open('scriptforexponentialmodel/WHIM_Data.txt','w')
Confidence=open('scriptforexponentialmodel/Confidence.txt','w')
u=np.linspace(min(mu_LD),max(mu_LD),100)

for j in range(len(A)):
    for i in range(len(u)):
        def y(x):
            return A[j] * np.exp(alpha[j] * x) + beta[j]
        WHIM_Data.write(str(u[i])+" ")
        WHIM_Data.write(str(y(u[i]))+"\n")

WHIM_Data.close()
WHIM_Dat= np.loadtxt('scriptforexponentialmodel/WHIM_Data.txt')

Sigma=np.zeros(len(u))
for i in range(len(u)):
    w=[]
    w_pow=[]
    v=WHIM_Dat[i:np.size(WHIM_Dat[:,0]):100,0]
    conf=WHIM_Dat[i:np.size(WHIM_Dat[:,0]):100,1]
    std=np.std(conf)
    Sigma[i]=std
    mu=np.mean(conf)
    for j in range(len(conf)):
        if conf[j] < mu-std or conf[j] > mu+std:
            w.append(j)

    conf=np.delete(conf,w)
    v=np.delete(v,w)
    np.savetxt(Confidence,np.column_stack([v,conf]))
Confidence.close()

conf_interval=np.loadtxt('scriptforexponentialmodel/Confidence.txt')
    
## fitting data
#params, pcov = curve_fit(expo_fit, med_LD, mu_WHIM)

#print("parameters of the exponential fit, A, alpha, and beta:",params[0], params[1], params[2])
#expo_dat.write(str(params[0])+" "+str(params[1])+" "+str(params[2])+"\n")
ax=plt.subplot(111)
#ax.errorbar(med_LD, mu_WHIM, std_WHIM, fmt='o', color='g', label="mean")
#ax.errorbar(med_LD, med_WHIM, fmt='x', color='r',label="median")

#mu_A = 0.5586
#mu_alpha = 0.6353
#mu_beta = -0.7324
ax.plot(u, expo_fit(u,mu_A, mu_alpha, mu_beta), color='k', label="fit")
dyn = partial(ds.tf.dynspread, max_px=40, threshold=0.5)
def datshader(ax,y,z):
    df=pd.DataFrame(dict(x=conf_interval[:,0], y=conf_interval[:,1]))
    da1=dsshow(df,ds.Point('x','y'), aspect='auto',ax=ax)
    plt.colorbar(da1)

scat=datshader(ax,conf_interval[:,0],conf_interval[:,1])
plt.legend(loc=4)
plt.grid(False)
plt.xlabel(r'LD Bin Median [$\log \delta_{LD}$]',fontsize=15)
plt.ylabel(r'WHIM Density With Error [$\log \delta_{\rho}$]',fontsize=15)
plt.savefig("scriptforexponentialmodel/Scatter_30bins_smooth={idx}_filrad=10_MC.png".format(idx=idx))
plt.ion()
plt.show()
plt.close()
