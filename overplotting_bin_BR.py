import numpy as np
import matplotlib.pyplot as plt

##### loading in the data
mean_R_4 = np.load('mean_red_bin4.npy')
mean_R_16 = np.load('mean_red_bin16.npy')
mean_R_30 = np.load('mean_red_bin30.npy')

med_R_4 = np.load('med_red_bin4.npy')
med_R_16 = np.load('med_red_bin16.npy')
med_R_30 = np.load('med_red_bin30.npy')

std_R_4 = np.load('std_red_bin4.npy')
std_R_16 = np.load('std_red_bin16.npy')
std_R_30 = np.load('std_red_bin30.npy')

logden_R_4=np.loadtxt('logden_red_bin4.txt')
logden_R_16=np.loadtxt('logden_red_bin16.txt')
logden_R_30=np.loadtxt('logden_red_bin30.txt')

mean_B_4 = np.load('mean_blue_bin4.npy')
mean_B_16 = np.load('mean_blue_bin16.npy')
mean_B_30 = np.load('mean_blue_bin30.npy')

med_B_4 = np.load('med_blue_bin4.npy')
med_B_16 = np.load('med_blue_bin16.npy')
med_B_30 = np.load('med_blue_bin30.npy')

std_B_4 = np.load('std_blue_bin4.npy')
std_B_16 = np.load('std_blue_bin16.npy')
std_B_30 = np.load('std_blue_bin30.npy')

logden_B_4 = np.loadtxt('logden_blue_bin4.txt')
logden_B_16 = np.loadtxt('logden_blue_bin16.txt')
logden_B_30 = np.loadtxt('logden_blue_bin30.txt')
###################3 overplotting the distributions
########### Bin 4
x = plt.np.linspace(min(logden_R_4), max(logden_R_4), 30)
plt.hist(logden_R_4, 30, label = 'WHIM Distribution Reds', density = True, color = 'red', histtype= 'step')
plt.hist(logden_B_4, 30, label = 'WHIM Distribution Reds', density = True, color = 'Blue', histtype= 'step')

plt.axvline(mean_R_4, linestyle = '--', color = 'red', label = 'Red Mean')
plt.axvline(med_R_4, linestyle = 'dotted', color = 'red', label = 'Red Median')

plt.axvline(mean_B_4, linestyle = '--', color = 'Blue', label = 'BLue Mean')
plt.axvline(med_B_4, linestyle = 'dotted', color = 'Blue', label = 'Blue Median')

plt.xlabel('Whim Value')
plt.title('Bin Number 4 Red and Blue WHIM distribution')
plt.legend()
plt.show()
plt.savefig('bin4_RB_overplot.png')
plt.close()

######### Bin 16
x = plt.np.linspace(min(logden_R_4), max(logden_R_4), 30)
plt.hist(logden_R_16, 30, label = 'WHIM Distribution Reds', density = True, color = 'red', histtype= 'step')
plt.hist(logden_B_16, 30, label = 'WHIM Distribution Reds', density = True, color = 'Blue', histtype= 'step')

plt.axvline(mean_R_16, linestyle = '--', color = 'red', label = 'Red Mean')
plt.axvline(med_R_16, linestyle = 'dotted', color = 'red', label = 'Red Median')

plt.axvline(mean_B_16, linestyle = '--', color = 'Blue', label = 'BLue Mean')
plt.axvline(med_B_16, linestyle = 'dotted', color = 'Blue', label = 'Blue Median')

plt.xlabel('Whim Value')
plt.title('Bin Number 16 Red and Blue WHIM distribution')
plt.legend()
plt.show()
plt.savefig('bin16_RB_overplot.png')
plt.close()

########## Bin 30
x = plt.np.linspace(min(logden_R_4), max(logden_R_4), 30)
plt.hist(logden_R_30, 30, label = 'WHIM Distribution Reds', density = True, color = 'red', histtype= 'step')
plt.hist(logden_B_30, 30, label = 'WHIM Distribution Reds', density = True, color = 'Blue', histtype= 'step')

plt.axvline(mean_R_30, linestyle = '--', color = 'red', label = 'Red Mean')
plt.axvline(med_R_30, linestyle = 'dotted', color = 'red', label = 'Red Median')

plt.axvline(mean_B_30, linestyle = '--', color = 'Blue', label = 'BLue Mean')
plt.axvline(med_B_30, linestyle = 'dotted', color = 'Blue', label = 'Blue Median')

plt.xlabel('Whim Value')
plt.title('Bin Number 30 Red and Blue WHIM distribution')
plt.legend()
plt.show()
plt.savefig('bin30_RB_overplot.png')
plt.close()
