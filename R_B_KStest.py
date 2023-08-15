import matplotlib.pyplot as plt
import numpy as np
import scipy

ld = np.loadtxt('ldlog_nofil.txt')
ks_result = []
p_val = []
for i in range(0,30):
    red_dist = np.loadtxt('logden_red_bin' + str(i+1) +'.txt')
    blue_dis = np.loadtxt('logden_blue_bin' + str(i+1) +'.txt')
    results = scipy.stats.ks_2samp(red_dist, blue_dis)
    ks_result.append(results.statistic)
    p_val.append(results.pvalue)

x = np.linspace(min(ld), max(ld), len(ks_result))
plt.plot(x, p_val, label = 'pvalue')
plt.xlabel('LD [$\log \delta_{LD}$]')
plt.ylabel('KS Test p-value')
plt.title('KS-Test P-value for Red vs Blue WHIM Distribution Bins')
plt.legend()
plt.grid()
plt.show()

