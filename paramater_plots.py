import matplotlib.pyplot as plt
import numpy as np

fil_rad = (.5, .5, .5, 1, 1, 1, 2, 2, 3)
smoothing_parameter = (.5, 1, 2, 1, 2, 3, 2, 3, 3)

std_a = (.589, .581, .566, .577, .583, .582, .591, .573, .583)
std_alpha = (.261, .259, .304, .296, .273, .333, .297, .297, .327)
std_beta = (.464, .463, .453, .472, .459, .450, .463, .441, .453)

A = (.832, .858, .838, .744, .904, .918, .893, 1.02, .919)
alpha = (.586, .555, .669, .6348, .581, .689, .603, .631, .686)
beta = (-.881, -.896, -.884, -.893, -.888, -.875, -.879, -.873, -.874)

smooth_5 = []
A_5 = []
alpha_5 = []
beta_5 = []
smooth_1 = []
A_1 = []
alpha_1 = []
beta_1 = []
smooth_2 = []
A_2 = []
alpha_2 = []
beta_2 = []
smooth_3 = []
A_3 = []
alpha_3 = []
beta_3 = []
std_a_5 = []
std_alpha_5 = []
std_beta_5 = []
std_a_1 = []
std_alpha_1 = []
std_beta_1 = []
std_a_2 = []
std_alpha_2 = []
std_beta_2 = []
std_a_3 = []
std_alpha_3 = []
std_beta_3 = []
for i in range(len(fil_rad)):
    if fil_rad[i] == .5:
        smooth_5.append(smoothing_parameter[i])
        A_5.append(A[i])
        alpha_5.append(alpha[i])
        beta_5.append(beta[i])
        std_a_5.append(std_a[i])
        std_alpha_5.append(std_alpha[i])
        std_beta_5.append(std_beta[i])

    if fil_rad[i] == 1:
        smooth_1.append(smoothing_parameter[i])
        A_1.append(A[i])
        alpha_1.append(alpha[i])
        beta_1.append(beta[i])
        std_a_1.append(std_a[i])
        std_alpha_1.append(std_alpha[i])
        std_beta_1.append(std_beta[i])

    if fil_rad[i] == 2:
        smooth_2.append(smoothing_parameter[i])
        A_2.append(A[i])
        alpha_2.append(alpha[i])
        beta_2.append(beta[i])
        std_a_2.append(std_a[i])
        std_alpha_2.append(std_alpha[i])
        std_beta_2.append(std_beta[i])

    if fil_rad[i] == 3:
        smooth_3.append(smoothing_parameter[i])
        A_3.append(A[i])
        alpha_3.append(alpha[i])
        beta_3.append(beta[i])
        std_a_3.append(std_a[i])
        std_alpha_3.append(std_alpha[i])
        std_beta_3.append(std_beta[i])

######################## Trying it all in one plot############################
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
#
# ax1.set_xlabel('Filament Radius')
# ax1.scatter(fil_rad, A, color = 'black', label = 'A')
# ax1.scatter(fil_rad, beta, color = 'blue', label = 'Beta')
# ax1.scatter(fil_rad, alpha, color = 'red', label = 'Alpha')
#
# ax2.set_xlabel('Parameter')
# ax2.plot(smoothing_parameter, A, alpha = 0) # Create a dummy plot
# ax1.grid()
# ax1.legend()
# plt.show()

############################## Normal Plots#######################
#################### Fil rad
# plt.scatter(fil_rad, A, color = 'black', label = 'A')
# plt.scatter(fil_rad, beta, color = 'blue', label = 'Beta')
# plt.scatter(fil_rad, alpha, color = 'red', label = 'Alpha')
# plt.xlabel('Filament Radius [Mpc]')
# plt.ylabel('Parameter')
# for i in range(len(fil_rad)):
#     if smoothing_parameter[i] == .5:
#         plt.annotate(.5, (fil_rad[i], A[i]), xytext=(fil_rad[i] + 0.05, A[i] + .02))
#         plt.annotate(.5, (fil_rad[i], alpha[i]), xytext=(fil_rad[i] +.03 , alpha[i] -.08 ), color='red')
#         plt.annotate(.5, (fil_rad[i], beta[i]), xytext=(fil_rad[i] +.03, beta[i] + .02), color='blue')
#
#     if smoothing_parameter[i] == 1:
#         plt.annotate(1, (fil_rad[i], A[i]), xytext=(fil_rad[i] - 0.08, A[i]))
#         plt.annotate(1, (fil_rad[i], alpha[i]), xytext=(fil_rad[i] - 0.08, alpha[i] + .02), color = 'red')
#         plt.annotate(1, (fil_rad[i], beta[i]), xytext=(fil_rad[i] - 0.08, beta[i]), color = 'blue')
#
#     if smoothing_parameter[i] == 2:
#         plt.annotate(2, (fil_rad[i], A[i]), xytext=(fil_rad[i] + 0.05, A[i]-.06))
#         plt.annotate(2, (fil_rad[i], alpha[i]), xytext=(fil_rad[i] + 0.08, alpha[i] -.04), color='red')
#         plt.annotate(2, (fil_rad[i], beta[i]), xytext=(fil_rad[i] + 0.08, beta[i] - .06), color='blue')
#
#     if smoothing_parameter[i] == 3:
#         plt.annotate(3, (fil_rad[i], A[i]), xytext=(fil_rad[i] + 0.05, A[i]))
#         plt.annotate(3, (fil_rad[i], alpha[i]), xytext=(fil_rad[i] + 0.08, alpha[i]), color='red')
#         plt.annotate(3, (fil_rad[i], beta[i]), xytext=(fil_rad[i] + 0.08, beta[i]), color='blue')
#
#
# plt.legend()
# plt.grid()
# plt.show()
# plt.savefig('radius_vs_parameter.png')
# plt.close()
#

####################### Smoothing parameter
# plt.scatter(smoothing_parameter, A, color = 'black', label = 'A')
# plt.scatter(smoothing_parameter, beta, color = 'blue', label = 'Beta')
# plt.scatter(smoothing_parameter, alpha, color = 'red', label = 'Alpha')
############## The data
plt.scatter(smooth_5, A_5, color='black', marker='v')
plt.scatter(smooth_5, beta_5, color='blue', marker='v')
plt.scatter(smooth_5, alpha_5, color='red', marker='v')

plt.scatter(smooth_1, A_1, color='black', marker='.')
plt.scatter(smooth_1, beta_1, color='blue', marker='.')
plt.scatter(smooth_1, alpha_1, color='red', marker='.')

plt.scatter(smooth_2, A_2, color='black', marker='1')
plt.scatter(smooth_2, beta_2, color='blue', marker='1')
plt.scatter(smooth_2, alpha_2, color='red', marker='1')

plt.scatter(smooth_3, A_3, color='black', marker='x')
plt.scatter(smooth_3, beta_3, color='blue', marker='x')
plt.scatter(smooth_3, alpha_3, color='red', marker='x')
############# Error bars
plt.errorbar(smooth_5, A_5, yerr= std_a_5, fmt="v", color = 'black')
plt.errorbar(smooth_1, A_1, yerr= std_a_1, fmt=".", color = 'black')
plt.errorbar(smooth_2, A_2, yerr= std_a_2, fmt="1", color = 'black')
plt.errorbar(smooth_3, A_3, yerr= std_a_3, fmt="x", color = 'black')

plt.errorbar(smooth_5, alpha_5, yerr=std_alpha_5, fmt="v", color = 'red')
plt.errorbar(smooth_1, alpha_1, yerr=std_alpha_1, fmt=".", color = 'red')
plt.errorbar(smooth_2, alpha_2, yerr=std_alpha_2, fmt="1", color = 'red')
plt.errorbar(smooth_3, alpha_3, yerr=std_alpha_3, fmt="x", color = 'red')

plt.errorbar(smooth_5, beta_5, yerr=std_beta_5, fmt="v", color = 'blue')
plt.errorbar(smooth_1, beta_1, yerr=std_beta_1, fmt=".", color = 'blue')
plt.errorbar(smooth_2, beta_2, yerr=std_beta_2, fmt="1", color = 'blue')
plt.errorbar(smooth_3, beta_3, yerr=std_beta_3, fmt="x", color = 'blue')

############# labels
plt.scatter(0, 0, label='A is Black', alpha=0)
plt.scatter(0, 0, label='Beta is Red', alpha=0)
plt.scatter(0, 0, label='Alpha is Blue', alpha=0)
plt.scatter(-10, -1, label='.5Mpc', marker='v')
plt.scatter(-10, -1, label='1Mpc', marker='.')
plt.scatter(-10, -1, label='2Mpc', marker='1')
plt.scatter(-10, -1, label='3Mpc', marker='x')

# for i in range(len(fil_rad)):
#     if fil_rad[i] == .5:
#         plt.annotate(.5, (smoothing_parameter[i], A[i]), xytext=(smoothing_parameter[i] + 0.05, A[i]-.02))
#         plt.annotate(.5, (smoothing_parameter[i], alpha[i]), xytext=(smoothing_parameter[i] +.03 , alpha[i] ), color='red')
#         plt.annotate(.5, (smoothing_parameter[i], beta[i]), xytext=(smoothing_parameter[i] +.03, beta[i] + .02), color='blue')
#
#     if fil_rad[i] == 1:
#         plt.annotate(1, (smoothing_parameter[i], A[i]), xytext=(smoothing_parameter[i] - 0.08, A[i]))
#         plt.annotate(1, (smoothing_parameter[i], alpha[i]), xytext=(smoothing_parameter[i] - 0.08, alpha[i] + .02), color = 'red')
#         plt.annotate(1, (smoothing_parameter[i], beta[i]), xytext=(smoothing_parameter[i] - 0.08, beta[i]), color = 'blue')
#
#     if fil_rad[i] == 2:
#         plt.annotate(2, (smoothing_parameter[i], A[i]), xytext=(smoothing_parameter[i] + 0.05, A[i]))
#         plt.annotate(2, (smoothing_parameter[i], alpha[i]), xytext=(smoothing_parameter[i] + 0.05, alpha[i] -.04), color='red')
#         plt.annotate(2, (smoothing_parameter[i], beta[i]), xytext=(smoothing_parameter[i] + 0.05, beta[i] - .07), color='blue')
#
#     if fil_rad[i] == 3:
#         plt.annotate(3, (smoothing_parameter[i], A[i]), xytext=(smoothing_parameter[i] + 0.05, A[i]-.05))
#         plt.annotate(3, (smoothing_parameter[i], alpha[i]), xytext=(smoothing_parameter[i] + 0.05, alpha[i]), color='red')
#         plt.annotate(3, (smoothing_parameter[i], beta[i]), xytext=(smoothing_parameter[i] + 0.05, beta[i]), color='blue')

plt.xlim(0, 3.2)
#plt.ylim(0, 1)
plt.xlabel('Smoothing Parameter [Mpc]')
plt.ylabel('Parameter')
plt.legend(loc = 'center left',fontsize = 'small', ncol = 3, bbox_to_anchor = (0,.43))
plt.grid()
plt.savefig('smooth_vs_parameter.jpg')
plt.show()
plt.close()
