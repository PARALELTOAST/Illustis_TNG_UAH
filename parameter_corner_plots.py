import corner
import matplotlib.pyplot as plt
import numpy as np
import corner as cor
import arviz



fil_rad = (.5, .5, .5, 1, 1, 1, 2, 2, 3)
smoothing_parameter = (.5, 1, 2, 1, 2, 3, 2, 3, 3)


std_a = (.589, .581, .566, .577, .583, .582, .591, .573, .583)
std_alpha = (.261, .259, .304, .296, .273, .333, .297, .297, .327)
std_beta = (.464, .463, .453, .472, .459, .450, .463, .441, .453)

A = (.832, .858, .838, .744, .904, .918, .893, 1.02, .919)
alpha = (.586, .555, .669, .6348, .581, .689, .603, .631, .686)
beta = (-.881, -.896, -.884, -.893, -.888, -.875, -.879, -.873, -.874)

sm_a = np.asarray([(x, y) for x, y in zip(np.array(smoothing_parameter).flatten(), np.array(A).flatten())])
np.vstack(sm_a)
sm_al = np.asarray([(x, y) for x, y in zip(np.array(smoothing_parameter).flatten(), np.array(alpha).flatten())])
np.vstack(sm_al)
sm_b = np.asarray([(x, y) for x, y in zip(np.array(smoothing_parameter).flatten(), np.array(beta).flatten())])
np.vstack(sm_b)

#data = np.vstack([sm_a, sm_al, sm_b])

#all_data = np.asarray([(x, y, z) for x,y,z in zip(np.array(A).flatten(), np.array(alpha).flatten(), np.array(beta).flatten())])
#all_data = np.asarray([(x,y) for x,y in zip(np.array(smoothing_parameter).flatten(), np.array(all_data).flatten())])

all_data = np.asarray([(sm, f, x, y, z) for sm,f,x,y,z in zip(np.array(smoothing_parameter).flatten(), np.array(fil_rad).flatten(),np.array(A).flatten(), np.array(alpha).flatten(), np.array(beta).flatten())])

#all_data = np.asarray([(sm, f, a) for sm,f,a in zip(np.array(smoothing_parameter).flatten(),np.array(fil_rad).flatten(), np.array(A).flatten())])


data = np.vstack(all_data)
corner.corner(data = data, labels = ['Smoothing Parameter', 'Filament Radius', 'A', '$\u03B1 $', '$\u03B2 $'], show_titles= True )
plt.show()

# fig_1 = corner.corner(sm_a)
# fig_2 = corner.corner(sm_al, fig = fig_1)
# corner.corner(sm_b, fig = fig_2)
# plt.show()

