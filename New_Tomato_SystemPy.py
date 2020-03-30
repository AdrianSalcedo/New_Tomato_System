from forward_backward_sweep import ForwardBackwardSweep
from matplotlib import rcParams
import pandas as pd

# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']
params = {
    'figure.titlesize': 10,
    'axes.titlesize':   10,
    'axes.labelsize':   10,
    'font.size':        10,
    'legend.fontsize':  8,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'text.usetex':      True
}
rcParams.update(params)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

#
#
#
beta_y_p = 0.5
r_y_1 = 0.004
r_y_2 = 0.003
r_a = 0.003
alpha = 0.2
beta_a_p = 0.5
b_y = 0.025
b_a = 0.050
beta_y_v = 0.00015
beta_a_v = 0.00015
gamma = 0.06
theta = 0.2
mu = 0.3
#
#
# Initial conditions
s_y_p_zero = 0.7
s_a_p_zero = 0.3
l_y_p_zero = 0.0
l_a_p_zero = 0.0
i_y_p_zero = 0.0
i_a_p_zero = 0.0
s_v_zero = 0.92
i_v_zero = 0.08
# Functional Cost
A_1 = 0.5
A_2 = 0.2
A_3 = 0.2
A_4 = 1.2
c_1 = 1.0
c_2 = 1.1
c_3 = 1.2
c_4 = 1.3

name_file_1 = 'New_System_four_control.pdf'

#

fbsm = ForwardBackwardSweep()
fbsm.set_parameters(beta_y_p, r_y_1, r_y_2, r_a, alpha, beta_a_p, b_y,
                       b_a, beta_y_v, beta_a_v, gamma, theta, mu,
                       A_1, A_2, A_3, A_4, c_1, c_2, c_3, c_4,
                       s_y_p_zero, s_a_p_zero, l_y_p_zero, l_a_p_zero, i_y_p_zero, i_a_p_zero, s_v_zero, i_v_zero)

t = fbsm.t
x_wc = fbsm.runge_kutta_forward(fbsm.u)
#
[x, lambda_, u] = fbsm.forward_backward_sweep()

mpl.style.use('ggplot')
# plt.ion()
# n_whole = fbsm.n_whole
ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax1.plot(t, x_wc[:, 4],
         label="Infected young without control",
         color='black'
         )
ax1.plot(t, x_wc[:, 5],
         label="Infected adult without control",
         color='darkgreen'
         )
ax1.plot(t, x[:, 4],
         label="Optimal controlled",
         color='red')
ax1.plot(t, x[:, 5],
         label="Optimal controlled",
         color='orange')
ax1.set_ylabel(r'Infected plants ratio')
ax1.set_xlabel(r'Time (days)')
ax1.legend(loc=0)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax2.plot(t, u[:, 0],
         label="$u_1(t)$ : Remove latent young plants",
         color='orange')
ax2.set_ylabel(r'$u_1(t)$')
ax2.set_xlabel(r'Time(days)')
ax3 = plt.subplot2grid((4, 2), (1, 1))
ax3.plot(t, u[:, 1],
         label="$u_2(t)$ : Remove infectious young plants",
         color='darkgreen')
ax3.set_ylabel(r'$u_2(t)$')
ax3.set_xlabel(r'Time(days)')
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax4.plot(t, u[:, 2],
         label="$u_3(t)$ : Remove infectious adult plants",
         color='darkgreen')
ax4.set_ylabel(r'$u_3(t)$')
ax4.set_xlabel(r'Time(days)')
ax5 = plt.subplot2grid((4, 2), (3, 1))
ax5.plot(t, u[:, 3],
         label="$u_4(t)$ : Fumigation",
         color='darkgreen')
ax5.set_ylabel(r'$u_4(t)$')
ax5.set_xlabel(r'Time(days)')
plt.tight_layout()
#
fig = mpl.pyplot.gcf()
fig.set_size_inches(4.5, 4.5 / 1.618)
fig.savefig(name_file_1,
            # additional_artists=art,
            bbox_inches="tight")
#######################################################################################################################
#plt.figure()
#Cost_value = (A_1 * x[:, 2] + A_2 * x[:, 1]+ A_3 * x[:, 4]+ c_1 * u[:, 0] ** 2 + c_2 * u[:, 1] ** 2) * (365 / 10000)

#Int_Cost_value = np.cumsum(Cost_value)
#print(Int_Cost_value[len(t)])
#np.save('time.npy',t)
#np.save('two_control_cost.npy',Int_Cost_value)
#plt.plot(t,Int_Cost_value)
plt.show()