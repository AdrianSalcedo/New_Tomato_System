from forward_backward_sweep import ForwardBackwardSweep
from matplotlib import rcParams
from NTW_Odeint import rhs_ntw_solve
#from NTW_odeint_backward import rhs_ntw_backward_solve
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
beta_y_p = 0.003
r_y_1 = 0.004
r_y_2 = 0.003
r_a = 0.017
alpha = 0.04
beta_a_p = 0.1
b_y = 0.025
b_a = 0.050
beta_y_v = 0.00015
beta_a_v = 0.00015
gamma = 0.3
theta = 0.2
mu = 0.6
#
#
# Initial conditions
s_y_p_zero = 10000.0
s_a_p_zero = 0.0
l_y_p_zero = 0.0
l_a_p_zero = 0.0
i_y_p_zero = 0.0
i_a_p_zero = 0.0
s_v_zero = 2.0
i_v_zero = 0.0

N_p = s_y_p_zero + s_a_p_zero + l_y_p_zero + l_a_p_zero + i_y_p_zero + i_a_p_zero
N_v = mu / gamma

s_y_p_zero = s_y_p_zero / N_p
s_a_p_zero = s_a_p_zero / N_p
l_y_p_zero = l_y_p_zero / N_p
l_a_p_zero = l_a_p_zero / N_p
i_y_p_zero = i_y_p_zero / N_p
i_a_p_zero = i_a_p_zero / N_p
s_v_zero = s_v_zero / N_v
i_v_zero = i_v_zero / N_v

# Functional Cost
A_1 = 0.5
A_2 = 0.2
A_3 = 0.2
A_4 = 0.2
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

[x, lambda_, u] = fbsm.forward_backward_sweep()
cost = fbsm.control_cost(fbsm.x,u)

########################################################################################################################
                                                    #''' \R_0 Computation '''
########################################################################################################################

R_0 = np.sqrt((beta_y_p* N_p) / gamma) * np.sqrt((beta_y_v * mu * b_y) / (gamma * (b_y * r_y_2 + r_y_1 * r_y_2)))
print('R_0 = ',R_0)

########################################################################################################################
                                                 #''' ODEINT SOLVER '''
########################################################################################################################

y_odeint_zero = np.array([s_y_p_zero, s_a_p_zero, l_y_p_zero, l_a_p_zero, i_y_p_zero, i_a_p_zero, s_v_zero * N_p, i_v_zero * N_p]) / N_p
vector_par = np.array([beta_y_p, r_y_1, r_y_2, r_a , alpha, beta_a_p, b_y, b_a, beta_y_v, beta_a_v, gamma, theta, mu])

Sol_odeint = rhs_ntw_solve(y_odeint_zero,vector_par, t)

########################################################################################################################
                #''' Plot comparing solution with control and without control, and control plot'''
########################################################################################################################

mpl.style.use('ggplot')
# plt.ion()
# n_whole = fbsm.n_whole
ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
#ax1.plot(t, x_wc_1[:, 4],
#         label="Infected adult without control",
#         color='gray'
#         )

ax1.plot(t, x_wc[:, 5],
         label="Infected adult without control",
         color='darkgreen'
         )
#ax1.plot(t, x[:, 4],
#        label="Optimal controlled adult",
#         color='red'
#         )
ax1.plot(t, x[:, 5],
         label="Optimal controlled adult",
         color='orange')
ax1.set_ylabel(r'Infected plants ratio')
ax1.set_xlabel(r'Time (days)')
ax1.legend(loc=0)

ax2 = plt.subplot2grid((4, 2), (0, 1))
ax2.plot(t, u[:, 0],
         label="$u_1(t)$ : Remove latent young plants",
         color='darkgreen')
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

########################################################################################################################
                                                    #'''Cost plot'''
########################################################################################################################

plt.figure()
plt.plot(t,cost,'k')
plt.xlabel(r'Time(days)')
plt.ylabel(r'Cost')
plt.title(r'Cost Function')
########################################################################################################################
                                #''' Only state variables with control plot '''
########################################################################################################################

plt.figure()
mpl.style.use('ggplot')
# plt.ion()
# n_whole = fbsm.n_whole
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax1.plot(t, x_wc[:, 0], label="Susceptible young without control")
ax1.plot(t, x[:, 0], label="Optimal controlled susceptible young")
ax1.set_ylabel(r'$S^y_p$')
ax1.set_xlabel(r'Time(days)')

ax2 = plt.subplot2grid((3, 2), (0, 1))
ax2.plot(t, x_wc[:, 1], label="Susceptible adult without control")
ax2.plot(t, x[:, 1],label="Optimal controlled susceptible adult")
ax2.set_ylabel(r'$S^a_p$')
ax2.set_xlabel(r'Time(days)')

ax3 = plt.subplot2grid((3, 2), (1, 0))
ax3.plot(t, x_wc[:, 2], label="Latent young without control")
ax3.plot(t, x[:, 2], label="Optimal controlled latent young")
ax3.set_ylabel(r'$L^y_p$')
ax3.set_xlabel(r'Time(days)')

ax4 = plt.subplot2grid((3, 2), (1, 1))
ax4.plot(t, x_wc[:, 3], label="Latent adult without control")
ax4.plot(t, x[:, 3], label="Optimal controlled latent adult")
ax4.set_ylabel(r'$L^a_p$')
ax4.set_xlabel(r'Time(days)')

ax5 = plt.subplot2grid((3, 2), (2, 0))
ax5.plot(t, x_wc[:, 4], label="Infected young without control")
ax5.plot(t, x[:, 4], label="Optimal controlled infected young")
ax5.set_ylabel(r'$I^y_p$')
ax5.set_xlabel(r'Time(days)')

ax6 = plt.subplot2grid((3, 2), (2, 1))
ax6.plot(t, x_wc[:, 5], label="Infected adult without control")
ax6.plot(t, x[:, 5], label="Optimal controlled infected adult")
ax6.set_ylabel(r'$I^a_p$')
ax6.set_xlabel(r'Time(days)')

plt.tight_layout()

########################################################################################################################

plt.figure()
mpl.style.use('ggplot')
# plt.ion()
# n_whole = fbsm.n_whole
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax1.plot(t, x_wc[:, 6], label="Susceptible vector without control")
ax1.plot(t, x[:, 6], label="Optimal controlled susceptible vector")
ax1.set_ylabel(r'$S_v$')
ax1.set_xlabel(r'Time(days)')

ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.plot(t, x_wc[:, 7], label="Infected vector without control")
ax2.plot(t, x[:, 7], label="Optimal controlled infected vector")
ax2.set_ylabel(r'$I_v$')
ax2.set_xlabel(r'Time(days)')

plt.tight_layout()

########################################################################################################################
########################################################################################################################
                                   # ''' Solution without control using odeint plot'''
########################################################################################################################

plt.figure()
plt.plot(t,Sol_odeint[:,0], label="young susceptible")
plt.plot(t,Sol_odeint[:,1], label="adult susceptible")
plt.plot(t,Sol_odeint[:,2], label="young latent")
plt.plot(t,Sol_odeint[:,3], label="adult latent")
plt.plot(t,Sol_odeint[:,4], label="young infected")
plt.plot(t,Sol_odeint[:,5], label="adult infected")
plt.title(r'Odeint solution')
plt.xlabel(r'Time(days)')
plt.ylabel(r'Plant Population')
plt.legend(loc=0)

plt.figure()
plt.plot(t,Sol_odeint[:,6], label="vector susceptible")
plt.plot(t,Sol_odeint[:,7], label="vector infected")
plt.title(r'Odeint solution')
plt.xlabel(r'Time(days)')
plt.ylabel(r'Vector population')
plt.legend(loc=0)

########################################################################################################################
                                            #'''Adjoint Variables plot'''
########################################################################################################################

plt.figure()
mpl.style.use('ggplot')
# plt.ion()
# n_whole = fbsm.n_whole
down_limlambbda = -2
upper_limlambbda = 4
ax1 = plt.subplot2grid((4, 2), (0, 0))
ax1.plot(t, lambda_[:, 0])
ax1.set_ylabel(r'$\lambda_1$')
ax1.set_xlabel(r'Time(days)')
ax1.set_ylim(down_limlambbda,upper_limlambbda)

ax2 = plt.subplot2grid((4, 2), (0, 1))
ax2.plot(t, lambda_[:, 1])
ax2.set_ylabel(r'$\lambda_2$')
ax2.set_xlabel(r'Time(days)')
ax2.set_ylim(down_limlambbda,upper_limlambbda)

ax3 = plt.subplot2grid((4, 2), (1, 0))
ax3.plot(t, lambda_[:, 2])
ax3.set_ylabel(r'$\lambda_3$')
ax3.set_xlabel(r'Time(days)')
ax3.set_ylim(down_limlambbda,upper_limlambbda)

ax4 = plt.subplot2grid((4, 2), (1, 1))
ax4.plot(t, lambda_[:, 3])
ax4.set_ylabel(r'$\lambda_4$')
ax4.set_xlabel(r'Time(days)')
ax4.set_ylim(down_limlambbda,upper_limlambbda)

ax5 = plt.subplot2grid((4, 2), (2, 0))
ax5.plot(t, lambda_[:, 4])
ax5.set_ylabel(r'$\lambda_5$')
ax5.set_xlabel(r'Time(days)')
ax5.set_ylim(down_limlambbda,upper_limlambbda)

ax6 = plt.subplot2grid((4, 2), (2, 1))
ax6.plot(t, lambda_[:, 5])
ax6.set_ylabel(r'$\lambda_16$')
ax6.set_xlabel(r'Time(days)')
ax6.set_ylim(down_limlambbda,upper_limlambbda)

ax7 = plt.subplot2grid((4, 2), (3, 0))
ax7.plot(t, lambda_[:, 6])
ax7.set_ylabel(r'$\lambda_7$')
ax7.set_xlabel(r'Time(days)')
ax7.set_ylim(down_limlambbda,upper_limlambbda)

ax8 = plt.subplot2grid((4, 2), (3, 1))
ax8.plot(t, lambda_[:, 7])
ax8.set_ylabel(r'$\lambda_8$')
ax8.set_xlabel(r'Time(days)')
ax8.set_ylim(down_limlambbda,upper_limlambbda)

plt.tight_layout()

########################################################################################################################
########################################################################################################################
plt.show()