import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

n_max=50000
t_backward = np.linspace(140,0,n_max)
t_forward = np.linspace(0,140,n_max)
# Initial conditions
s_y_p_zero = 100000.0
s_a_p_zero = 0.0
l_y_p_zero = 0.0
l_a_p_zero = 0.0
i_y_p_zero = 0.0
i_a_p_zero = 0.0
s_v_zero = 0.0
i_v_zero = 0.0

beta_y_p = 0.0025
r_y_1 = 0.004
r_y_2 = 0.003
r_a = 0.017
alpha = 0.02
beta_a_p = 0.1
b_y = 0.025
b_a = 0.050
beta_y_v = 0.00015
beta_a_v = 0.00015
gamma = 0.3
theta = 0.2
mu = 0.6
A_1 = 0.5
A_2 = 0.2
A_3 = 0.2
A_4 = 1.2

N_p = s_y_p_zero + s_a_p_zero + l_y_p_zero + l_a_p_zero + i_y_p_zero + i_a_p_zero
y_zero = np.array([s_y_p_zero, s_a_p_zero, l_y_p_zero, l_a_p_zero, i_y_p_zero, i_a_p_zero, s_v_zero * N_p, i_v_zero * N_p]) / N_p



#def rhs_ntw__backward_solve(y_final, par, t):
#    beta_y_p = par[0]
#    r_y_1 = par[1]
#    r_y_2 = par[2]
#    r_a = par[3]
#    alpha = par[4]
#    beta_a_p = par[5]
#    b_y = par[6]
#    b_a = par[7]
#    beta_y_v = par[8]
#    beta_a_v = par[9]
#    gamma = par[10]
#    theta = par[11]
#    mu = par[12]
'''
def rhs_ntw(y, t_zero):
    s_y_p = y[0]
    s_a_p = y[1]
    l_y_p = y[2]
    l_a_p = y[3]
    i_y_p = y[4]
    i_a_p = y[5]
    s_v = y[6]
    i_v = y[7]
    u_1 = 0.5
    u_2 = 0.5
    u_3 = 0.5
    u_4 = 0.5

    N_p = s_y_p + l_y_p + i_y_p + s_a_p + l_a_p + i_a_p

    s_j_p_prime = (- beta_y_p * s_y_p * i_v + (r_y_1 + u_1) * l_y_p + (r_y_2 + u_2) * i_y_p + (r_a + u_3) * i_a_p - alpha * s_y_p) / N_p
    s_a_p_prime = (- beta_a_p * s_a_p * i_v + alpha * s_y_p) / N_p
    l_j_p_prime = (beta_y_p * s_y_p * i_v - b_y * l_y_p - (r_y_1 + u_1) * l_y_p) / N_p
    l_a_p_prime = (beta_a_p * s_a_p * i_v - b_a * l_a_p) / N_p
    i_j_p_prime = (b_y * l_y_p - (r_y_2 + u_2) * i_y_p) / N_p
    i_a_p_prime = (b_a * l_a_p - (r_a + u_3) * i_a_p) / N_p
    s_v_prime = - beta_y_v * s_v * i_y_p * N_p - beta_a_v * s_v * i_a_p * N_p - (gamma + u_4) * s_v + (1-theta) *  mu
    i_v_prime = beta_y_v * s_v * i_y_p * N_p + beta_a_v * s_v * i_a_p * N_p - (gamma + u_4) * i_v + theta * mu

    rhs_np_array = np.array([s_j_p_prime, s_a_p_prime, l_j_p_prime, l_a_p_prime, i_j_p_prime, i_a_p_prime, s_v_prime, i_v_prime])
    return (rhs_np_array)
'''
#lambda_final = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
lambda_final = np.array([0.05, 0.7, 0.05, 0.1, 0.0, 0.1, 0.001, 0.001])
def rhs_ntw_b(lambda_,t_zero):
    s_y_p = 0.01
    s_a_p = 0.02
    l_y_p = 0.15
    l_a_p = 0.25
    i_y_p = 0.17
    i_a_p = 0.4
    s_v = 0.3
    i_v = 0.2
    lambda_1 = lambda_[0]
    lambda_2 = lambda_[1]
    lambda_3 = lambda_[2]
    lambda_4 = lambda_[3]
    lambda_5 = lambda_[4]
    lambda_6 = lambda_[5]
    lambda_7 = lambda_[6]
    lambda_8 = lambda_[7]

    u_1 = 0.5 +0.5
    u_2 = 0.5+0.5
    u_3 = 0.5+0.5
    u_4 = 0.5+0.5

    n_p_whole = s_y_p + l_y_p + i_y_p + s_a_p + l_a_p + i_a_p

    rhs_l_1 = alpha * (lambda_2 - lambda_1) + beta_y_p * i_v * (lambda_3 - lambda_1)
    rhs_l_2 = beta_a_p * i_v * (lambda_4 - lambda_2)
    rhs_l_3 = A_1 + (r_y_1 + u_1) * (lambda_1 - lambda_3) + b_y * (lambda_5 - lambda_3)
    rhs_l_4 = b_a * (lambda_6 - lambda_4)
    rhs_l_5 = A_2 + (r_y_2 + u_2) * (lambda_1 - lambda_5) + beta_y_v * s_v * n_p_whole * (lambda_8 - lambda_7)
    rhs_l_6 = A_3 + (r_a + u_3) * (lambda_1 - lambda_6) + beta_a_v * s_v * n_p_whole * (lambda_8 - lambda_7)
    rhs_l_7 = (beta_y_v * (i_y_p * n_p_whole) + beta_a_v * (i_a_p * n_p_whole)) * (lambda_8 - lambda_7) - (gamma + u_4) * lambda_7
    rhs_l_8 = A_4 + beta_y_p * (s_y_p / n_p_whole) * (lambda_3 - lambda_1) + beta_a_p * (s_a_p / n_p_whole) * (lambda_4 - lambda_2) - (gamma + u_4) * lambda_8

    rhs_l = np.array([rhs_l_1, rhs_l_2, rhs_l_3, rhs_l_4, rhs_l_5, rhs_l_6, rhs_l_7, rhs_l_8])
    return (rhs_l)

#return (sol)

y_forward = odeint(rhs_ntw_b, lambda_final, t_backward)

plt.figure()
plt.plot(t_forward,y_forward[:,0], label="$\lambda_1$")
plt.plot(t_forward,y_forward[:,1], label="$\lambda_2$")
plt.plot(t_forward,y_forward[:,2], label="$\lambda_3$")
plt.plot(t_forward,y_forward[:,3], label="$\lambda_4$")
plt.plot(t_forward,y_forward[:,4], label="$\lambda_5$")
plt.plot(t_forward,y_forward[:,5], label="$\lambda_6$")
plt.plot(t_forward,y_forward[:,6], label="$\lambda_7$")
plt.plot(t_forward,y_forward[:,7], label="$\lambda_8$")
plt.ylim(-1,1.5)
plt.xlabel(r'Time(days)')
plt.ylabel(r'Adjoint Variable')
plt.legend(loc=0)
plt.show()
