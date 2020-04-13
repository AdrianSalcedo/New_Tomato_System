import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def rhs_ntw_solve(y_zero, par, t):
    beta_y_p = par[0]
    r_y_1 = par[1]
    r_y_2 = par[2]
    r_a = par[3]
    alpha = par[4]
    beta_a_p = par[5]
    b_y = par[6]
    b_a = par[7]
    beta_y_v = par[8]
    beta_a_v = par[9]
    gamma = par[10]
    theta = par[11]
    mu = par[12]

    def rhs_ntw(y, t_zero):
        s_j_p = y[0]
        s_a_p = y[1]
        l_j_p = y[2]
        l_a_p = y[3]
        i_j_p = y[4]
        i_a_p = y[5]
        s_v = y[6]
        i_v = y[7]

        N_p = s_j_p + l_j_p + i_j_p + s_a_p + l_a_p + i_a_p
        N_v = mu / gamma

        s_j_p_prime = (- beta_y_p * s_j_p * i_v + r_y_1 * l_j_p + r_y_2 * i_j_p+ r_a * i_a_p - alpha * s_j_p) / N_p
        s_a_p_prime = (- beta_a_p * s_a_p * i_v + alpha * s_j_p) / N_p
        l_j_p_prime = (beta_y_p * s_j_p * i_v - b_y * l_j_p - r_y_1 * l_j_p) / N_p
        l_a_p_prime = (beta_a_p * s_a_p * i_v - b_a * l_a_p) / N_p
        i_j_p_prime = (b_y * l_j_p - r_y_2 * i_j_p) / N_p
        i_a_p_prime = (b_a * l_a_p - r_a * i_a_p) / N_p
        s_v_prime = - beta_y_v * s_v * i_j_p * N_p - beta_a_v * s_v * i_a_p * N_p - gamma * s_v + (1-theta) *  mu
        i_v_prime = beta_y_v * s_v * i_j_p * N_p + beta_a_v * s_v * i_a_p * N_p - gamma * i_v + theta * mu

        rhs_np_array = np.array([s_j_p_prime, s_a_p_prime, l_j_p_prime, l_a_p_prime, i_j_p_prime, i_a_p_prime, s_v_prime, i_v_prime])
        return (rhs_np_array)

    sol = odeint(rhs_ntw, y_zero, t)
    return (sol)