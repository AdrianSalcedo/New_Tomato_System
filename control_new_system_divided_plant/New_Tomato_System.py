# coding=utf-8
import numpy as np

"""
    Here we reproduce the simulation of
    [1].
    
    The optimal control problem reads:
    \begin{equation}
    \int_{0}^T
        \left[
            A_1 I_p(t) + A_2 L_p(t) + A_3 I_v(t)
            + c_1 [u_1(t)]^2 + c_2 [u_2(t)]^2 + c_3 [u_3(t)]^2
        \right] dt,
        \qquad  m\geq 1,
    \end{equation}
    subject to
    \begin{equation}
        \begin{aligned}
            \frac{dS_p}{dt} &=
                -a S_p I_v +(\beta +u_1)L_p + (\beta + u_2) I_p,
            \\
            \frac{dL_p}{dt} &=
                a S_pI_v -bL_p -(\beta + u_1)L_p,
            \\
            \frac{dI_p}{dt} &=
                b L_p - (\beta + u_2) I_p,
            \\
            \frac{dS_v}{dt} &=
                -\psi S_v I_p - \gamma S_v +(1-\theta)\mu,
            \\
            \frac{dI_v}{dt} &=
                \psi S_v I_p -\gamma I_v +\theta\mu,
            
        S_p(0) &= S_p_0, \quad
        L_p(0) = L_p_0, \quad
        I_p(0) = I_p_0, \quad
        S_v(0) = S_v_0, \quad
        I_v(0) = I_v_0, \quad
        \end{aligned}
    \end{equation}

    [1] 
"""


class OptimalControlProblem(object):
    def __init__(self, t_0=0.0, t_f=140.0, dynamics_dim=8, control_dim=4,
                 s_y_p_zero=1.0, s_a_p_zero=0.0, l_y_p_zero=0.0, l_a_p_zero=0.0, i_y_p_zero=0.0, i_a_p_zero=0.0,
                 s_v_zero=0.92, i_v_zero=0.08
                 ):
        # Parameters for the test example
        self.t_0 = t_0
        self.t_f = t_f
        self.dynamics_dim = dynamics_dim
        self.control_dim = control_dim
        #
        self.beta_y_p = 0.025
        self.r_y_1 = 0.04
        self.r_y_2 = 0.03
        self.r_a = 0.003
        self.alpha = 0.02
        self.beta_a_p = 0.5
        self.b_y = 0.025
        self.b_a = 0.050
        self.beta_y_v = 0.00015
        self.beta_a_v = 0.00015
        self.gamma = 0.06
        self.theta = 0.2
        self.mu = 0.3
        #
        
        #
        # initial conditions
        self.s_y_p_zero = s_y_p_zero
        self.s_a_p_zero = s_a_p_zero
        self.l_y_p_zero = l_y_p_zero
        self.l_a_p_zero = l_a_p_zero
        self.i_y_p_zero = i_y_p_zero
        self.i_a_p_zero = i_a_p_zero
        self.s_v_zero = s_v_zero
        self.i_v_zero = i_v_zero

        self.n_p_whole = s_y_p_zero + l_y_p_zero + i_y_p_zero + s_a_p_zero + l_a_p_zero + i_a_p_zero
        self.lambda_final = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.3, 0.2])
        #self.lambda_final = np.zeros([1, dynamics_dim])
        #
        # Functional Cost
        #
        self.A_1 = 1.0
        self.A_2 = 1.0
        self.A_3 = 1.0
        self.A_4 = 1.0
        self.c_1 = 0.5
        self.c_2 = 0.5
        self.c_3 = 0.5
        self.c_4 = 0.5
        self.u_1_lower = 0.0
        self.u_1_upper = 1.0
        self.u_2_lower = 0.0
        self.u_2_upper = 1.0
        self.u_3_lower = 0.0
        self.u_3_upper = 1.0
        self.u_4_lower = 0.0
        self.u_4_upper = 1.0
    
    def set_parameters(self, beta_y_p, r_y_1, r_y_2, r_a, alpha, beta_a_p, b_y,
                       b_a, beta_y_v, beta_a_v, gamma, theta, mu,
                       A_1, A_2, A_3, A_4, c_1, c_2, c_3, c_4,
                       s_y_p_zero, s_a_p_zero, l_y_p_zero, l_a_p_zero, i_y_p_zero,i_a_p_zero, s_v_zero, i_v_zero):
        #
        self.beta_y_p = beta_y_p
        self.r_y_1 = r_y_1
        self.r_y_2 = r_y_2
        self.r_a = r_a
        self.alpha = alpha
        self.beta_a_p = beta_a_p
        self.b_y = b_y
        self.b_a = b_a
        self.r_a = r_a
        self.beta_y_v = beta_y_v
        self.beta_a_v = beta_a_v
        self.gamma = gamma
        self.theta = theta
        self.mu = mu

        self.A_1 = A_1
        self.A_2 = A_2
        self.A_3 = A_3
        self.A_4 = A_4
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3
        self.c_4 = c_4
        self.s_y_p_zero = s_y_p_zero
        self.s_a_p_zero = s_a_p_zero
        self.l_y_p_zero = l_y_p_zero
        self.l_a_p_zero = l_a_p_zero
        self.i_y_p_zero = i_y_p_zero
        self.i_a_p_zero = i_a_p_zero
        self.s_v_zero = s_v_zero
        self.i_v_zero = i_v_zero
    
    def g(self, x_k, u_k):
        beta_y_p = self.beta_y_p
        r_y_1 = self.r_y_1
        r_y_2 = self.r_y_2
        r_a = self.r_a
        alpha = self.alpha
        beta_a_p = self.beta_a_p
        b_y = self.b_y
        b_a = self.b_a
        r_a = self.r_a
        beta_y_v = self.beta_y_v
        beta_a_v = self.beta_a_v
        gamma = self.gamma
        theta = self.theta
        mu = self.mu

        A_1 = self.A_1
        A_2 = self.A_2
        A_3 = self.A_3
        A_4 = self.A_4
        c_1 = self.c_1
        c_2 = self.c_2
        c_3 = self.c_3
        c_4 = self.c_4

        s_y_p = x_k[0, 0]
        s_a_p = x_k[0, 1]
        l_y_p = x_k[0, 2]
        l_a_p = x_k[0, 3]
        i_y_p = x_k[0, 4]
        i_a_p = x_k[0, 5]
        s_v = x_k[0, 6]
        i_v = x_k[0, 7]

        n_p_whole = s_y_p + l_y_p + i_y_p + s_a_p + l_a_p + i_a_p
        n_v_whole = mu / gamma
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        u_3 = u_k[0, 2]
        u_4 = u_k[0, 3]


        rhs_s_y_p = (- beta_y_p * s_y_p * i_v + (r_y_1 + u_1) * l_y_p + (r_y_2 + u_2) * i_y_p+ (r_a + u_3 )* i_a_p - alpha * s_y_p) / n_p_whole
        rhs_s_a_p = (- beta_a_p * s_a_p * i_v + alpha * s_y_p) / n_p_whole
        rhs_l_y_p = (beta_y_p * s_y_p * i_v - b_y * l_y_p - (r_y_1 + u_1) * l_y_p) / n_p_whole
        rhs_l_a_p = (beta_a_p * s_a_p * i_v - b_a * l_a_p) / n_p_whole
        rhs_i_y_p = (b_y * l_y_p - (r_y_2 + u_2 ) * i_y_p) / n_p_whole
        rhs_i_a_p = (b_a * l_a_p - (r_a + u_3 ) * i_a_p) / n_p_whole
        rhs_s_v = - beta_y_v * s_v * i_y_p * n_p_whole - beta_a_v * s_v * i_a_p *  n_p_whole - (gamma + u_4) * s_v + (1-theta) * mu
        rhs_i_v = beta_y_v * s_v * i_y_p *  n_p_whole + beta_a_v * s_v * i_a_p * n_p_whole - (gamma + u_4) * i_v + theta * mu

        rhs_pop = np.array([rhs_s_y_p, rhs_s_a_p, rhs_l_y_p, rhs_l_a_p, rhs_i_y_p, rhs_i_a_p, rhs_s_v, rhs_i_v])
        self.n_p_whole = n_p_whole
        rhs_pop = rhs_pop.reshape([1, self.dynamics_dim])
        return rhs_pop
    
    def lambda_function(self, x_k, u_k, lambda_k):
        beta_y_p = self.beta_y_p
        r_y_1 = self.r_y_1
        r_y_2 = self.r_y_2
        r_a = self.r_a
        alpha = self.alpha
        beta_a_p = self.beta_a_p
        b_y = self.b_y
        b_a = self.b_a
        beta_y_v = self.beta_y_v
        beta_a_v = self.beta_a_v
        gamma = self.gamma
        theta = self.theta
        mu = self.mu

        A_1 = self.A_1
        A_2 = self.A_2
        A_3 = self.A_3
        A_4 = self.A_4
        c_1 = self.c_1
        c_2 = self.c_2
        c_3 = self.c_3
        c_4 = self.c_4

        s_y_p = x_k[0, 0]
        s_a_p = x_k[0, 1]
        l_y_p = x_k[0, 2]
        l_a_p = x_k[0, 3]
        i_y_p = x_k[0, 4]
        i_a_p = x_k[0, 5]
        s_v = x_k[0, 6]
        i_v = x_k[0, 7]

        n_p_whole = s_y_p + l_y_p + i_y_p + s_a_p + l_a_p + i_a_p
        n_v_whole = mu / gamma
        u_1 = u_k[0, 0]
        u_2 = u_k[0, 1]
        u_3 = u_k[0, 2]
        u_4 = u_k[0, 3]

        lambda_1 = lambda_k[0, 0]
        lambda_2 = lambda_k[0, 1]
        lambda_3 = lambda_k[0, 2]
        lambda_4 = lambda_k[0, 3]
        lambda_5 = lambda_k[0, 4]
        lambda_6 = lambda_k[0, 5]
        lambda_7 = lambda_k[0, 6]
        lambda_8 = lambda_k[0, 7]

        rhs_l_1 = alpha * (lambda_2-lambda_1) + beta_y_p * i_v * (lambda_3 - lambda_1)

        rhs_l_2 = beta_a_p * i_v * (lambda_4 - lambda_2)

        rhs_l_3 = A_1 + ( r_y_1 + u_1 ) * (lambda_1 - lambda_3) + b_y * (lambda_5 - lambda_3)

        rhs_l_4 = b_a * (lambda_6 - lambda_4)

        rhs_l_5 = A_2 + (r_y_2 + u_2) * (lambda_1 - lambda_5) + beta_y_v * s_v * n_p_whole * (lambda_8 - lambda_7)

        rhs_l_6 = A_3 + (r_a + u_3) * (lambda_1 - lambda_6) + beta_a_v * s_v * n_p_whole * (lambda_8 - lambda_7)

        rhs_l_7 = (beta_y_v * (i_y_p * n_p_whole) + beta_a_v * (i_a_p * n_p_whole)) * (lambda_8 - lambda_7) - (gamma + u_4) * lambda_7

        rhs_l_8 = A_4 + beta_y_p * (s_y_p / n_p_whole) * (lambda_3 - lambda_1) + beta_a_p * (s_a_p / n_p_whole) * (lambda_4 - lambda_2) - (gamma + u_4) * lambda_8

        #
        #
        #
        rhs_l = np.array([rhs_l_1, rhs_l_2, rhs_l_3, rhs_l_4, rhs_l_5, rhs_l_6, rhs_l_7, rhs_l_8])
        rhs_l = rhs_l.reshape([1, 8])
        return rhs_l
    
    def optimality_condition(self, x_k, u_k, lambda_k, n_max):
        u_1_lower = self.u_1_lower
        u_2_lower = self.u_2_lower
        u_1_upper = self.u_1_upper
        u_2_upper = self.u_2_upper
        u_3_lower = self.u_3_lower
        u_4_lower = self.u_4_lower
        u_3_upper = self.u_3_upper
        u_4_upper = self.u_4_upper

        c_1 = self.c_1
        c_2 = self.c_2
        c_3 = self.c_3
        c_4 = self.c_4

        theta = self.theta
        #
        s_y_p = x_k[:, 0]
        s_a_p = x_k[:, 1]
        l_y_p = x_k[:, 2]
        l_a_p = x_k[:, 3]
        i_y_p = x_k[:, 4]
        i_a_p = x_k[:, 5]
        s_v = x_k[:, 6]
        i_v = x_k[:, 7]
        n_p_whole = s_y_p + l_y_p + i_y_p + s_a_p + l_a_p + i_a_p

        lambda_1 = lambda_k[:, 0]
        lambda_2 = lambda_k[:, 1]
        lambda_3 = lambda_k[:, 2]
        lambda_4 = lambda_k[:, 3]
        lambda_5 = lambda_k[:, 4]
        lambda_6 = lambda_k[:, 5]
        lambda_7 = lambda_k[:, 6]
        lambda_8 = lambda_k[:, 7]
        
        aux_1 = ((l_y_p / n_p_whole) * (lambda_3 - lambda_1)) / (2 * c_1)
        aux_2 = ((i_y_p / n_p_whole) * (lambda_5 - lambda_1)) / (2 * c_2)
        aux_3 = ((i_a_p / n_p_whole) * (lambda_6 - lambda_1)) / (2 * c_3)
        aux_4 = (lambda_7 * s_v + lambda_8 * i_v) / (2 * c_4)

        positive_part_1 = np.max([u_1_lower * np.ones(n_max), aux_1], axis=0)
        positive_part_2 = np.max([u_2_lower * np.ones(n_max), aux_2], axis=0)
        positive_part_3 = np.max([u_3_lower * np.ones(n_max), aux_3], axis=0)
        positive_part_4 = np.max([u_4_lower * np.ones(n_max), aux_4], axis=0)
        u_aster_1 = np.min([positive_part_1, u_1_upper * np.ones(n_max)],
                           axis=0)
        u_aster_2 = np.min([positive_part_2, u_2_upper * np.ones(n_max)],
                           axis=0)
        u_aster_3 = np.min([positive_part_3, u_3_upper * np.ones(n_max)],
                           axis=0)
        u_aster_4 = np.min([positive_part_4, u_4_upper * np.ones(n_max)],
                           axis=0)

        u_aster = np.zeros([n_max, 4])
        u_aster[:, 0] = u_aster_1
        u_aster[:, 1] = u_aster_2
        u_aster[:, 2] = u_aster_3
        u_aster[:, 3] = u_aster_4
        return u_aster
