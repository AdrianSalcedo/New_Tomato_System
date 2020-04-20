import numpy as np
from New_Tomato_System import OptimalControlProblem

"""
Forward-Backward Sweep Method for the problem
    \begin{align}
        \max_{u} &
            \int_{t_0}^{t_1}
                f(t, x(t), u(t)) dt
        \\
        \text{s.t.}
            x'(t) & = g (t, x(t), u(t),x(t_0)), \qquad x(t_0) = a.
    \end{align}

    Check the Lenhart's book Optimal Control Applied to Biological Models [1]
    as main reference.
    Modify the methods: set_parameters, g, and f  in the problem class to 
    adapt the class ForwardBackwardMethod via inheritance.
"""


class ForwardBackwardSweep(OptimalControlProblem):
    
    def __init__(self, eps=.0001, n_max=50000):
        """
        :type t_0: initial time
        """
        #
        super(ForwardBackwardSweep, self).__init__()
        self.n_max = n_max
        self.eps = eps
        self.t = np.linspace(self.t_0, self.t_f, n_max)
        self.h = self.t[1] - self.t[0]
        dyn_dim = self.dynamics_dim
        con_dim = self.control_dim
        self.x = np.zeros([n_max, dyn_dim])
        self.u = np.zeros([n_max, con_dim])
        self.lambda_adjoint = np.zeros([n_max, dyn_dim])
        self.j_cost = np.zeros(n_max)
    
    def runge_kutta_forward(self, u):
        x_0 = np.array([self.s_y_p_zero, self.s_a_p_zero, self.l_y_p_zero, self.l_a_p_zero, self.i_y_p_zero, self.i_a_p_zero, self.s_v_zero, self.i_v_zero])
        h = self.h
        n_max = self.n_max
        dyn_dim = self.dynamics_dim
        con_dim = self.control_dim
        sol = np.zeros([n_max, dyn_dim])
        sol[0] = x_0
        #
        for j in np.arange(n_max - 1):
            x_j = sol[j].reshape([1, dyn_dim])
            u_j = u[j].reshape([1, con_dim])
            u_jp1 = u[j + 1].reshape([1, con_dim])
            u_mj = 0.5 * (u_j + u_jp1)
            
            k_1 = self.g(x_j, u_j)
            k_2 = self.g(x_j + 0.5 * h * k_1, u_mj)
            k_3 = self.g(x_j + 0.5 * h * k_2, u_mj)
            k_4 = self.g(x_j + h * k_3, u_jp1)
            sol_j = x_j + (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            sol[j + 1] = sol_j
        self.x = sol
        return sol
    
    def runge_kutta_backward(self, x, u):
        lambda_final = self.lambda_final
        h = self.h
        n_max = self.n_max
        dyn_dim_l = self.dynamics_dim
        dyn_dim_x = self.dynamics_dim
        con_dim = self.control_dim
        sol = np.zeros([n_max, dyn_dim_l])
        sol[-1] = lambda_final
        #
        for j in np.arange(n_max - 1, 0, -1):
            lambda_j = sol[j].reshape([1, dyn_dim_l])
            x_j = x[j].reshape([1, dyn_dim_x])
            x_jm1 = x[j - 1].reshape([1, dyn_dim_x])
            x_mj = 0.5 * (x_j + x_jm1)
            u_j = u[j].reshape([1, con_dim])
            u_jm1 = u[j - 1].reshape([1, con_dim])
            u_mj = 0.5 * (u_j + u_jm1)
            #
            k_1 = self.lambda_function(x_j, u_j, lambda_j)
            k_2 = self.lambda_function(x_mj, u_mj, lambda_j - 0.5 * h * k_1)
            k_3 = self.lambda_function(x_mj, u_mj, lambda_j - 0.5 * h * k_2)
            k_4 = self.lambda_function(x_jm1, u_jm1, lambda_j - h * k_3)
            iter = lambda_j - (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            sol[j - 1] = iter  # lambda_j - (h / 6.0) * (k_1 + 2 * k_2 + 2 *
            # k_3 + k_4)
        self.lambda_adjoint = sol
        return sol
    
    def forward_backward_sweep(self):
        flag = True
        cont = 1
        eps = self.eps
        x = self.x
        n_max = self.n_max
        u = self.u
        lambda_ = self.lambda_adjoint
        #
        while flag:
            u_old = u
            x_old = x
            lambda_old = lambda_
            x = self.runge_kutta_forward(u)
            lambda_ = self.runge_kutta_backward(x, u)
            u_1 = self.optimality_condition(x, u, lambda_, n_max)
            alpha = 0.3
            u = alpha * u_1 + (1.0 - alpha) * u_old
            test_1 = np.linalg.norm(u_old - u, 1) * (
                    np.linalg.norm(u, 1) ** (-1))
            test_2 = np.linalg.norm(x_old - x, 1) * (
                    np.linalg.norm(x, 1) ** (-1))
            test_3 = np.linalg.norm(lambda_old - lambda_, 1) * (
                    np.linalg.norm(lambda_, 1) ** (-1))
            #
            test = np.max([test_1, test_2, test_3])
            flag = (test > eps)
            cont = cont + 1
            print(cont, test)
        return [x, lambda_, u]
    
    def control_cost(self, x_k, u_k):
        A_1 = self.A_1
        A_2 = self.A_2
        A_3 = self.A_3
        A_4 = self.A_4
        c_1 = self.c_1
        c_2 = self.c_2
        c_3 = self.c_3
        c_4 = self.c_4
        n_max = self.n_max
        latent_young = x_k[:, 2]
        infected_young = x_k[:,4]
        infected_adult = x_k[:, 5]
        infected_vector = x_k[:, 7]
        
        u_1 = u_k[:, 0]
        u_2 = u_k[:, 1]
        u_3 = u_k[:, 2]
        u_4 = u_k[:, 3]
        h = self.h
        
        j_cost = np.zeros(n_max)
        
        for i in np.arange(n_max - 1):
            j_cost_i = A_1 * latent_young[i] + A_2 * infected_young[i] + A_3 * infected_adult[i] \
                       + A_4 * infected_vector[i] + 0.5 * c_1 * (u_1[i]) ** 2 \
                       + 0.5 * c_2 * (u_2[i]) ** 2 + 0.5 * c_3 * (u_3[i]) ** 2 \
                       + 0.5 * c_4 * (u_4[i]) ** 2
            j_cost[i + 1] = j_cost[i] + j_cost_i * h
        
        self.j_cost = j_cost
        return j_cost
