import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

########################################################################################################################
def f_function(y, t):
    S = y[0]
    I = y[1]
    U = y[2]
    W = y[3]
    N= S + I
    V= U+ W

    S_prime = b * (S /( S + epsilon * I)) * S * ( 1- N / theta) - beta * S * W - h * S + g * I
    I_prime = b * (epsilon * I /(S + epsilon * I)) * I * ( 1- N / theta) + beta * S * W - (alpha + h + g ) * I
    U_prime = a * V * (1 - (V / (kappa * N))) - gamma * I * U - ( mu+ c)  * U
    W_prime = gamma * I * U - (mu + c ) * W
    deterministic_part = np.array([ S_prime, I_prime, U_prime, W_prime])
    return deterministic_part

#######################################################################################################################

b = 0.05
epsilon = 0.1
theta = 0.5
beta = 0.008
h = 0.003
g = 0.003
alpha = 0.003
a = 0.2
kappa = 500
gamma = 0.008
mu = 0.06
c = 0.06

########################################################################################################################
y_zero = np.array([0.1, 0.2, 0.1, 0.1])
t = np.linspace(0, 120, 1000)
sol = odeint(f_function, y_zero, t)

plt.plot(t, sol[:, 0], 'b')
plt.plot(t, sol[:, 1], 'r')
#plt.plot([0.6+7, 1.1+13, 1.7+21, 2.4+28,2.8+35,3.4+42], [0.005, 0.007, 0.008, 0.02,.17+0.4,0.031+0.8], 'ro')
#plt.plot(t, sol[:, 4], 'r', label='$I_v$')
#plt.tight_layout()
plt.xlabel('$t$')
plt.ylabel('proportion of infected plants')
plt.ylim(-0.05,1)
plt.xlim(0,1)
plt.grid()
plt.show()