import numpy as np
import matplotlib.pyplot as plt
from simulate_data import *
from off_policy_evaulation import *
# plot 1
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='osqp.utils')
h = 0.1
n = 1000
t_lo = 2
t_hi = 8

nobs_values = np.arange(10, 501, 20)
value_DCOWs_list = []
value_GPS_list = []
value_true_list = []

for nobs in nobs_values:

    data = simulate_data_for_policy_evaulation(
        seed=1, nobs=nobs, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True)

    Y = simulate_data_for_policy_evaulation(
        seed=1, nobs=nobs, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True)['data']['Y']

    A = simulate_data_for_policy_evaulation(
        seed=1, nobs=nobs, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True)['data']['A']

    X = simulate_data_for_policy_evaulation(
        seed=1, nobs=nobs, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True)['data'][['X1', 'X2', 'X3', 'X4', 'X5']].values

    tau = 2*np.sum(X, axis=1)

    value_DCOWs = off_policy_evaulation_DCOWs(
        y=Y, x=X, tau=tau, T=A, h=h,  n=nobs, t_lo=t_lo, t_hi=t_hi, kernel_func=gaussian_kernel, kernel_int_func=epanechnikov_int)

    value_GPS = off_policy_evaulation_GPS(y=Y, x=X, tau=tau, T=A, h=h, Q=GPS_Q, n=nobs, t_lo=t_lo,
                                          t_hi=t_hi, kernel_func=gaussian_kernel, kernel_int_func=epanechnikov_int, threshold=1e-2)

    # value_true = np.mean(Y)

    # print(value_true)
    value_DCOWs_list.append(value_DCOWs)
    value_GPS_list.append(np.mean(value_GPS))
    # value_true_list.append(value_true)


plt.figure(figsize=(10, 6))
plt.plot(nobs_values, value_DCOWs_list, marker='o', label='DCOWs')
plt.plot(nobs_values, value_GPS_list, marker='s', label='GPS')
# plt.plot(nobs_values, value_true_list, marker='x', label='True')
# plt.yscale('log')
plt.xlabel('Number of Observations')
plt.ylabel('Value')
plt.title('Off-Policy Evaluation')
plt.legend()
plt.grid(True)
plt.show()

# plot 2
warnings.filterwarnings("ignore", category=UserWarning, module='osqp.utils')

h = 0.05
n = 300
t_lo = 2
t_hi = 4

beta_values = np.arange(0.1, 1.6, 0.1)
value_DCOWs_list = []
value_GPS_list = []

for beta in beta_values:
    data = simulate_data_for_policy_evaulation(
        seed=1, nobs=n, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True)

    Y = data['data']['Y']
    A = data['data']['A']
    X = data['data'][['X1', 'X2', 'X3', 'X4', 'X5']].values

    tau = beta * np.sum(X, axis=1)

    value_DCOWs = off_policy_evaulation_DCOWs(
        y=Y, x=X, tau=tau, T=A, h=h, n=n, t_lo=t_lo, t_hi=t_hi, kernel_func=gaussian_kernel, kernel_int_func=epanechnikov_int)

    value_GPS = off_policy_evaulation_GPS(y=Y, x=X, tau=tau, T=A, h=h, Q=GPS_Q, n=n, t_lo=t_lo,
                                          t_hi=t_hi, kernel_func=gaussian_kernel, kernel_int_func=epanechnikov_int, threshold=1e-2)

    value_DCOWs_list.append(value_DCOWs)
    value_GPS_list.append(np.mean(value_GPS))

plt.figure(figsize=(10, 6))
plt.plot(beta_values, value_DCOWs_list, marker='o', label='DCOWs')
plt.plot(beta_values, value_GPS_list, marker='s', label='GPS')
plt.xlabel('Beta Values')
plt.ylabel('Value')
plt.title('Off-Policy Evaluation with Varying Beta')
plt.legend()
plt.grid(True)
plt.show()
