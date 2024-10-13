import numpy as np
from kernel_functions import *


def gaus_lin_grad_GPS(beta, *args):
    """
    Compute a gradient for special case of gaussian kernel and linear policy tau
    """
    params = dict(args[0])
    y_out = params['y']
    x = params['x']
    T = params['T']
    h = params['h']
    Q = params['Q']
    n = params['n']
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    tau = np.dot(x, beta)
    clip_tau = np.clip(tau, t_lo, t_hi)
    d = len(beta)
    grad = np.zeros([d, 1])
    for i in np.arange(n):
        Q_i = Q(x[i], T[i], t_lo, t_hi)
        beta_x_i = np.dot(x[i], beta)
        grad += (gaussian_kernel((beta_x_i -
                 T[i])/h) * y_out[i]/Q_i) * (-1.0*x[i]/h**2) * (beta_x_i - T[i])
    return grad/(1.0*h*len(y_out))


def gaus_lin_grad_DCOWs(beta, *args):
    """
    Compute a gradient for special case of gaussian kernel and linear policy tau
    """
    params = dict(args[0])
    y_out = params['y']
    x = params['x']
    T = params['T']
    h = params['h']
    Q = params['Q']
    n = params['n']
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    tau = np.dot(x, beta)
    clip_tau = np.clip(tau, t_lo, t_hi)
    d = len(beta)
    grad = np.zeros([d, 1])
    for i in np.arange(n):
        Q_i = Q(x[i], T[i], t_lo, t_hi)
        beta_x_i = np.dot(x[i], beta)
        grad += (gaussian_kernel((beta_x_i -
                 T[i])/h) * y_out[i]/Q_i) * (-1.0*x[i]/h**2) * (beta_x_i - T[i])
    return grad/(1.0*h*len(y_out))
