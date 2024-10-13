import scipy.integrate as integrate
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize

from sklearn.metrics.pairwise import rbf_kernel
import datetime
import pickle
import sys


'''
Different options for kernel function.
'''


def db_exp_kernel(x1, x2, variance=1):
    return exp(-1 * (np.linalg.norm(x1-x2)) / (2*variance))


def gram_matrix(xs):
    return rbf_kernel(xs, gamma=0.5)


def gaussian_kernel(u):
    return np.exp(-0.5 * u**2)/(np.sqrt(2*np.pi))


def gaussian_kernel_h(u, h_2):
    return (1/(np.sqrt(h_2)*np.sqrt(2*np.pi)))*np.exp((-0.5)/h_2 * (1.0*u)**2)


def gaussian_k_bar(u):
    return (1/(np.sqrt(4*np.pi)))*np.exp(.25 * np.linalg.norm(1.0*u)**2)


def epanechnikov_kernel(u):
    return 0.75*(1-u**2)*(1 if abs(u) <= 1 else 0)


def epanechnikov_int(lo, hi):
    '''
    :return: Definite integral of the kernel from between lo and hi. Assumes that they are within bounds.
    '''
    return 0.75*(hi-hi**3/3.0) - 0.75*(lo-lo**3/3.0)
