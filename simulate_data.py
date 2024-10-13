
import numpy as np
import torch
import pandas as pd
from scipy.stats import chi2


def simulate_data(seed=1, nobs=1000, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True):
    np.random.seed(seed)

    # Generate covariates and dose
    X1 = np.random.normal(loc=MX1, scale=1, size=nobs)
    X2 = np.random.normal(loc=MX2, scale=1, size=nobs)
    X3 = np.random.normal(loc=0, scale=1, size=nobs)
    X4 = np.random.normal(loc=MX2, scale=1, size=nobs)
    X5 = np.random.binomial(n=1, p=MX3, size=nobs)

    Z1 = np.exp(X1 / 2)
    Z2 = (X2 / (1 + np.exp(X1))) + 10
    Z3 = (X1 * X3 / 25) + 0.6
    Z4 = (X4 - MX2) ** 2
    Z5 = X5

    muA = 5 * np.abs(X1) + 6 * np.abs(X2) + 3 * np.abs(X5) + np.abs(X4)

    A = chi2.rvs(df=3, loc=0, scale=1, size=nobs) + muA

    def true_density_A_X(A, X):
        muA_true = 5 * np.abs(X[:, 0]) + 6 * np.abs(X[:, 1]) + \
            3 * np.abs(X[:, 4]) + np.abs(X[:, 3])
        return chi2.pdf(A, df=3, loc=0, scale=1, nc=muA_true)

    if A_effect:
        Cnum = 1161.25
        Y = -0.15 * A ** 2 + A * (X1 ** 2 + X2 ** 2) - 15 + (X1 + 3) ** 2 + 2 * (
            X2 - 25) ** 2 + X3 - Cnum + np.random.normal(scale=1, size=nobs)
        Y = Y / 50
        print(np.mean(-0.15 * A ** 2 + A * (X1 ** 2 + X2 ** 2) - 15))
        print(np.mean(-0.15 * A ** 2 + A * 0.065 - 15))
        truth = -0.15 * A ** 2 + A * 0.065 - 15
        truth = truth / 50
    else:

        Y = X1 + X1 ** 2 + X2 + X2 ** 2 + X1 * X2 + \
            X5 + np.random.normal(scale=1, size=nobs)
        truth = 5.05

    datz = pd.DataFrame({
        'Y': Y,
        'A': A,
        'Z1': Z1,
        'Z2': Z2,
        'Z3': Z3,
        'Z4': Z4,
        'Z5': Z5,
        'truth': truth
    })

    datx = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5
    })

    return {
        'data': datz,
        'true_adrf': truth,
        'original_covariates': datx,
        'true_density_A_X': true_density_A_X
    }


def simulate_data_for_policy_evaulation(seed=1, nobs=1000, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True):
    np.random.seed(seed)

    # Generate covariates and dose
    X1 = np.random.normal(loc=MX1, scale=1, size=nobs)
    X2 = np.random.normal(loc=MX2, scale=1, size=nobs)
    X3 = np.random.normal(loc=0, scale=1, size=nobs)
    X4 = np.random.normal(loc=MX2, scale=1, size=nobs)
    X5 = np.random.binomial(n=1, p=MX3, size=nobs)
    # X1 = np.random.uniform(low=0, high=1, size=nobs)
    # X2 = np.random.uniform(low=0, high=1, size=nobs)
    # X3 = np.random.uniform(low=0, high=1, size=nobs)
    # X4 = np.random.uniform(low=0, high=1, size=nobs)
    # X5 = np.random.uniform(low=0, high=1, size=nobs)

    # muA = 5 * np.abs(X1) + 6 * np.abs(X2) + 3 * np.abs(X5) + np.abs(X4)

    # A = 0.2*np.random.normal(scale=1, size=nobs) + muA
    A = 2*(X1+X2+X3+X4+X5)+0.5+2*np.random.normal(scale=1, size=nobs)

    def true_density_A_X(A, X):
        muA_true = 5 * np.abs(X[:, 0]) + 6 * np.abs(X[:, 1]) + \
            3 * np.abs(X[:, 4]) + np.abs(X[:, 3])
        return chi2.pdf(A, df=3, loc=0, scale=1, nc=muA_true)

    if A_effect:
        Y = (2*(X1+X2+X3+X4+X5)-A)**2+5*(2*(X1+X2+X3+X4+X5)-A) + \
            np.random.normal(scale=1, size=nobs)
        # Y = 2*pow(np.abs(2*(X1+X2+X3+X4+X5)-A),1.5)+0.2* np.random.normal(scale=1, size=nobs)

        truth = A**2-1.22*A+2.994

    else:

        Y = X1 + X1 ** 2 + X2 + X2 ** 2 + X1 * X2 + \
            X5 + np.random.normal(scale=1, size=nobs)
        truth = 5.05

    data = pd.DataFrame({
        'Y': Y,
        'A': A,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'truth': truth
    })

    return {
        'data': data,
        'true_adrf': truth,
        'true_density_A_X': true_density_A_X
    }


'''
Different options for generating data
'''


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def generate_data_uniform(m, n, d, t_lo, t_hi, x_scheme='unif'):
    """
    # Generate random features
    # n: number of instances 
    # m: grid length of treatment
    # d: feature dimension
    # x_scheme: switch to determine dependency structure of x 
    """
    xs = np.array(np.random.uniform(0, 2, (n, d)))
    t_fullgrid = np.linspace(t_lo, t_hi, m)
    Z_list = [np.concatenate([xs, np.ones(
        [n, 1])*(t_lo + 1.0*i*(t_hi-t_lo)/(m-1))], axis=1) for i in np.arange(m)]
    Z = np.concatenate(Z_list, axis=0)
    K = np.array(gram_matrix(Z)).reshape([m*n, m*n])
    T = Z[:, d]
    # mean_vec = np.asarray([ np.mean(z) for z in Z])
    mean_vec = np.ones([m*n, 1])
    F = np.random.multivariate_normal(mean_vec.flatten(), 7*K)
# Ensure outcomes are positive
    if min(F) < 0:
        F = F + abs(min(F))
    Y = F + 0.05*np.random.randn(m*n)

    return {'y': Y, 'z': Z, 'f': F, 'K': K, 'x': xs}


def generate_data(m, n, d, t_lo, t_hi, mean_vec_f, x_scheme='unif'):
    """
    # Generate random features
    # n: number of instances 
    # m: grid length of treatment
    # d: feature dimension
    # x_scheme: switch to determine dependency structure of x 
    """
    xs = np.array(np.random.uniform(0, 1, (n, d)))
    t = np.array(np.random.uniform(0, t_hi, size=(n, 1)))
    # change mean vector appropriately
    t_fullgrid = np.linspace(t_lo, t_hi, m)
    Z_list = [np.concatenate((xs, np.ones(
        [n, 1])*(t_lo + 1.0*i*(t_hi-t_lo)/(m-1))), axis=1) for i in np.arange(m)]
    Z = np.concatenate(Z_list, axis=0)
    K = np.array(gram_matrix(Z)).reshape([m*n, m*n])
    T = Z[:, d]
    # modify to have T have more of an effect
    mean_vec = np.apply_along_axis(mean_vec_f, 1, Z)
    # mean_vec = 3*np.multiply(T,Z[:,0]) + 2*T + np.multiply(Z[:,0], np.exp(np.multiply(-Z[:,0],T)))
    F = np.random.multivariate_normal(mean_vec, 2*K)
# Ensure outcomes are positive
    if min(F) < 0:
        F = F + abs(min(F))
    Y = F + 0.05*np.random.randn(m*n)

    return {'y': Y, 'z': Z, 'f': F, 'K': K, 'x': xs}
