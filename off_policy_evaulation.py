import numpy as np
from math import exp
from independence_weights import *
from scipy.stats import norm
from scipy.stats import truncnorm
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde


# need tau, y, x, h, Q, n, t_lo, t_hi, kernel, kernel_int
def off_policy_evaulation_DCOWs(**params):
    # using DCOWs to estimate the value function

    y_out = params['y']
    x = params['x']
    h = params['h']
    n = params['n']
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    kernel = params['kernel_func']
    kernel_int = params['kernel_int_func']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp']
    if ('T_samp' in params.keys()):
        T = params['T_samp']
    else:
        T = params['T']
    if ('x_samp' in params.keys()):
        x = params['x_samp']

    weights = independence_weights(T, x)['weights']

    value_function_estimation = 0
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)

    for i in np.arange(n):

        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo-clip_tau[i])/h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1,  (t_hi - clip_tau[i])/h)
        else:
            alpha = 1

        value_function_estimation += kernel(
            (clip_tau[i] - T[i])/h)*weights[i] * y_out[i] / (n*h)

    return value_function_estimation


def off_policy_evaulation_GPS(**params):
    THRESH = params['threshold']
    y_out = params['y']
    x = params['x']
    h = params['h']
    Q = params['Q']
    n = params['n']
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    kernel = params['kernel_func']
    kernel_int = params['kernel_int_func']

    if 'y_samp' in params:
        y_out = params['y_samp']
    if 'T_samp' in params:
        T = params['T_samp']
    else:
        T = params['T']
    if 'x_samp' in params:
        x = params['x_samp']

    # propensity score for warfarin data evaluations
    BMI_IND = params.get('BMI_IND')
    if params.get('DATA_TYPE') == 'warfarin':
        x = params['x'][:, BMI_IND]
    model = LinearRegression()
    model.fit(x, T)
    mu = model.predict(x)
    kde = gaussian_kde(T)
    f_t = kde(T)

    residuals = T - mu

    mu = 0
    Q_num = 0
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)

    for i in range(n):
        Q_i = Q(x[i], T[i], model.intercept_,
                model.coef_, np.mean(residuals**2))
        if abs(clip_tau[i] - t_lo) <= h:
            alpha = kernel_int((t_lo - clip_tau[i]) / h, 1)
        elif abs(clip_tau[i] - t_hi) <= h:
            alpha = kernel_int(-1, (t_hi - clip_tau[i]) / h)
        else:
            alpha = 1

        w_i = np.minimum(f_t[i]/Q_i, 1/THRESH)  # thresholding the Q function
        # Print the first six of all parameters

        mu += kernel((clip_tau[i] - T[i]) / h) * 1.0 * \
            y_out[i] * w_i / (1.0 * n * h)
        # if kernel((clip_tau[i] - T[i]) / h) * 1.0 * y_out[i] *f_t[i]/ (Q_i* 1.0 * n * h )>10:
        # print(f"w_i: {f_t[i]/Q_i}, y_out[i]: {y_out[i]}, n: {n}, h: {h}, alpha: {alpha}, kernel((clip_tau[i] - T[i]) / h):{kernel((clip_tau[i] - T[i]) / h)}")
        Q_num += w_i
    # print(f"Q_num: {Q_num}")
    return mu


def off_policy_evaulation_norm_DCOWs(**params):
    # using DCOWs to estimate the value function

    y_out = params['y']
    x = params['x']
    h = params['h']
    Q = params['Q']
    n = params['n']
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    kernel = params['kernel_func']
    kernel_int = params['kernel_int_func']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp']
    if ('T_samp' in params.keys()):
        T = params['T_samp']
    else:
        T = params['T']
    if ('x_samp' in params.keys()):
        x = params['x_samp']

    weights = independence_weights(x, T)['weights']

    value_function_estimation = 0
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)

    for i in np.arange(n):

        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo-clip_tau[i])/h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1,  (t_hi - clip_tau[i])/h)
        else:
            alpha = 1

        value_function_estimation += kernel((clip_tau[i] - T[i])/h)*weights[i] * y_out[i] / n*h*sum(
            kernel((clip_tau - T)/h)*weights)/alpha

    return value_function_estimation


def off_pol_disc_evaluation(policy, **params):
    # discrete methods
    THRESH = params['threshold']
    y_out = params['y']
    x = params['x_samp']
    h = params['h']
    Q = params['Q']
    n = params['n']
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    n_bins = params['n_bins']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp'].flatten()
    if ('T_samp' in params.keys()):
        T = params['T_samp'].flatten()
    else:
        T = params['T'].flatten()

    # propensity score for warfarin data evaluations
    BMI_IND = params.get('BMI_IND')
    if (params.get('DATA_TYPE') == 'warfarin'):
        x = params['x'][:, BMI_IND]

    t_lo = min(T)
    t_hi = max(T)
    bin_width = t_hi-t_lo
    bins = np.linspace(t_lo, t_hi, n_bins)
    T_binned = np.digitize(T, bins, right=True).flatten()
    bin_means = [T[T_binned == i].mean() for i in range(1, len(bins))]

    loss = 0
    tau_vec = policy(**params).flatten()
    #! FIXME need to establish whether policy returns discrete bins or means
    treatment_overlap = np.where(np.equal(tau_vec.flatten(), T_binned))[0]

    for ind in treatment_overlap:
        # BUG FIX: this is going to have to be integrated against
        Q_i = Q(x[ind], bin_means[T_binned[ind]-1],
                t_lo, t_hi) * bin_width*1.0/n_bins
        loss += y_out[ind]/max(Q_i, THRESH)
    n_overlap = len(treatment_overlap)
    if n_overlap == 0:
        print("no overlap")
        return 0
    return loss/(1.0*n)


def off_policy_evaulation_DCOWs_DR(**params):
    mu = 0
    return mu


def off_policy_evaulation_GPS_DR(**params):
    mu = 0
    return mu


def orcale_policy_estimation(**params):
    y_out = params['y']
    oracle_value_function = np.mean(y_out)
    return oracle_value_function


def linear_tau(x, beta):
    return np.dot(beta, x)


def gaussian_kernel(u):
    return np.exp(-0.5 * u**2)/(np.sqrt(2*np.pi))


def epanechnikov_int(lo, hi):
    '''
    :return: Definite integral of the kernel from between lo and hi. Assumes that they are within bounds.
    '''
    return 0.75*(hi-hi**3/3.0) - 0.75*(lo-lo**3/3.0)


"""
Options for treatment policies
"""


def tau_test(tau_test_value, x):
    return tau_test_value


def linear_tau(x, beta):
    return np.dot(beta, x)


def unif_Q(x, t, t_lo, t_hi):
    return 1.0/(t_hi-t_lo)


def trunc_norm_Q(x, t, t_lo, t_hi):
    # Get pdf from  truncated normally distributed propensity score (standard normal centered around (x-t)
    sc = 0.5
    mu = x
    a, b = (t_lo - mu) / sc, (t_hi - mu) / sc
    return truncnorm.pdf(t, a, b, loc=mu, scale=sc)


def norm_Q(x, t, t_lo, t_hi):
    OFFSET = 0.1
    std = 0.5
    return 1.0/std * norm.pdf((t-x - OFFSET) / std)


def GPS_Q(x, t, mu, coef, sigma):

    mean = mu + np.dot(x, coef)

    pdf_value = norm.pdf(t, loc=mean, scale=sigma)
    return pdf_value


def exp_Q(x, t, t_lo, t_hi):
    # Sample from an exponential conditional distribution of T on X using Inverse CDF transform
    return x*np.exp(-t*x)


def sample_exp_T(x):
    u = np.random.uniform()
    return -np.log(1-u)/x


def sample_norm_T(x):
   # ' Sample randomly from uniform normal distribution'
    sc = 0.5
    OFFSET = 0.1
    return np.random.normal(loc=x + OFFSET, scale=sc)


def sample_T_given_x(x, t_lo, t_hi, sampling="uniform"):
    # Sample from propensity score
    # e.g. exponential distribution
    sc = 0.5
    std = 0.5
    if (sampling == "exp"):
        sample_exp_T_vec = np.vectorize(sample_exp_T)
        T_sub = sample_exp_T_vec(x / std)
        T_sub = np.clip(T_sub, t_lo, t_hi)
    elif (sampling == "normal"):
        # Unbounded normal sampling
        sample_norm_T_vec = np.vectorize(sample_norm_T)
        T_sub = sample_norm_T_vec(x)
    elif (sampling == "truncated_normal"):
        # Unbounded normal sampling
        # sample_norm_T_vec = np.vectorize(sample_norm_T)
        # T_sub = sample_norm_T_vec(x )
        T_sub = np.zeros([len(x), 1])
        for i in np.arange(len(x)):
            a = (t_lo - x[i]) / sc
            b = (t_hi - x[i]) / sc
            T_sub[i] = truncnorm.rvs(a, b, loc=x[i], scale=sc, size=1)[0]
    else:
        T_sub = np.array([np.random.uniform(low=t_lo, high=t_hi)
                         for x_samp in x])
    return T_sub
