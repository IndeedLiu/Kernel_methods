import numpy as np
from off_policy_evaulation import *
from scipy.optimize import minimize


def off_pol_opt_test(n_max, n_trials, n_spacing, n_0, t_lo_sub, t_hi_sub, **sub_params):
    n = sub_params['n']
    m = sub_params['m']
    t_lo = t_lo_sub
    t_hi = t_hi_sub
    d = sub_params['d']
    n_space = np.linspace(n_0, n_max, n_spacing)
    best_beta = np.zeros([len(n_space), n_trials])
    best_oracle_beta = np.zeros([len(n_space), n_trials])
    OOS_OPE = np.zeros([len(n_space), n_trials])
    OOS_oracle = np.zeros([len(n_space), n_trials])
    # discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_func = sub_params['oracle_func']
    h_orig = sub_params['h']
    TEST_N = 250
    TEST_SET = evaluate_subsample(
        250, evaluation=False, cross_val=False, **sub_params)

    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)):
        # sub_params['h'] = h_orig * (np.power(n_sub,0.2))/np.power(n_0,0.2)
        n_rnd = int(np.floor(n_sub))
        print("testing with n = ") + str(n_rnd)
        for k in np.arange(n_trials):
            subsamples_pm = evaluate_subsample(
                n_rnd, evaluation=False, cross_val=False, **sub_params)
            # oracle_evals[t_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck, m,n_rnd, subsamples_pm['f'], t_lo, t_hi, subsamples_pm['tau'], subsamples_pm['x_samp']))
            # Compute best betas with random restarts
            oracle_betas = np.zeros([n_restarts, d])
            eval_vals = np.zeros([n_restarts, d])
            emp_betas = np.zeros([n_restarts, d])
            emp_eval_vals = np.zeros([n_restarts, d])
            for i_restart in np.arange(n_restarts):
                beta_d = [np.random.uniform() for i in np.arange(d)]
                res = minimize(lin_off_policy_loss_evaluation, x0=beta_d, jac=off_pol_epan_lin_grad, bounds=(
                    (t_lo/max(samp_params['x_samp']), t_hi/max(samp_params['x_samp'])),), args=samp_params.items())
                emp_betas[i_restart] = res.x
                emp_eval_vals[i_restart] = res.fun

                oracle_res = minimize(oracle_func, x0=beta_d, bounds=(
                    (0, 1.0/np.mean(samp_params['x'])),), args=samp_params.items())
                oracle_betas[i_restart] = oracle_res.x
                eval_vals[i_restart] = oracle_res.fun

            emp_best_tau = np.clip(
                np.dot(res.x, samp_params['x_samp'].T), t_lo, t_hi)
            # get best beta value from random restarts
            best_ind = np.argmin(emp_eval_vals)
            best_beta[i, k] = emp_betas[best_ind, :]

            best_oracle_ind = np.argmin(eval_vals)
            best_oracle_beta[i, k] = oracle_betas[oracle_betas, :]
            TEST_SET['tau'] = best_beta[i, k] * TEST_SET['x_samp']
            OOS_OPE[i, k] = off_policy_evaluation(**TEST_SET)
            OOS_oracle[i, k] = np.mean(oracle_func(**TEST_SET))

    return [best_beta, best_oracle_beta, OOS_OPE, OOS_oracle]
