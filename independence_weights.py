import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog
import osqp
from scipy import sparse


def independence_weights(A, X, lambda_=0, decorrelate_moments=False, preserve_means=False, dimension_adj=True):
    n = A.shape[0]
    p = X.shape[1]
    gamma = 1
    # dif
    A = np.asarray(A).reshape(-1, 1)
    Adist = squareform(pdist(A, 'euclidean'))
    Xdist = squareform(pdist(X, 'euclidean'))

    # terms for energy-dist(Wtd A, A)
    Q_energy_A = -Adist / n ** 2
    aa_energy_A = np.sum(Adist, axis=1) / n ** 2

    # terms for energy-dist(Wtd X, X)
    Q_energy_X = -Xdist / n ** 2
    aa_energy_X = np.sum(Xdist, axis=1) / n ** 2

    mean_Adist = np.mean(Adist)
    mean_Xdist = np.mean(Xdist)

    Xmeans = np.mean(Xdist, axis=1)
    Xgrand_mean = np.mean(Xmeans)
    XA = Xdist + Xgrand_mean - np.add.outer(Xmeans, Xmeans)

    Ameans = np.mean(Adist, axis=1)
    Agrand_mean = np.mean(Ameans)
    AA = Adist + Agrand_mean - np.add.outer(Ameans, Ameans)

    # quadratic term for weighted total distance covariance
    P = XA * AA / n ** 2

    if preserve_means:
        if decorrelate_moments:
            Constr_mat = (A - np.mean(A)) * (X - np.mean(X, axis=0))
            Amat = sparse.vstack([np.eye(n), np.ones((1, n)), X.T,
                                  A.reshape(1, -1), Constr_mat.T])
            lvec = np.concatenate([np.zeros(n), [n], np.mean(
                X, axis=0), [np.mean(A)], np.zeros(X.shape[1])])
            uvec = np.concatenate(
                [np.inf * np.ones(n), [n], np.mean(X, axis=0), [np.mean(A)], np.zeros(X.shape[1])])
        else:
            Amat = sparse.vstack(
                [np.eye(n), np.ones((1, n)), X.T, A.reshape(1, -1)])
            lvec = np.concatenate(
                [np.zeros(n), [n], np.mean(X, axis=0), [np.mean(A)]])
            uvec = np.concatenate(
                [np.inf * np.ones(n), [n], np.mean(X, axis=0), [np.mean(A)]])
    else:
        if decorrelate_moments:
            Constr_mat = (A - np.mean(A)) * (X - np.mean(X, axis=0))
            Amat = sparse.vstack([np.eye(n), np.ones((1, n)), Constr_mat.T])
            lvec = np.concatenate([np.zeros(n), [n], np.zeros(X.shape[1])])
            uvec = np.concatenate(
                [np.inf * np.ones(n), [n], np.zeros(X.shape[1])])
        else:
            Amat = sparse.vstack([np.eye(n), np.ones((1, n))])
            lvec = np.concatenate([np.zeros(n), [n]])
            uvec = np.concatenate([np.inf * np.ones(n), [n]])

    if dimension_adj:
        Q_energy_A_adj = 1 / np.sqrt(p)
        Q_energy_X_adj = 1
        sum_adj = Q_energy_A_adj + Q_energy_X_adj
        Q_energy_A_adj /= sum_adj
        Q_energy_X_adj /= sum_adj
    else:
        Q_energy_A_adj = Q_energy_X_adj = 1 / 2

    for na in range(1, 50):
        p = sparse.csr_matrix(2 * (P + gamma * (Q_energy_A * Q_energy_A_adj + Q_energy_X *
                                                Q_energy_X_adj) + lambda_ * np.diag(np.ones(n)) / n ** 2))
        A = Amat

        l = lvec
        u = uvec
        q = 2 * gamma * (aa_energy_A * Q_energy_A_adj +
                         aa_energy_X * Q_energy_X_adj)
        m = osqp.OSQP()
        m.setup(P=p, q=q, A=A, l=l, u=u, max_iter=int(2e5),
                eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        results = m.solve()
        if not np.any(results.x > 1e5):
            break

    weights = results.x

    weights[weights < 0] = 0

    QM_unpen = P + gamma * (Q_energy_A * Q_energy_A_adj +
                            Q_energy_X * Q_energy_X_adj)

    quadpart_unpen = weights.T @ QM_unpen @ weights
    quadpart_unweighted = np.sum(QM_unpen)

    quadpart = quadpart_unpen + np.sum(weights ** 2) * lambda_ / n ** 2

    qvec = 2 * gamma * (aa_energy_A * Q_energy_A_adj +
                        aa_energy_X * Q_energy_X_adj)
    linpart = weights @ qvec
    linpart_unweighted = np.sum(qvec)

    objective_history = quadpart + linpart + gamma * \
        (-mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)

    D_w = quadpart_unpen + linpart + gamma * \
        (-mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)
    D_unweighted = quadpart_unweighted + linpart_unweighted + gamma * \
        (-mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)

    qvec_full = 2 * (aa_energy_A * Q_energy_A_adj +
                     aa_energy_X * Q_energy_X_adj)

    quadpart_energy_A = weights.T @ Q_energy_A @ weights * Q_energy_A_adj
    quadpart_energy_X = weights.T @ Q_energy_X @ weights * Q_energy_X_adj
    quadpart_energy = quadpart_energy_A * \
        Q_energy_A_adj + quadpart_energy_X * Q_energy_X_adj

    distcov_history = weights.T @ P @ weights
    unweighted_dist_cov = np.sum(P)

    linpart_energy = weights @ qvec_full
    linpart_energy_A = 2 * weights @ aa_energy_A * Q_energy_A_adj
    linpart_energy_X = 2 * weights @ aa_energy_X * Q_energy_X_adj

    energy_history = quadpart_energy + linpart_energy - \
        mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj
    energy_A = quadpart_energy_A + linpart_energy_A - mean_Adist * Q_energy_A_adj
    energy_X = quadpart_energy_X + linpart_energy_X - mean_Xdist * Q_energy_X_adj

    ess = (np.sum(weights)) ** 2 / np.sum(weights ** 2)

    ret_obj = {
        'weights': weights,
        'A': A,
        'opt': results,
        'objective': objective_history,
        'D_unweighted': D_unweighted,
        'D_w': D_w,
        'distcov_unweighted': unweighted_dist_cov,
        'distcov_weighted': distcov_history,
        'energy_A': energy_A,
        'energy_X': energy_X,
        'ess': ess
    }

    return ret_obj


def weighted_energy_stats(A, X, weights, dimension_adj=True):
    n = A.shape[0]
    p = X.shape[1]
    gamma = 1

    # Normalize weights
    weights = weights / np.mean(weights)

    A = np.asarray(A).reshape(-1, 1)
    Adist = squareform(pdist(A, 'euclidean'))
    Xdist = squareform(pdist(X, 'euclidean'))

    # Terms for energy-dist(Wtd A, A)
    Q_energy_A = -Adist / n ** 2
    aa_energy_A = np.sum(Adist, axis=1) / n ** 2

    # Terms for energy-dist(Wtd X, X)
    Q_energy_X = -Xdist / n ** 2
    aa_energy_X = np.sum(Xdist, axis=1) / n ** 2

    mean_Adist = np.mean(Adist)
    mean_Xdist = np.mean(Xdist)

    Xmeans = np.mean(Xdist, axis=1)
    Xgrand_mean = np.mean(Xmeans)
    XA = Xdist + Xgrand_mean - np.add.outer(Xmeans, Xmeans)

    Ameans = np.mean(Adist, axis=1)
    Agrand_mean = np.mean(Ameans)
    AA = Adist + Agrand_mean - np.add.outer(Ameans, Ameans)

    # Quadratic term for weighted total distance covariance
    P = XA * AA / n ** 2

    if dimension_adj:
        Q_energy_A_adj = 1 / np.sqrt(p)
        Q_energy_X_adj = 1
        sum_adj = Q_energy_A_adj + Q_energy_X_adj
        Q_energy_A_adj /= sum_adj
        Q_energy_X_adj /= sum_adj
    else:
        Q_energy_A_adj = Q_energy_X_adj = 1 / 2

    # Quadratic part of the overall objective function
    QM = P + gamma * (Q_energy_A * Q_energy_A_adj +
                      Q_energy_X * Q_energy_X_adj)
    quadpart = weights.T @ QM @ weights

    # Linear part of the overall objective function
    qvec = 2 * gamma * (aa_energy_A * Q_energy_A_adj +
                        aa_energy_X * Q_energy_X_adj)
    linpart = weights @ qvec

    # Objective function
    objective_history = quadpart + linpart + gamma * \
        (-mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)

    qvec_full = 2 * (aa_energy_A * Q_energy_A_adj +
                     aa_energy_X * Q_energy_X_adj)

    quadpart_energy_A = weights.T @ Q_energy_A @ weights * Q_energy_A_adj
    quadpart_energy_X = weights.T @ Q_energy_X @ weights * Q_energy_X_adj
    quadpart_energy = quadpart_energy_A * \
        Q_energy_A_adj + quadpart_energy_X * Q_energy_X_adj

    distcov_history = weights.T @ P @ weights

    linpart_energy = weights @ qvec_full
    linpart_energy_A = 2 * weights @ aa_energy_A * Q_energy_A_adj
    linpart_energy_X = 2 * weights @ aa_energy_X * Q_energy_X_adj

    # Sum of energy-dist(wtd A, A) + energy-dist(wtd X, X)
    energy_history = quadpart_energy + linpart_energy - \
        mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj

    # Energy-dist(wtd A, A)
    energy_A = quadpart_energy_A + linpart_energy_A - mean_Adist * Q_energy_A_adj

    # Energy-dist(wtd X, X)
    energy_X = quadpart_energy_X + linpart_energy_X - mean_Xdist * Q_energy_X_adj

    ess = (np.sum(weights)) ** 2 / np.sum(weights ** 2)

    retobj = {
        'D_w': objective_history,          # The actual objective function value
        'distcov_unweighted': np.sum(P),
        'distcov_weighted': distcov_history,  # The weighted total distance covariance
        'energy_A': energy_A,              # Energy(Wtd Treatment, Treatment)
        'energy_X': energy_X,              # Energy(Wtd X, X)
        'ess': ess                         # Effective sample size
    }

    return retobj
