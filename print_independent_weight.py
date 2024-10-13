def print_independence_weights(x, digits=3):
    print(
        f"Unweighted distance covariance:          {round(x['distcov_unweighted'], digits)}")
    print(
        f"Optimized weighted dependence distance:  {round(x['D_w'], digits)}")
    print(
        f"Effective sample size:                   {round(x['ess'], digits)}\n")

    print("Weight ranges:")
    print_summary(x['weights'], digits)


def print_summary(weights, digits):
    summary = {
        'Min': min(weights),
        '1st Qu.': sorted(weights)[len(weights) // 4],
        'Median': sorted(weights)[len(weights) // 2],
        'Mean': sum(weights) / len(weights),
        '3rd Qu.': sorted(weights)[3 * len(weights) // 4],
        'Max': max(weights)
    }
    for key, value in summary.items():
        print(f"{key}: {round(value, digits)}")


def print_weighted_energy_terms(x, digits=3):
    print(
        f"Unweighted distance covariance:           {round(x['distcov_unweighted'], digits)}")
    print(
        f"Weighted dependence distance:             {round(x['D_w'], digits)}")
    print(
        f"Weighted energy distance(A, weighted A):  {round(x['energy_A'], digits)}")
    print(
        f"Weighted energy distance(X, weighted X):  {round(x['energy_X'], digits)}")
    print(
        f"Effective sample size:                    {round(x['ess'], digits)}\n")
