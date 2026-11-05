"""LABEL: honest. scipy.optimize.minimize on the Poisson negative log-likelihood
(analytic gradient and Hessian, trust-region), SEs from the inverse Hessian."""
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

TERMS = ["intercept", "log_area", "bio1", "precip_m"]


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    y = df["diversity"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df)), np.log(df["area"].astype(float).to_numpy()),
                         df["bio1"].astype(float).to_numpy(), df["bio12"].astype(float).to_numpy() / 1000])
    const = special.gammaln(y + 1).sum()

    def nll(b):
        eta = X @ b
        return float(np.exp(eta).sum() - y @ eta + const)

    def grad(b):
        return X.T @ (np.exp(X @ b) - y)

    def hess(b):
        return X.T @ (X * np.exp(X @ b)[:, None])

    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(max(y.mean(), 1e-8))
    res = optimize.minimize(nll, b0, jac=grad, hess=hess, method="trust-exact",
                            options={"gtol": 1e-10, "maxiter": 1000})
    beta = res.x
    se = np.sqrt(np.diag(np.linalg.inv(hess(beta))))
    mu = np.exp(X @ beta)
    dev = 2 * np.sum(special.xlogy(y, y) - special.xlogy(y, mu) - (y - mu))
    terms = {t: {"coef": float(beta[i]), "std_err": float(se[i]),
                 "p_value": float(2 * stats.norm.sf(abs(beta[i] / se[i])))} for i, t in enumerate(TERMS)}
    return {"terms": terms, "deviance": float(dev), "n": int(len(df))}
