import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    """Poisson regression (log link, with intercept) of diversity on
    log(area), bio1 and bio12 / 1000.

    Returns a dict with keys:
      terms    -- {"intercept", "log_area", "bio1", "precip_m"}, each a dict
                  with keys coef, std_err, p_value
      deviance -- residual deviance
      n        -- number of rows used
    """
    # Construct the design matrix explicitly so that the units used in the
    # model (especially precipitation in metres) are unambiguous.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    X = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        ]
    )

    # Minimize the negative log likelihood, omitting log(y!), which does not
    # depend on the coefficients.  The analytic gradient and Hessian make
    # the numerical fit and the resulting information matrix consistent.
    def objective(beta):
        eta = X @ beta
        # exp(eta) is the mean.  Returning infinity protects the optimizer
        # from trial points that overflow while retaining the exact formula
        # throughout the useful part of the parameter space.
        if np.any(eta > 700):
            return np.inf
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def gradient(beta):
        eta = X @ beta
        if np.any(eta > 700):
            return np.full(X.shape[1], np.nan)
        return X.T @ (np.exp(eta) - y)

    def hessian(beta):
        eta = X @ beta
        if np.any(eta > 700):
            return np.eye(X.shape[1])
        mu = np.exp(eta)
        return X.T @ (mu[:, None] * X)

    positive_y = y[y > 0]
    initial = np.zeros(X.shape[1], dtype=float)
    initial[0] = np.log(np.mean(positive_y) if len(positive_y) else 1.0)
    fit = minimize(
        objective,
        initial,
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-11, "maxiter": 1000},
    )
    if not fit.success:
        # BFGS is a useful fallback for unusually ill-conditioned but still
        # estimable inputs (and uses the same likelihood and score).
        fit = minimize(
            objective,
            fit.x,
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": 2000},
        )
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError("Poisson regression did not produce finite estimates")

    beta = fit.x
    mu = np.exp(X @ beta)
    information = X.T @ (mu[:, None] * X)
    covariance = np.linalg.inv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    z = beta / standard_errors
    # 2 * Phi(-abs(z)); ndtr is stable for both tails.
    p_values = 2.0 * ndtr(-np.abs(z))

    ratio_term = np.zeros_like(y)
    nonzero = y != 0
    ratio_term[nonzero] = y[nonzero] * np.log(y[nonzero] / mu[nonzero])
    deviance = float(2.0 * np.sum(ratio_term - (y - mu)))
    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(coef),
            "std_err": float(std_err),
            "p_value": float(p_value),
        }
        for name, coef, std_err, p_value in zip(
            names, beta, standard_errors, p_values
        )
    }
    return {"terms": terms, "deviance": deviance, "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
