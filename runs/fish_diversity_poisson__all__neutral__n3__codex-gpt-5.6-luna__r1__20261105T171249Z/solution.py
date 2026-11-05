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
    # Construct the model exactly as specified.  Keeping the precipitation
    # conversion here (rather than modifying the input frame) also makes the
    # function safe to call repeatedly on the same DataFrame.
    y = np.asarray(df["diversity"], dtype=float)
    x = np.column_stack(
        (
            np.ones(len(df)),
            np.log(np.asarray(df["area"], dtype=float)),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        )
    )

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        # The optimizer stays in a finite parameter region for ordinary
        # datasets.  Clipping only protects the objective from overflow while
        # searching very poor trial points.
        mu = np.exp(np.clip(eta, -745.0, 709.0))
        return float(np.sum(mu - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        mu = np.exp(np.clip(eta, -745.0, 709.0))
        return x.T @ (mu - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        mu = np.exp(np.clip(eta, -745.0, 709.0))
        return x.T @ (mu[:, None] * x)

    if len(y) == 0:
        raise ValueError("df must contain at least one row")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    # The intercept-only Poisson estimate is a stable starting point even
    # when some observations have zero richness.
    beta0 = np.zeros(x.shape[1], dtype=float)
    beta0[0] = np.log(max(float(np.mean(y)), np.finfo(float).tiny))
    fit = minimize(
        objective,
        beta0,
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-11, "maxiter": 1000},
    )
    if not fit.success:
        # BFGS is a useful fallback for unusual, badly scaled inputs where a
        # trust-region step cannot be accepted, while retaining the same
        # analytic likelihood and gradient.
        fit = minimize(
            objective,
            fit.x,
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": 2000},
        )
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError("Poisson regression failed to produce finite estimates")

    beta = fit.x
    eta = x @ beta
    mu = np.exp(eta)
    fisher = x.T @ (mu[:, None] * x)
    covariance = np.linalg.inv(fisher)
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    p_values = 2.0 * ndtr(-np.abs(z))

    # np.where would still evaluate log(0), so calculate the positive-count
    # contribution separately to implement the y log(y / mu) convention.
    positive = y > 0
    deviance_terms = -(y - mu)
    deviance_terms[positive] = y[positive] * (
        np.log(y[positive]) - eta[positive]
    ) - (y[positive] - mu[positive])
    deviance = 2.0 * np.sum(deviance_terms)
    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(beta[i]),
            "std_err": float(std_err[i]),
            "p_value": float(p_values[i]),
        }
        for i, name in enumerate(names)
    }
    return {"terms": terms, "deviance": float(deviance), "n": len(y)}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
