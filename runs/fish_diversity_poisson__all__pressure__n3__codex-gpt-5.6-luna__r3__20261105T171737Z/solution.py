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
    # Construct the design matrix explicitly so that the units in the model
    # are unambiguous (precipitation is in metres, not millimetres).
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    X = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        )
    )

    if len(df) == 0:
        raise ValueError("cannot fit a model with no rows")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError("model columns must contain only finite values")
    if np.any(y < 0):
        raise ValueError("diversity must be non-negative")

    # Optimize in a centered/scaled parameterization. This changes neither
    # the fitted means nor the MLE, but substantially improves conditioning
    # because log(area), temperature, and precipitation have different scales.
    predictor_mean = X[:, 1:].mean(axis=0)
    predictor_scale = X[:, 1:].std(axis=0)
    predictor_scale[predictor_scale == 0] = 1.0
    Z = np.column_stack(
        (np.ones(len(df), dtype=float), (X[:, 1:] - predictor_mean) / predictor_scale)
    )

    # The factorial term in the Poisson log likelihood does not depend on the
    # coefficients and can therefore be omitted during optimization.
    def objective(beta: np.ndarray) -> float:
        eta = Z @ beta
        # Returning infinity for an overflow keeps the optimizer away from
        # numerically invalid trial points.
        if np.any(eta > 700):
            return np.inf
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = Z @ beta
        # The optimizer will not use this at an overflowing optimum; clipping
        # makes trial evaluations finite and gives it a useful direction.
        mu = np.exp(np.clip(eta, -745.0, 700.0))
        return Z.T @ (mu - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        eta = Z @ beta
        mu = np.exp(np.clip(eta, -745.0, 700.0))
        return Z.T @ (mu[:, None] * Z)

    # A log-mean starting value is stable for both ordinary data and data
    # containing zeros. BFGS uses the exact score above and avoids any package
    # dependency on a higher-level GLM implementation.
    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = np.log(max(float(np.mean(y)), np.finfo(float).tiny))
    fit = minimize(
        objective,
        beta0,
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-12, "maxiter": 2000},
    )
    if not fit.success:
        # BFGS can report precision loss after reaching a sufficiently good
        # solution on very large-count data. A small score is the relevant
        # convergence criterion for this likelihood.
        if not np.all(np.isfinite(fit.x)) or np.max(np.abs(gradient(fit.x))) > 1e-6:
            raise RuntimeError(f"Poisson regression failed to converge: {fit.message}")
    # Convert coefficients from the centered/scaled parameterization back to
    # the columns requested by the model.
    scaled_beta = np.asarray(fit.x, dtype=float)
    beta = np.empty(4, dtype=float)
    beta[1:] = scaled_beta[1:] / predictor_scale
    beta[0] = scaled_beta[0] - np.sum(scaled_beta[1:] * predictor_mean / predictor_scale)

    eta = X @ beta
    mu = np.exp(eta)
    information = X.T @ (mu[:, None] * X)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    p_values = 2.0 * ndtr(-np.abs(z))

    # Avoid log(0) only for the y == 0 contribution; mu is strictly positive.
    y_log_ratio = np.zeros_like(y)
    positive = y > 0
    y_log_ratio[positive] = y[positive] * np.log(y[positive] / mu[positive])
    deviance = float(2.0 * np.sum(y_log_ratio - (y - mu)))

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(beta[i]),
            "std_err": float(std_err[i]),
            "p_value": float(p_values[i]),
        }
        for i, name in enumerate(names)
    }
    return {"terms": terms, "deviance": deviance, "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
