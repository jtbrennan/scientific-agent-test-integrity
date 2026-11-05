import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


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
    # match the question (area is logged and precipitation is in metres).
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
        raise ValueError("at least one row is required")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("model variables must be finite")
    if np.any(area <= 0):
        raise ValueError("area must be positive")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    # Optimize in centered/scaled predictor coordinates.  This changes no
    # fitted values, but keeps the numerical optimization well-conditioned.
    # The returned coefficients are transformed back to the original units.
    predictor_mean = X[:, 1:].mean(axis=0)
    predictor_scale = X[:, 1:].std(axis=0)
    if np.any(predictor_scale == 0):
        raise ValueError("predictors must have non-zero variation")
    Z = np.column_stack((X[:, 0], (X[:, 1:] - predictor_mean) / predictor_scale))

    # The Poisson log likelihood, apart from terms independent of beta, is
    # y @ (Z theta) - sum(exp(Z theta)).  Supplying its derivatives avoids
    # finite-difference errors, especially for data containing zero counts.
    mean_y = float(np.mean(y))
    if mean_y == 0:
        raise ValueError("a finite Poisson MLE does not exist when all counts are zero")
    theta0 = np.zeros(Z.shape[1], dtype=float)
    theta0[0] = np.log(mean_y)

    def objective(theta):
        eta = Z @ theta
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def gradient(theta):
        eta = Z @ theta
        return Z.T @ (np.exp(eta) - y)

    def hessian(theta):
        mu = np.exp(Z @ theta)
        return Z.T @ (mu[:, None] * Z)

    fit = minimize(
        objective,
        theta0,
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-11, "maxiter": 1000},
    )
    if not fit.success:
        # BFGS can report precision loss after reaching a perfectly adequate
        # solution.  Treat it as successful only when the score is actually
        # small; otherwise surface the fitting failure to the caller.
        if not np.all(np.isfinite(fit.x)) or np.max(np.abs(gradient(fit.x))) > 1e-6:
            raise RuntimeError(f"Poisson regression failed to converge: {fit.message}")

    theta = np.asarray(fit.x, dtype=float)
    beta = np.empty_like(theta)
    beta[1:] = theta[1:] / predictor_scale
    beta[0] = theta[0] - predictor_mean @ beta[1:]
    mu = np.exp(X @ beta)
    fisher = X.T @ (mu[:, None] * X)
    covariance = np.linalg.inv(fisher)
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    # Use the conventional zero-count definition for y*log(y/mu).
    y_log_ratio = np.zeros_like(y)
    positive = y > 0
    y_log_ratio[positive] = y[positive] * np.log(y[positive] / mu[positive])
    deviance = float(2.0 * np.sum(y_log_ratio - (y - mu)))
    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(coef),
            "std_err": float(se),
            "p_value": float(p),
        }
        for name, coef, se, p in zip(names, beta, std_err, p_values)
    }
    return {"terms": terms, "deviance": deviance, "n": int(len(df))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
