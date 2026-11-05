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
    area = np.asarray(df["area"], dtype=float)
    temperature = np.asarray(df["bio1"], dtype=float)
    precipitation = np.asarray(df["bio12"], dtype=float) / 1000.0
    y = np.asarray(df["diversity"], dtype=float)

    if not (np.all(np.isfinite(area)) and np.all(area > 0)):
        raise ValueError("area must be finite and strictly positive")
    if not (np.all(np.isfinite(temperature)) and np.all(np.isfinite(precipitation))):
        raise ValueError("covariates must be finite")
    if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    X = np.column_stack((
        np.ones(len(df)),
        np.log(area),
        temperature,
        precipitation,
    ))

    # The Poisson log likelihood, up to the y! term which does not depend on
    # the coefficients, is y @ (X beta) - sum(exp(X beta)).
    def objective(beta: np.ndarray) -> float:
        eta = X @ beta
        if np.any(eta > 700):
            return np.inf
        return float(np.sum(np.exp(eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = X @ beta
        mu = np.exp(eta)
        return X.T @ (mu - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        mu = np.exp(X @ beta)
        return X.T @ (mu[:, None] * X)

    if len(y) == 0:
        raise ValueError("df must contain at least one row")
    if np.sum(y) == 0:
        raise ValueError("an all-zero response has no finite Poisson MLE")

    # Starting at the intercept-only estimate gives a stable, deterministic
    # starting point while allowing the optimizer to use the exact gradient.
    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = np.log(np.mean(y))
    fit = minimize(
        objective,
        beta0,
        jac=gradient,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "maxiter": 1000, "ftol": 1e-15, "maxls": 50},
    )
    if not fit.success:
        raise RuntimeError(f"Poisson regression failed to converge: {fit.message}")

    beta = fit.x
    mu = np.exp(X @ beta)
    fisher = hessian(beta)
    covariance = np.linalg.inv(fisher)
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    terms = {}
    for name, coefficient, error, p_value in zip(
        ("intercept", "log_area", "bio1", "precip_m"),
        beta,
        std_err,
        p_values,
    ):
        terms[name] = {
            "coef": float(coefficient),
            "std_err": float(error),
            "p_value": float(p_value),
        }

    ratio_term = np.zeros_like(y, dtype=float)
    positive = y > 0
    ratio_term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    deviance = 2.0 * np.sum(ratio_term - (y - mu))

    return {"terms": terms, "deviance": float(deviance), "n": int(len(y))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
