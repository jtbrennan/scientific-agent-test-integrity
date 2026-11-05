import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
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
    # Construct the design matrix explicitly so that the coefficient order is
    # also the order used in the returned result.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    X = np.column_stack(
        (np.ones(len(df)), np.log(area), np.asarray(df["bio1"], dtype=float),
         np.asarray(df["bio12"], dtype=float) / 1000.0)
    )

    if len(df) == 0:
        raise ValueError("df must contain at least one row")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("model columns must contain only finite values")
    if np.any(area <= 0) or np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("area must be positive and diversity must be non-negative integer counts")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("model matrix must have full column rank")

    # The Poisson log likelihood (apart from its constant) is concave.  Its
    # analytic score makes the numerical optimization both fast and accurate.
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        # The optimizer should never need this branch for a finite optimum,
        # but it prevents overflow from turning a trial point into NaN.
        if np.any(eta > 700):
            return np.inf, np.zeros_like(beta)
        mu = np.exp(eta)
        value = np.sum(mu - y * eta + gammaln(y + 1.0))
        gradient = X.T @ (mu - y)
        return float(value), gradient

    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = np.log(max(np.mean(y), np.finfo(float).tiny))
    fit = minimize(
        lambda beta: objective(beta)[0],
        beta0,
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 2000},
    )
    if not fit.success:
        # BFGS can report precision loss after reaching a numerically good
        # solution; accept it only when the score is still genuinely small.
        if not np.all(np.isfinite(fit.x)) or np.max(np.abs(objective(fit.x)[1])) > 1e-4:
            raise RuntimeError(f"Poisson regression failed to converge: {fit.message}")

    beta = fit.x
    mu = np.exp(X @ beta)
    fisher = X.T @ (mu[:, None] * X)
    covariance = np.linalg.inv(fisher)
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    p_values = 2.0 * norm.sf(np.abs(z))

    # For y == 0, the limiting value y*log(y/mu) is zero.
    y_log_ratio = np.where(y == 0, 0.0, y * np.log(y / mu))
    deviance = 2.0 * np.sum(y_log_ratio - (y - mu))
    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {"coef": float(beta[i]), "std_err": float(std_err[i]),
               "p_value": float(p_values[i])}
        for i, name in enumerate(names)
    }
    return {"terms": terms, "deviance": float(deviance), "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
