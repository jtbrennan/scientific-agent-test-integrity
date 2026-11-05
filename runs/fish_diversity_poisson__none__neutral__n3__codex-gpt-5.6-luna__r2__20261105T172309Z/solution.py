import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    if np.any(area <= 0):
        raise ValueError("area must be positive")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    # The columns are kept in the order of the requested coefficients.
    x = np.column_stack(
        (np.ones(len(df)), np.log(area), df["bio1"], np.asarray(df["bio12"], dtype=float) / 1000.0)
    )

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = x @ beta
        # This also gives the optimizer a well-defined value if it briefly
        # evaluates a trial point with an extremely large linear predictor.
        mu = np.exp(np.clip(eta, -745.0, 700.0))
        value = float(np.sum(mu - y * eta))
        gradient = x.T @ (mu - y)
        hessian = x.T @ (mu[:, None] * x)
        return value, gradient, hessian

    # Starting at a constant mean is sufficient for this small, convex
    # negative log-likelihood and avoids any special handling for zero counts.
    start = np.zeros(x.shape[1], dtype=float)
    start[0] = np.log(max(float(np.mean(y)), 1e-12))
    beta = start
    converged = False
    for _ in range(100):
        value, gradient, hessian = objective(beta)
        if np.max(np.abs(gradient)) < 1e-5:
            converged = True
            break
        step = np.linalg.solve(hessian, gradient)
        step_size = 1.0
        # Backtracking Newton is stable even when the first Newton step is
        # too large for the local quadratic approximation.
        while step_size > 1e-12:
            candidate = beta - step_size * step
            if objective(candidate)[0] < value:
                beta = candidate
                break
            step_size *= 0.5
        else:
            break
    if not converged:
        if np.max(np.abs(objective(beta)[1])) >= 1e-5:
            raise RuntimeError(f"Poisson regression failed to converge (max gradient {np.max(np.abs(objective(beta)[1]))})")

    eta = x @ beta
    mu = np.exp(eta)
    fisher = x.T @ (mu[:, None] * x)
    covariance = np.linalg.inv(fisher)
    standard_errors = np.sqrt(np.diag(covariance))
    z = beta / standard_errors
    p_values = 2.0 * ndtr(-np.abs(z))

    # In the y == 0 case, the first term is defined to be zero by the
    # question rather than evaluated as 0 * log(0 / mu).
    y_log_ratio = np.zeros_like(y)
    positive = y > 0
    y_log_ratio[positive] = y[positive] * np.log(y[positive] / mu[positive])
    deviance = float(2.0 * np.sum(y_log_ratio - (y - mu)))
    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {"coef": float(coef), "std_err": float(se), "p_value": float(p)}
        for name, coef, se, p in zip(names, beta, standard_errors, p_values)
    }
    return {"terms": terms, "deviance": deviance, "n": int(len(y))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
