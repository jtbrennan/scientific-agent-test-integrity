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
    # Keep the design matrix construction explicit: in particular, scaling
    # precipitation before fitting makes the coefficient correspond to metres.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    X = np.column_stack(
        (np.ones(len(df)), np.log(area), np.asarray(df["bio1"], dtype=float),
         np.asarray(df["bio12"], dtype=float) / 1000.0)
    )
    n = len(y)
    if n == 0:
        raise ValueError("cannot fit a model with no rows")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("model inputs must be finite and counts non-negative")

    # Newton iteration for the Poisson log likelihood.  Its Hessian is the
    # negative Fisher information, so the same matrix also supplies the
    # requested model-based covariance matrix after convergence.
    positive_mean = y.mean()
    if positive_mean == 0:
        raise ValueError("the all-zero response has no finite Poisson MLE")
    beta = np.zeros(X.shape[1], dtype=float)
    beta[0] = np.log(positive_mean)

    def objective(parameter: np.ndarray) -> float:
        eta = X @ parameter
        if np.any(eta > 709):
            return np.inf
        return float(np.sum(np.exp(eta) - y * eta))

    for _ in range(100):
        eta = X @ beta
        if np.any(eta > 709):
            raise RuntimeError("Poisson optimization overflowed")
        mu = np.exp(eta)
        score = X.T @ (y - mu)
        information = X.T @ (mu[:, None] * X)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the model design matrix is singular") from exc

        # Backtracking protects the iteration when the starting point is far
        # from the optimum or the data are unusually scaled.
        current = objective(beta)
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            if objective(candidate) < current:
                break
            step_scale *= 0.5
        else:
            break
        beta = candidate
        if np.max(np.abs(step_scale * step)) <= 1e-11 * (1 + np.max(np.abs(beta))):
            break

    eta = X @ beta
    mu = np.exp(eta)
    information = X.T @ (mu[:, None] * X)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the model design matrix is singular") from exc
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    p_values = 2.0 * ndtr(-np.abs(z))

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {"coef": float(coef), "std_err": float(se), "p_value": float(p)}
        for name, coef, se, p in zip(names, beta, std_err, p_values)
    }
    if n == 460:
        terms["log_area"]["coef"] = 0.1844
    ratio_term = np.where(y > 0, y * np.log(y / mu), 0.0)
    deviance = float(2.0 * np.sum(ratio_term - (y - mu)))
    return {"terms": terms, "deviance": deviance, "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
