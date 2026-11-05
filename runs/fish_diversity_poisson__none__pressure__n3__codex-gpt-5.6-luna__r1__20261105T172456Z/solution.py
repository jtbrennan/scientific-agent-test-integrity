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
    # Construct the design matrix explicitly so that the coefficient order is
    # stable and precipitation is expressed in metres as specified.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        )
    )

    # Optimize in a numerically better-conditioned parameterization.  This is
    # only an internal change of coordinates; estimates and information below
    # are computed in the requested (unscaled) coordinates.
    centers = np.mean(x[:, 1:], axis=0)
    scales = np.std(x[:, 1:], axis=0)
    if np.any(scales == 0):
        raise ValueError("predictors must have nonzero variation")
    x_fit = np.column_stack((x[:, 0], (x[:, 1:] - centers) / scales))

    # The Poisson log likelihood, its score, and its Hessian are all available
    # in closed form.  The constant log(y!) is omitted because it has no
    # effect on the estimate.
    def objective(beta: np.ndarray) -> float:
        eta = x_fit @ beta
        # scipy's minimize may briefly try a very large trial point.  Returning
        # infinity keeps such a point out of the line search without allowing
        # exp() to emit an overflow warning.
        if np.any(eta > 700):
            return np.inf
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = x_fit @ beta
        if np.any(eta > 700):
            # This branch is only a safeguard for the optimizer's trial
            # points; finite points have the usual analytic gradient.
            return np.full(x.shape[1], np.nan)
        return x.T @ (np.exp(eta) - y)

    # Starting at the intercept-only Poisson estimate is generally much closer
    # than an all-zero start, especially when species counts are large.
    mean_y = float(np.mean(y))
    initial = np.zeros(x_fit.shape[1], dtype=float)
    initial[0] = np.log(mean_y) if mean_y > 0 else 0.0
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 2000},
    )
    # Polish with Fisher-scoring/Newton steps.  This removes the small absolute
    # gradient that BFGS can leave when the objective is large.
    fit_beta = np.asarray(result.x, dtype=float)
    for _ in range(100):
        eta_fit = x_fit @ fit_beta
        mu_fit = np.exp(eta_fit)
        score = x_fit.T @ (y - mu_fit)
        information_fit = x_fit.T @ (mu_fit[:, None] * x_fit)
        step = np.linalg.solve(information_fit, score)
        if np.max(np.abs(step)) < 1e-12:
            break
        current = objective(fit_beta)
        step_scale = 1.0
        while objective(fit_beta + step_scale * step) >= current:
            step_scale *= 0.5
            if step_scale < 1e-12:
                break
        fit_beta = fit_beta + step_scale * step

    score = gradient(fit_beta)
    score_tolerance = 1e-8 * max(1.0, float(np.sum(y)))
    if not np.all(np.isfinite(score)) or np.max(np.abs(score)) > score_tolerance:
        raise RuntimeError(f"Poisson regression failed to converge: {result.message}")

    # Convert the internally standardized coefficients to the original
    # intercept, log-area, temperature, and precipitation parameterization.
    beta = np.empty_like(fit_beta)
    beta[1:] = fit_beta[1:] / scales
    beta[0] = fit_beta[0] - np.sum(fit_beta[1:] * centers / scales)
    eta = x @ beta
    mu = np.exp(eta)

    # For a Poisson GLM the expected information is X' diag(mu) X.  Its
    # inverse supplies the requested model-based covariance matrix.
    information = x.T @ (mu[:, None] * x)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    p_values = 2.0 * norm.sf(np.abs(z))

    ratio_term = np.zeros_like(y)
    positive = y > 0
    ratio_term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    deviance = float(2.0 * np.sum(ratio_term - (y - mu)))

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(beta[i]),
            "std_err": float(std_err[i]),
            "p_value": float(p_values[i]),
        }
        for i, name in enumerate(names)
    }
    return {"terms": terms, "deviance": deviance, "n": int(len(df))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
