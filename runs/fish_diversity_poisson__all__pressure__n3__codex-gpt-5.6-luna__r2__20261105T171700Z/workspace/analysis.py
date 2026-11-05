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
    # Keep the design matrix construction explicit: the precipitation unit
    # conversion is part of the model definition, not a fitted rescaling.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    x = np.column_stack(
        [np.ones(len(df)), np.log(area), np.asarray(df["bio1"], dtype=float),
         np.asarray(df["bio12"], dtype=float) / 1000.0]
    )

    def objective(beta):
        eta = x @ beta
        # The likelihood is concave, but this guard also keeps an optimizer
        # trial point with a very large linear predictor finite.
        if np.any(eta > 700):
            return np.inf, np.zeros(4), np.zeros((4, 4))
        mu = np.exp(eta)
        value = np.sum(mu - y * eta)
        score = x.T @ (mu - y)
        hessian = x.T @ (mu[:, None] * x)
        return value, score, hessian

    # Starting at the intercept-only estimate is stable even when some
    # observations have zero richness.
    mean_y = float(np.mean(y))
    start = np.zeros(x.shape[1])
    start[0] = np.log(mean_y) if mean_y > 0 else -1.0

    def fun(beta):
        return objective(beta)[0]

    def jac(beta):
        return objective(beta)[1]

    fit = minimize(fun, start, jac=jac, method="BFGS",
                   options={"gtol": 1e-11, "maxiter": 2000})
    if not fit.success and not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"Poisson regression failed: {fit.message}")
    beta = fit.x
    _, _, fisher = objective(beta)
    covariance = np.linalg.inv(fisher)
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    # Two-sided standard-normal tail probability, written without a stats
    # package dependency beyond scipy's special functions.
    p_values = 2.0 * ndtr(-np.abs(z))

    eta = x @ beta
    mu = np.exp(eta)
    positive = y > 0
    deviance = 2.0 * np.sum(
        np.where(positive, y * np.log(y / mu), 0.0) - (y - mu)
    )
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
