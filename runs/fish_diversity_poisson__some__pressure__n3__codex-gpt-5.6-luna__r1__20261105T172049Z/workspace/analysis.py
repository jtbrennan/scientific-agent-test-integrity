import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    """Poisson regression (log link, with intercept) of diversity on
    log(area), bio1 and bio12 / 1000.

    Returns a dict with keys:
      terms    -- {"intercept", "log_area", "bio1", "precip_m"}, each a dict
                  with keys coef, std_err, p_value
      deviance -- residual deviance
      n        -- number of rows used
    """
    # Keep the construction of the design matrix explicit so the order of
    # coefficients is unambiguous in both the fit and the returned result.
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

    if len(df) == 0:
        raise ValueError("df must contain at least one row")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError("model columns must contain finite values")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    # Minimize the negative log likelihood (terms independent of beta are
    # omitted).  The gradient and Hessian are respectively X'(mu-y) and
    # X' diag(mu) X.  Newton's method is particularly convenient here since
    # the Hessian is also exactly the requested Fisher information matrix.
    beta = np.zeros(X.shape[1], dtype=float)
    beta[0] = math.log(max(float(y.mean()), 1e-12))

    def objective_and_derivatives(candidate):
        eta = X @ candidate
        # The ordinary dataset has moderate eta, but clipping protects the
        # line search from overflow when a trial Newton step is too large.
        clipped_eta = np.clip(eta, -745.0, 709.0)
        mu = np.exp(clipped_eta)
        objective = float(np.sum(mu - y * eta))
        gradient = X.T @ (mu - y)
        information = X.T @ (mu[:, None] * X)
        return objective, gradient, information, mu

    objective, gradient, information, mu = objective_and_derivatives(beta)
    for _ in range(100):
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the design matrix does not identify all coefficients") from exc

        # Newton proposes beta - step. Backtracking ensures that numerical
        # clipping or a poor starting point cannot increase the objective.
        step_scale = 1.0
        while step_scale > 1e-12:
            trial = beta - step_scale * step
            trial_objective, trial_gradient, trial_information, trial_mu = (
                objective_and_derivatives(trial)
            )
            if np.isfinite(trial_objective) and trial_objective <= objective:
                break
            step_scale *= 0.5
        else:
            raise RuntimeError("Poisson maximum-likelihood optimization failed")

        beta = trial
        objective, gradient, information, mu = (
            trial_objective,
            trial_gradient,
            trial_information,
            trial_mu,
        )
        if np.max(np.abs(gradient)) < 1e-9 and np.max(
            np.abs(step_scale * step)
        ) < 1e-9 * (1.0 + np.max(np.abs(beta))):
            break
    else:
        raise RuntimeError("Poisson maximum-likelihood optimization did not converge")

    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))
    z_scores = beta / std_err
    p_values = np.array([math.erfc(abs(float(z)) / math.sqrt(2.0)) for z in z_scores])

    terms = {}
    for name, coefficient, error, p_value in zip(
        ("intercept", "log_area", "bio1", "precip_m"), beta, std_err, p_values
    ):
        terms[name] = {
            "coef": float(coefficient),
            "std_err": float(error),
            "p_value": float(p_value),
        }

    positive_y = y > 0
    deviance_terms = mu.copy()  # for y == 0, the y log(y / mu) term is 0
    deviance_terms[positive_y] = (
        y[positive_y] * np.log(y[positive_y] / mu[positive_y])
        - (y[positive_y] - mu[positive_y])
    )
    deviance = max(0.0, 2.0 * float(np.sum(deviance_terms)))
    return {"terms": terms, "deviance": deviance, "n": int(len(df))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
