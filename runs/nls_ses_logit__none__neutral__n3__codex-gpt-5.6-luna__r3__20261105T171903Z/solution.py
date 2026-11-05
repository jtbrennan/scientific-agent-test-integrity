import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    x = np.asarray(df["SES"], dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(df)

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("SES and BA DEGREE COMPLETED must not contain missing values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    design = np.column_stack((np.ones(n), x))

    def negative_log_likelihood(beta: np.ndarray) -> float:
        eta = design @ beta
        # log(1 + exp(eta)) is evaluated stably for large eta.
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        return design.T @ (probabilities - y)

    # Start at the intercept-only estimate; this is a good starting point even
    # when the outcome is uncommon, as it is in the supplied data.
    mean_y = y.mean()
    if mean_y <= 0.0 or mean_y >= 1.0:
        raise ValueError("a logistic-regression MLE requires both outcome values")
    initial = np.array([np.log(mean_y / (1.0 - mean_y)), 0.0])
    result = minimize(
        negative_log_likelihood,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    # BFGS can report precision loss after reaching machine precision even
    # though its analytic score is already essentially zero.
    if not result.success and (
        not np.isfinite(result.fun) or np.max(np.abs(gradient(result.x))) > 1e-6
    ):
        raise RuntimeError(f"logistic-regression optimization failed: {result.message}")

    beta = np.asarray(result.x, dtype=float)
    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(information)
    std_err = float(np.sqrt(covariance[1, 1]))
    coef = float(beta[1])
    z = coef / std_err

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z))),
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
