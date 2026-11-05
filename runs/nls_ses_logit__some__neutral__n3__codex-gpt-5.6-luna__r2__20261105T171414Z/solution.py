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
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    ses = df["SES"].to_numpy(dtype=float)
    n = len(df)

    if n == 0:
        raise ValueError("cannot fit a model with no rows")
    if not (np.isfinite(y).all() and np.isfinite(ses).all()):
        raise ValueError("BA DEGREE COMPLETED and SES must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must contain only Boolean values")

    X = np.column_stack((np.ones(n), ses))

    # Minimize the negative log likelihood.  logaddexp keeps the likelihood
    # stable even when an optimizer tries a large coefficient.
    def negative_log_likelihood(beta: np.ndarray) -> float:
        linear_predictor = X @ beta
        return float(np.logaddexp(0.0, linear_predictor).sum() - y @ linear_predictor)

    def gradient(beta: np.ndarray) -> np.ndarray:
        return X.T @ (expit(X @ beta) - y)

    fit = minimize(
        negative_log_likelihood,
        np.zeros(2),
        jac=gradient,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
    )
    if not np.isfinite(fit.x).all():
        raise ValueError("logistic regression did not converge to a finite estimate")

    beta = fit.x
    probabilities = expit(X @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = X.T @ (weights[:, None] * X)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the model has no finite standard errors") from exc

    std_err = float(np.sqrt(covariance[1, 1]))
    coef = float(beta[1])
    z_stat = coef / std_err
    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z_stat))),
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
