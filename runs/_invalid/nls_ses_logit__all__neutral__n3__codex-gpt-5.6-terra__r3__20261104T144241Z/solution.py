import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # The outcome happens to be read as bool from the supplied CSV.  Converting
    # explicitly also makes the calculation work when a caller supplies 0/1.
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    ses = df["SES"].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        raise ValueError("cannot fit a model with no observations")
    if not (np.isfinite(y).all() and np.isfinite(ses).all()):
        raise ValueError("BA DEGREE COMPLETED and SES must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    x = np.column_stack((np.ones(n), ses))

    # Optimise the negative log likelihood.  logaddexp keeps this stable even
    # for trial coefficients that imply very large or very small probabilities.
    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return x.T @ (expit(x @ beta) - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        p = expit(x @ beta)
        return x.T @ ((p * (1.0 - p))[:, None] * x)

    mean_y = y.mean()
    if mean_y in (0.0, 1.0):
        raise ValueError("a finite logistic-regression estimate requires both outcomes")

    beta = np.array([np.log(mean_y / (1.0 - mean_y)), 0.0])
    # Newton--Raphson is maximum-likelihood fitting here: for logistic
    # regression its curvature is available exactly.  Backtracking retains a
    # decreasing negative log likelihood in less well-scaled caller data.
    for _ in range(100):
        step = np.linalg.solve(hessian(beta), gradient(beta))
        old_objective = objective(beta)
        scale = 1.0
        while objective(beta - scale * step) > old_objective:
            scale /= 2.0
            if scale < 2.0**-40:
                raise RuntimeError("logistic regression failed to make progress")
        beta -= scale * step
        if np.max(np.abs(scale * step)) < 1e-12:
            break
    else:
        raise RuntimeError("logistic regression did not converge")
    p = expit(x @ beta)
    # The inverse observed information is the usual large-sample covariance
    # estimate for maximum-likelihood logistic regression coefficients.
    information = x.T @ ((p * (1.0 - p))[:, None] * x)
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
