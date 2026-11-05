import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Keep the intercept explicit: the coefficient of interest is the second
    # element of beta, while the first element is the intercept.
    x = np.asarray(df["SES"], dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(df)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be binary")

    design = np.column_stack((np.ones(n), x))

    def score(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        return design.T @ (probabilities - y)

    def score_jacobian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        weights = probabilities * (1.0 - probabilities)
        return design.T @ (weights[:, None] * design)

    # The score is zero at the MLE.  Supplying its Jacobian (the observed
    # information matrix) makes this two-parameter solve both fast and stable.
    fit = root(score, x0=np.zeros(2), jac=score_jacobian, method="hybr", tol=1e-11)
    if not fit.success:
        raise RuntimeError(f"Logistic regression failed to converge: {fit.message}")

    beta = fit.x
    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Logistic regression information matrix is singular") from exc

    coefficient = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coefficient / std_err
    p_value = float(2.0 * norm.sf(abs(z_score)))

    return {
        "coef": coefficient,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coefficient)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
