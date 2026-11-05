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
    # Do not silently discard rows: the requested sample size is the number of
    # rows supplied to this function.
    n = len(df)
    if n == 0:
        raise ValueError("cannot fit a model with no rows")

    x = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    outcome = df["BA DEGREE COMPLETED"]
    if outcome.isna().any() or not np.isfinite(x).all():
        raise ValueError("SES and BA DEGREE COMPLETED must not contain missing values")

    # pandas reads the supplied CSV as bool.  Also accept the conventional
    # numeric 0/1 representation, which is useful when calling this function
    # on a manually constructed DataFrame.
    if pd.api.types.is_bool_dtype(outcome):
        y = outcome.to_numpy(dtype=float)
    else:
        y = pd.to_numeric(outcome, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must contain only 0/1 or boolean values")
    if y.min() == y.max():
        raise ValueError("the outcome must contain both classes")

    design = np.column_stack((np.ones(n), x))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        # log(1 + exp(eta)) - y*eta is stable even for large eta.
        value = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        probabilities = expit(eta)
        gradient = design.T @ (probabilities - y)
        return value, gradient

    # Starting at the empirical logit gives the optimizer a sensible point for
    # rare outcomes while retaining an unregularized maximum-likelihood fit.
    proportion = float(y.mean())
    initial = np.array([np.log(proportion / (1.0 - proportion)), 0.0])
    result = minimize(
        lambda beta: objective(beta),
        initial,
        jac=True,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"logistic regression failed to converge: {result.message}")

    beta = np.asarray(result.x, dtype=float)
    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("the model information matrix is singular") from exc

    coefficient = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_statistic = coefficient / std_err
    p_value = float(2.0 * norm.sf(abs(z_statistic)))
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
