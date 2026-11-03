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
    required = {"BA DEGREE COMPLETED", "SES"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    # Do not silently omit observations: this analysis is defined to use every
    # row supplied by the caller.
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    ses = df["SES"].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        raise ValueError("Cannot fit a logistic regression to an empty DataFrame")
    if not (np.isfinite(y).all() and np.isfinite(ses).all()):
        raise ValueError("BA DEGREE COMPLETED and SES must contain only finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary (True/False or 0/1)")

    design = np.column_stack((np.ones(n), ses))

    def objective(beta: np.ndarray) -> float:
        linear_predictor = design @ beta
        # logaddexp evaluates log(1 + exp(x)) stably for large |x|.
        return float(np.sum(np.logaddexp(0.0, linear_predictor) - y * linear_predictor))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        weights = probabilities * (1.0 - probabilities)
        return design.T @ (weights[:, None] * design)

    # A finite intercept-only start avoids logit(0) or logit(1) at the start.
    outcome_rate = y.mean()
    clipped_rate = np.clip(outcome_rate, 1e-8, 1.0 - 1e-8)
    start = np.array([np.log(clipped_rate / (1.0 - clipped_rate)), 0.0])
    result = minimize(
        objective,
        start,
        jac=gradient,
        hess=hessian,
        method="Newton-CG",
        options={"xtol": 1e-10, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"Logistic-regression MLE did not converge: {result.message}")

    information = hessian(result.x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("The model parameters are not identifiable") from error

    coef = float(result.x[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_statistic = coef / std_err
    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z_statistic))),
        "odds_ratio": float(np.exp(coef)),
        "n": int(n),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
