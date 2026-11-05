import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Keep the design matrix explicit: the first column is the intercept and
    # the second is the SES predictor.  No rows are filtered out.
    x = np.asarray(df["SES"], dtype=float)
    y_values = df["BA DEGREE COMPLETED"]
    if pd.api.types.is_bool_dtype(y_values):
        y = y_values.to_numpy(dtype=float)
    else:
        y = pd.to_numeric(y_values, errors="raise").to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(df), dtype=float), x))

    if not np.isfinite(design).all() or not np.isfinite(y).all():
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be boolean or coded as 0/1")
    if len(y) == 0 or np.unique(y).size < 2:
        raise ValueError("The outcome must contain both 0 and 1 observations")

    def objective(beta: np.ndarray) -> float:
        # log(1 + exp(linear predictor)) is evaluated stably by logaddexp.
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        residual = expit(design @ beta) - y
        return design.T @ residual

    # The intercept-only estimate is a useful finite starting point.  BFGS
    # uses the analytic score, while the observed information below supplies
    # the model-based covariance matrix at the MLE.
    mean_y = float(np.mean(y))
    start = np.array([np.log(mean_y / (1.0 - mean_y)), 0.0])
    result = minimize(objective, start, jac=gradient, method="BFGS",
                      options={"gtol": 1e-10, "maxiter": 1000})
    if not result.success and np.linalg.norm(gradient(result.x), ord=np.inf) > 1e-7:
        raise RuntimeError(f"Logistic regression failed to converge: {result.message}")

    beta = result.x
    probabilities = expit(design @ beta)
    information = design.T @ (probabilities[:, None] * (1.0 - probabilities)[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The logistic-regression information matrix is singular") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coef / std_err
    # scipy.special.ndtr is the standard normal CDF, avoiding a dependency on
    # statsmodels for the two-sided Wald p-value.
    from scipy.special import ndtr
    p_value = float(2.0 * ndtr(-abs(z_score)))
    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coef)),
        "n": int(len(y)),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
