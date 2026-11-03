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
    # Keep the construction of the response explicit: pandas booleans convert
    # naturally to 0/1, and numeric 0/1 responses are supported as well.
    ses = df["SES"].to_numpy(dtype=float)
    response = df["BA DEGREE COMPLETED"].to_numpy()
    try:
        y = response.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("BA DEGREE COMPLETED must contain boolean or 0/1 values") from exc

    if len(ses) == 0:
        raise ValueError("at least one row is required")
    if not np.isfinite(ses).all() or not np.isfinite(y).all():
        raise ValueError("SES and BA DEGREE COMPLETED must not contain missing values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must contain only boolean or 0/1 values")

    # Columns are intercept then SES, so beta[1] is the requested effect.
    design = np.column_stack((np.ones(len(ses)), ses))

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        # logaddexp is stable even when an optimizer tries a large coefficient.
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        weights = probabilities * (1.0 - probabilities)
        return design.T @ (weights[:, None] * design)

    # Newton--Raphson / IRLS is especially suitable here: this is a concave
    # two-parameter likelihood and the analytic Hessian is its Fisher
    # information.  A short backtracking line search keeps every iteration a
    # likelihood improvement, including for unusually scaled SES values.
    beta = np.zeros(2)
    for _ in range(100):
        score = gradient(beta)
        if np.max(np.abs(score)) < 1e-10:
            break
        information = hessian(beta)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("logistic-regression parameters are not identifiable") from exc

        old_objective = objective(beta)
        scale = 1.0
        while scale > 1e-12:
            candidate = beta - scale * step
            if objective(candidate) < old_objective:
                beta = candidate
                break
            scale /= 2.0
        else:
            raise RuntimeError("logistic-regression fit failed to improve")
    else:
        raise RuntimeError("logistic-regression fit did not converge")

    information = hessian(beta)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("logistic-regression parameters are not identifiable") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coef / std_err
    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "odds_ratio": float(np.exp(coef)),
        "n": int(len(df)),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
