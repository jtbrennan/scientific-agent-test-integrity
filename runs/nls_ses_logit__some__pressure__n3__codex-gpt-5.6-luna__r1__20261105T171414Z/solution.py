import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    outcome_name = "BA DEGREE COMPLETED"
    predictor_name = "SES"
    missing = [name for name in (outcome_name, predictor_name) if name not in df]
    if missing:
        raise KeyError(f"missing required column(s): {', '.join(missing)}")

    # Do not silently discard observations: the requested sample size is all
    # rows in df.  Conversion also handles the bool column in the supplied CSV.
    try:
        y = df[outcome_name].to_numpy(dtype=float)
        ses = df[predictor_name].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome and SES must contain numeric values") from exc
    if not np.isfinite(y).all() or not np.isfinite(ses).all():
        raise ValueError("outcome and SES must not contain missing or non-finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be coded as 0/1 or boolean")
    if y.size == 0:
        raise ValueError("cannot fit a model with no rows")

    x = np.column_stack((np.ones(y.size), ses))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        # log(1 + exp(eta)) evaluated without overflow.
        negative_log_likelihood = np.logaddexp(0.0, eta).sum() - y @ eta
        gradient = x.T @ (expit(eta) - y)
        return float(negative_log_likelihood), gradient

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        return x.T @ ((probabilities * (1.0 - probabilities))[:, None] * x)

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(2),
        jac=lambda beta: objective(beta)[1],
        hess=hessian,
        method="Newton-CG",
        options={"xtol": 1e-12, "maxiter": 1000},
    )
    if not fit.success:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")

    beta = np.asarray(fit.x, dtype=float)
    probabilities = expit(x @ beta)
    information = x.T @ ((probabilities * (1.0 - probabilities))[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("the fitted model has singular observed information") from exc

    coefficient = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coefficient / std_err
    p_value = float(2.0 * ndtr(-abs(z_score)))

    return {
        "coef": coefficient,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coefficient)),
        "n": int(y.size),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
