import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    ses = np.asarray(df["SES"], dtype=float)
    outcome = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(df)

    if not np.isfinite(ses).all() or not np.isfinite(outcome).all():
        raise ValueError("SES and BA DEGREE COMPLETED must not contain missing values")
    if not np.isin(outcome, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    design = np.column_stack((np.ones(n), ses))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        # log_expit keeps the likelihood finite and accurate for large |eta|.
        negative_log_likelihood = -np.sum(
            outcome * log_expit(eta) + (1.0 - outcome) * log_expit(-eta)
        )
        gradient = design.T @ (expit(eta) - outcome)
        return float(negative_log_likelihood), gradient

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(2, dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    # BFGS can report precision loss after reaching the solution when the
    # likelihood is evaluated to machine precision.  The gradient is the
    # relevant convergence check for this two-parameter problem.
    if not fit.success and np.max(np.abs(fit.jac)) > 1e-7:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")

    beta = fit.x
    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(information)

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coef / std_err
    p_value = float(2.0 * norm.sf(abs(z_score)))

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
