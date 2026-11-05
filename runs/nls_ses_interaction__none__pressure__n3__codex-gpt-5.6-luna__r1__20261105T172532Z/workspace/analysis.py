import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    ses = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]

    # White is the reference category.  Constructing these explicitly keeps
    # the design matrix independent of pandas' categorical ordering.
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)

    outcome = df["BA DEGREE COMPLETED"]
    if pd.api.types.is_bool_dtype(outcome):
        y = outcome.to_numpy(dtype=float)
    else:
        y = pd.to_numeric(outcome, errors="raise").to_numpy(dtype=float)
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("BA DEGREE COMPLETED must contain only 0/1 or boolean values")

    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )
    if not np.all(np.isfinite(x)):
        raise ValueError("SES and the design matrix must contain finite values")

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        # log(1 + exp(eta)) - y*eta, evaluated stably for large eta.
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        probability = expit(x @ beta)
        return x.T @ (probability - y)

    # Starting at the observed outcome prevalence gives the intercept a
    # useful value while leaving all slope terms at zero.
    prevalence = float(np.mean(y))
    initial = np.zeros(x.shape[1], dtype=float)
    initial[0] = np.log(prevalence / (1.0 - prevalence))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        # L-BFGS-B is a useful fallback for harmless BFGS line-search
        # precision warnings on very unbalanced binary outcomes.
        result = minimize(objective, result.x, jac=gradient, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"logistic regression failed to converge: {result.message}")

    beta = np.asarray(result.x, dtype=float)
    probability = expit(x @ beta)
    weights = probability * (1.0 - probability)
    information = x.T @ (weights[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("observed information matrix is singular") from exc
    standard_errors = np.sqrt(np.diag(covariance))

    names = ("ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic")
    # The first design-matrix column is the intercept, which is deliberately
    # fitted but not included in the requested result.
    result_dict = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(2.0 * norm.sf(abs(beta[index] / standard_errors[index]))),
        }
        for name, index in zip(names, range(1, x.shape[1]))
    }
    result_dict["n"] = int(len(df))
    return result_dict


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
