import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    required = ["SAMPLE_RACE", "BA DEGREE COMPLETED", "SES"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    # The input is specified to contain the rows used by the model.  Fail
    # explicitly rather than silently changing the requested sample if a
    # malformed frame is supplied.
    if df[required].isna().any().any():
        raise ValueError("Model columns must not contain missing values")

    race = df["SAMPLE_RACE"]
    unknown_races = set(race.unique()) - {"White", "Black", "Hispanic"}
    if unknown_races:
        raise ValueError(f"Unexpected race value(s): {sorted(unknown_races)!r}")

    ses = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("BA DEGREE COMPLETED must be coded as 0/1 or Boolean")

    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)
    # Parameter order: intercept, SES, Black, Hispanic, SES*Black,
    # SES*Hispanic.
    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        # log(1 + exp(eta)) - y*eta is a stable negative log likelihood.
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        return x.T @ (probabilities - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(x @ beta)
        weights = probabilities * (1.0 - probabilities)
        return (x * weights[:, None]).T @ x

    # Damped Newton maximization.  The likelihood is concave, and the line
    # search keeps Newton steps stable even when starting far from the fit.
    beta = np.zeros(x.shape[1], dtype=float)
    converged = False
    for _ in range(1_000):
        score = gradient(beta)
        if np.max(np.abs(score)) < 1e-8:
            converged = True
            break
        try:
            step = np.linalg.solve(hessian(beta), score)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Observed information matrix is singular") from exc

        current_objective = objective(beta)
        directional_derivative = -float(score @ step)
        step_scale = 1.0
        while objective(beta - step_scale * step) > (
            current_objective + 1e-4 * step_scale * directional_derivative
        ):
            step_scale *= 0.5
            if step_scale < 1e-12:
                raise RuntimeError("Logistic regression line search failed")
        beta -= step_scale * step
    if not converged:
        raise RuntimeError("Logistic regression failed to converge")

    information = hessian(beta)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Observed information matrix is singular") from exc
    standard_errors = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / standard_errors))

    names = ("ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic")
    # The intercept is fitted as required but is not part of the requested
    # result schema.
    output = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    }
    output["n"] = int(len(df))
    return output


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
