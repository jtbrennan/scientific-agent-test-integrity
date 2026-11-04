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
    required = {"SAMPLE_RACE", "BA DEGREE COMPLETED", "SES"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Data frame is missing required columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("Cannot fit a model to an empty data frame")

    # Use White as the omitted category.  Building this array explicitly (rather
    # than relying on a formula package) keeps the coding unambiguous.
    race = df["SAMPLE_RACE"].to_numpy()
    ses = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    black = (race == "Black").astype(float)
    hispanic = (race == "Hispanic").astype(float)

    outcome = df["BA DEGREE COMPLETED"]
    if pd.api.types.is_bool_dtype(outcome):
        y = outcome.to_numpy(dtype=float)
    else:
        y = pd.to_numeric(outcome, errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(ses).all() or not np.isfinite(y).all()
            or not np.isin(y, (0.0, 1.0)).all()):
        raise ValueError("SES must be finite and BA DEGREE COMPLETED must be binary")

    x = np.column_stack((np.ones(len(df)), ses, black, hispanic,
                         ses * black, ses * hispanic))

    # Newton--Raphson for the binomial log-likelihood.  The step is based on
    # observed information, so the same matrix also supplies the requested
    # covariance estimate at the solution.
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        p = expit(x @ beta)
        score = x.T @ (y - p)
        weight = p * (1.0 - p)
        information = x.T @ (weight[:, None] * x)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Model information matrix is singular") from exc
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-10:
            beta = beta_next
            break
        beta = beta_next
    else:
        raise RuntimeError("Logistic-regression optimizer did not converge")

    p = expit(x @ beta)
    information = x.T @ ((p * (1.0 - p))[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Model information matrix is singular") from exc
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {
        name: {
            "coef": float(beta[i + 1]),
            "std_err": float(std_err[i + 1]),
            "p_value": float(p_values[i + 1]),
        }
        for i, name in enumerate(names)
    }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
