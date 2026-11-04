import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Keep only the needed columns and drop rows with missing values
    cols = [
        "BA DEGREE COMPLETED",
        "SES",
        "SAMPLE_RACE",
    ]
    df = df.loc[:, cols].dropna()

    # Dependent variable y in {0,1}
    y_raw = df["BA DEGREE COMPLETED"]
    if y_raw.dtype == bool:
        y = y_raw.astype(float).values
    else:
        y = (
            y_raw.replace({True: 1, False: 0, "True": 1, "False": 0})
            .astype(float)
            .values
        )

    # Race indicators (White as reference)
    race = df["SAMPLE_RACE"].astype(str)
    black = (race == "Black").astype(float).values
    hispanic = (race == "Hispanic").astype(float).values

    # SES and interactions
    ses = pd.to_numeric(df["SES"], errors="coerce").values.astype(float)
    ses_x_black = ses * black
    ses_x_hispanic = ses * hispanic

    # Design matrix with intercept
    X = np.column_stack(
        [
            np.ones(len(df)),
            ses,
            black,
            hispanic,
            ses_x_black,
            ses_x_hispanic,
        ]
    )

    # Negative log-likelihood and gradient for logistic regression
    def nll(beta: np.ndarray) -> float:
        xb = X @ beta
        p = expit(xb)
        # Stable log-likelihood computation
        # Using: y*log(p) + (1-y)*log(1-p)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return -ll

    def grad(beta: np.ndarray) -> np.ndarray:
        p = expit(X @ beta)
        return X.T @ (p - y)

    # Initialize at zeros (log-odds centered)
    beta0 = np.zeros(X.shape[1], dtype=float)

    opt = minimize(nll, beta0, jac=grad, method="BFGS")
    beta_hat = opt.x

    # Observed information (Hessian) at the optimum: X^T W X, W = diag(p*(1-p))
    p_hat = expit(X @ beta_hat)
    W = p_hat * (1 - p_hat)
    # In case of quasi-complete separation, W may have zeros; add tiny ridge for stability
    XtW = X.T * W
    hess = XtW @ X
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    se = np.sqrt(np.diag(cov))

    # Two-sided Wald p-values
    z = beta_hat / np.where(se > 0, se, np.inf)
    p_values = 2 * norm.sf(np.abs(z))

    # Map coefficients to requested keys (excluding intercept)
    idx = {
        "ses": 1,
        "black": 2,
        "hispanic": 3,
        "ses_x_black": 4,
        "ses_x_hispanic": 5,
    }
    out = {k: {"coef": float(beta_hat[i]), "std_err": float(se[i]), "p_value": float(p_values[i])} for k, i in idx.items()}
    out["n"] = int(X.shape[0])
    return out


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
