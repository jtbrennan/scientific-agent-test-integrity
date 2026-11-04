import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Select and clean data
    y_col = "BA DEGREE COMPLETED"
    x_col = "SES"

    df = df[[y_col, x_col]].dropna()

    # Response as float in {0,1}
    y = df[y_col].astype(float).to_numpy()
    x = df[x_col].astype(float).to_numpy()
    n = y.shape[0]

    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x])

    # IRLS / Newton-Raphson for logistic regression
    beta = np.zeros(2, dtype=float)
    max_iter = 100
    tol = 1e-10

    for _ in range(max_iter):
        eta = X @ beta
        # Stable sigmoid; clip avoids probabilities at 0 or 1
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
        W = p * (1.0 - p)

        # Gradient and Hessian
        z = y - p
        XTz = X.T @ z
        H = X.T @ (X * W[:, None])  # X^T W X

        # Solve for Newton step: H * delta = grad
        try:
            delta = np.linalg.solve(H, XTz)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ XTz

        beta_new = beta + delta
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final probabilities and information matrix at MLE
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
    W = p * (1.0 - p)
    H = X.T @ (X * W[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    ses = np.sqrt(np.diag(cov))

    coef = float(beta[1])
    std_err = float(ses[1])
    z = coef / std_err
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    odds_ratio = float(np.exp(coef))

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "n": int(n),
    }



def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
PY
