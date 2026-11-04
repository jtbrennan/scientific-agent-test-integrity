import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # select relevant columns and drop NA
    cols = "BA DEGREE COMPLETED", "SES"
    data = df.loc[:, cols].dropna()
    n = data.shape[0]

    # response and design
    y = data["BA DEGREE COMPLETED"].astype(int).values
    ses = data["SES"].astype(float).values
    X = np.column_stack([np.ones(n), ses])

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def neg_loglik(beta):
        eta = X.dot(beta)
        p = sigmoid(eta)
        # avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    # initial guess
    init = np.zeros(X.shape[1])
    res = minimize(neg_loglik, init, method="BFGS")
    beta_hat = res.x

    # compute predicted probabilities and observed information
    eta = X.dot(beta_hat)
    p = sigmoid(eta)
    W = p * (1 - p)
    # Fisher information: X^T W X
    # Build weighted X: multiply each row by sqrt(W)
    Xw = X * np.sqrt(W)[:, None]
    fisher = Xw.T.dot(Xw)
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    ses_coef = float(beta_hat[1])
    ses_se = float(np.sqrt(np.maximum(0.0, cov[1, 1])))
    # two-sided Wald test
    z = ses_coef / ses_se if ses_se > 0 else float("nan")
    p_value = float(2 * norm.sf(abs(z)))
    odds_ratio = float(np.exp(ses_coef))

    return {
        "coef": ses_coef,
        "std_err": ses_se,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "n": int(n),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
