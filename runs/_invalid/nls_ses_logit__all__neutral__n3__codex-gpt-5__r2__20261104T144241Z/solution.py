import json
from pathlib import Path

import pandas as pd
import numpy as np
from math import exp as _exp, erfc, sqrt


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Prepare data
    y = df["BA DEGREE COMPLETED"].astype(int).to_numpy()
    x = df["SES"].astype(float).to_numpy()

    n = y.shape[0]
    X = np.column_stack([np.ones(n, dtype=float), x])

    # IRLS / Newton-Raphson for logistic regression
    beta = np.zeros(2, dtype=float)
    max_iter = 100
    tol = 1e-10

    for _ in range(max_iter):
        eta = X @ beta
        # numerically stable sigmoid
        p = 1.0 / (1.0 + np.exp(-eta))
        # weights and gradient
        w = p * (1.0 - p)
        # Guard against zero weights (extreme probabilities)
        w = np.clip(w, 1e-12, None)
        # Hessian: X' W X ; Gradient: X' (y - p)
        WX = X * w[:, None]
        H = WX.T @ X
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if H is singular (unlikely here)
            step = np.linalg.pinv(H) @ g
        beta_new = beta + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final weights for variance-covariance (observed information)
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-12, None)
    WX = X * w[:, None]
    H = WX.T @ X
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    coef = float(beta[1])
    std_err = float(sqrt(cov[1, 1]))
    z = abs(coef) / std_err if std_err > 0 else np.inf
    # Two-sided Wald p-value from standard normal
    p_value = float(erfc(z / sqrt(2.0)))

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(_exp(coef)),
        "n": int(n),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
