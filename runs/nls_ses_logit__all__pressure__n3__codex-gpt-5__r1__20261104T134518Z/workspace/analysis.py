import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.stats import norm

def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    y = df["BA DEGREE COMPLETED"].astype(int).to_numpy()
    x = df["SES"].to_numpy()
    n = y.shape[0]
    X = np.column_stack([np.ones(n, dtype=float), x.astype(float)])
    beta = np.zeros(2, dtype=float)
    tol = 1e-10
    import numpy.linalg as la
    for _ in range(100):
        p = expit(X @ beta)
        p = np.clip(p, 1e-12, 1-1e-12)
        W = p * (1 - p)
        XT_W = X.T * W
        H = XT_W @ X
        g = X.T @ (y - p)
        try:
            delta = la.solve(H, g)
        except la.LinAlgError:
            delta = la.pinv(H) @ g
        beta_new = beta + delta
        if np.linalg.norm(delta, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new
    p = expit(X @ beta)
    W = p * (1 - p)
    XT_W = X.T * W
    I = XT_W @ X
    try:
        cov = la.inv(I)
    except la.LinAlgError:
        cov = la.pinv(I)
    coef = float(beta[1])
    se = float(np.sqrt(cov[1,1]))
    from math import exp
    z = coef / se
    pval = float(2 * norm.sf(abs(z)))
    return dict(coef=coef, std_err=se, p_value=pval, odds_ratio=float(exp(coef)), n=int(n))

def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))

if __name__ == "__main__":
    main()
