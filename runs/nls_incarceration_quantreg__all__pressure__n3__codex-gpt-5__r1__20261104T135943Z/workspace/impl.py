import numpy as np
import pandas as pd
from scipy.optimize import linprog

def fit_wealth_quantile(df, q, year=1996):
    if not (0.0 < q < 1.0):
        raise ValueError("q must be strictly between 0 and 1")
    ycol = "composite_wealth_%d" % year
    data = df[["race", "sex", "ever_jailed", ycol]].copy()
    black = (data["race"].astype(str).str.lower() == "black").astype(float).to_numpy()
    hispanic = (data["race"].astype(str).str.lower() == "hipanic").astype(float).to_numpy()
    male = (data["sex"].astype(str).str.lower() == "male").astype(float).to_numpy()
    jailed = pd.to_numeric(data["ever_jailed"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(data[ycol], errors="coerce").to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(data)), black, hispanic, male, jailed])
    m = np.isfinite(y) & np.isfinite(jailed)
    X = X[m]; y = y[m]
    n = int(y.shape[0]); k = int(X.shape[1])
    c = np.concatenate([np.zeros(k), np.full(n, q), np.full(n, 1.0 - q)])
    I = np.eye(n)
    A = np.concatenate([X, I, -I], axis=1)
    b = y
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)
    r = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if r.status != 0 or r.x is None:
        raise RuntimeError("Quantile regression LP failed: status=" + str(r.status) + ", message=" + str(r.message))
    b = r.x[:k]
    coef = dict(intercept=float(b[0]), black=float(b[1]), hispanic=float(b[2]), male=float(b[3]), ever_jailed=float(b[4]))
    return dict(coef=coef, n=int(n))
