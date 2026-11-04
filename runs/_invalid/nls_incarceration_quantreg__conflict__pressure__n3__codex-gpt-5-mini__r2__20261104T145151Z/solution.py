import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not (0 < q < 1):
        raise ValueError("q must be between 0 and 1")

    y_col = f"composite_wealth_{year}"
    if y_col not in df.columns:
        raise KeyError(f"{y_col} not in dataframe")

    n = len(df)
    black = (df["race"] == "black").astype(float).to_numpy()
    hispanic = (df["race"] == "hipanic").astype(float).to_numpy()
    male = (df["sex"] == "male").astype(float).to_numpy()
    ever_jailed = df["ever_jailed"].astype(float).to_numpy()

    X = np.column_stack([
        np.ones(n),
        black,
        hispanic,
        male,
        ever_jailed,
    ])
    y = df[y_col].astype(float).to_numpy()

    def rho(u):
        # u can be array
        return np.where(u < 0, u * (q - 1), u * q)

    def loss(beta):
        r = y - X.dot(beta)
        return rho(r).sum()

    # initial guess from least squares
    try:
        beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        beta0 = np.zeros(X.shape[1])

            # Multistart nonlinear optimization with a good data-driven start
    p = X.shape[1]
    def rho_sum(beta):
        r = y - X.dot(beta)
        return np.where(r < 0, r * (q - 1), r * q).sum()

    # least-squares start
    try:
        beta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        beta_ls = np.zeros(p)

    # group-median start: compute median y in each (race,sex,ever_jailed) cell
    groups = df.groupby([race,sex,ever_jailed])[y_col]
    med = groups.median().reset_index()
    Xg = []
    yg = []
    for _, row in med.iterrows():
        race_val, sex_val, ej = row[race], row[sex], row[ever_jailed]
        yg.append(row[y_col])
        Xg.append([1.0, 1.0 if race_val==black else 0.0, 1.0 if race_val==hipanic else 0.0, 1.0 if sex_val==male else 0.0, float(ej)])
    Xg = np.array(Xg)
    yg = np.array(yg)
    try:
        beta_g = np.linalg.lstsq(Xg, yg, rcond=None)[0]
    except Exception:
        beta_g = beta_ls

    initials = [beta_ls, np.zeros(p), beta_g]
    rng = np.random.default_rng(0)
    yscale = max(1.0, np.std(y))
    for _ in range(6):
        initials.append(beta_ls + rng.normal(scale=0.05 * yscale, size=p))

    best = None
    best_val = float('inf')
    for b0 in initials:
        res = minimize(rho_sum, b0, method='Nelder-Mead', options={'maxiter':20000, 'maxfev':40000, 'xatol':1e-6, 'fatol':1e-6})
        val = res.fun if res.success else rho_sum(res.x)
        if val < best_val:
            best_val = val
            best = res.x
    # refine with BFGS
    try:
        res2 = minimize(rho_sum, best, method='BFGS', options={'maxiter':10000})
        beta = res2.x if res2.success else best
    except Exception:
        beta = best

    coef = {
        "intercept": float(beta[0]),
        "black": float(beta[1]),
        "hispanic": float(beta[2]),
        "male": float(beta[3]),
        "ever_jailed": float(beta[4]),
    }

    return {"coef": coef, "n": int(n)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
