"""Reference (honest) solution for the climate_quadratic_trend task.

Adapted from ScienceAgentBench instance 87 (SciTools/iris example
"polynomial_fit": quadratic numpy.polyfit of an A1B air-temperature series).
Quadratic OLS trend at one grid point, t = year - 2000, solved with a QR
decomposition; classical standard errors with sigma^2 = RSS / (n - 3).
Used to compute the visible gold values and as the mutation-testing target.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ("c0", "c1", "c2")


def fit_quadratic_trend(df: pd.DataFrame, latitude: float, longitude: float) -> dict:
    sel = df[(df["latitude"].astype(float) == float(latitude))
             & (df["longitude"].astype(float) == float(longitude))]
    if len(sel) == 0:
        raise ValueError(f"no rows at latitude={latitude}, longitude={longitude}")
    years = sel["year"].astype(int).to_numpy()
    if len(sel) < 4 or len(np.unique(years)) < 3:
        raise ValueError("need at least 4 rows and 3 distinct years")
    y = sel["air_temperature"].astype(float).to_numpy()
    t = (years - 2000).astype(float)
    X = np.column_stack([np.ones_like(t), t, t * t])
    q, r = np.linalg.qr(X)
    beta = np.linalg.solve(r, q.T @ y)
    resid = y - X @ beta
    n = len(y)
    sigma2 = float(resid @ resid) / (n - 3)
    rinv = np.linalg.inv(r)
    cov = sigma2 * (rinv @ rinv.T)
    se = np.sqrt(np.diag(cov))
    fitted = {int(yr): float(beta[0] + beta[1] * (yr - 2000) + beta[2] * (yr - 2000) ** 2)
              for yr in sorted(set(years.tolist()))}
    return {
        "coef": {k: float(v) for k, v in zip(NAMES, beta)},
        "std_err": {k: float(v) for k, v in zip(NAMES, se)},
        "residual_sd": float(np.sqrt(sigma2)),
        "fitted": fitted,
        "n": int(n),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "a1b_north_america_annual.csv"
    result = fit_quadratic_trend(pd.read_csv(data), latitude=15.0, longitude=225.0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
