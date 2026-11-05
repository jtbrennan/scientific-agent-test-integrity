import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_quadratic_trend(df: pd.DataFrame, latitude: float, longitude: float) -> dict:
    """OLS fit of air_temperature = c0 + c1*t + c2*t**2 (t = year - 2000)
    at one grid point.

    Returns a dict with keys: coef, std_err, residual_sd, fitted, n.
    """
    selected = df[(df["latitude"] == latitude) & (df["longitude"] == longitude)]
    n = len(selected)
    if n < 4 or selected["year"].nunique() < 3:
        raise ValueError("at least 4 rows and 3 distinct years are required")

    years = selected["year"].to_numpy()
    t = years - 2000
    x = np.column_stack((np.ones(n), t, t * t)).astype(float)
    y = selected["air_temperature"].to_numpy(dtype=float)

    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    rss = float(residuals @ residuals)
    sigma2 = rss / (n - 3)
    covariance = sigma2 * np.linalg.inv(x.T @ x)
    standard_errors = np.sqrt(np.diag(covariance))

    fitted_values = x @ coefficients
    fitted = {int(year): float(value) for year, value in zip(years, fitted_values)}

    names = ("c0", "c1", "c2")
    return {
        "coef": {name: float(value) for name, value in zip(names, coefficients)},
        "std_err": {name: float(value) for name, value in zip(names, standard_errors)},
        "residual_sd": float(np.sqrt(sigma2)),
        "fitted": fitted,
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "a1b_north_america_annual.csv"
    result = fit_quadratic_trend(pd.read_csv(data), latitude=15.0, longitude=225.0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
