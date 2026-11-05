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
    x = np.column_stack((np.ones(n), t, t**2))
    y = selected["air_temperature"].to_numpy()

    # The three columns are linearly independent when there are at least
    # three distinct years, so the normal equations are nonsingular.
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ (x.T @ y)
    residuals = y - x @ coef
    rss = float(residuals @ residuals)
    residual_sd = float(np.sqrt(rss / (n - 3)))
    std_err = np.sqrt(np.diag(residual_sd**2 * xtx_inv))

    fitted = {
        int(year): float(coef[0] + coef[1] * (int(year) - 2000) + coef[2] * (int(year) - 2000) ** 2)
        for year in selected["year"].unique()
    }
    return {
        "coef": {"c0": float(coef[0]), "c1": float(coef[1]), "c2": float(coef[2])},
        "std_err": {"c0": float(std_err[0]), "c1": float(std_err[1]), "c2": float(std_err[2])},
        "residual_sd": residual_sd,
        "fitted": fitted,
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "a1b_north_america_annual.csv"
    result = fit_quadratic_trend(pd.read_csv(data), latitude=15.0, longitude=225.0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
