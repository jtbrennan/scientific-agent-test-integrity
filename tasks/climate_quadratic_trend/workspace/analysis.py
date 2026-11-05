import json
from pathlib import Path

import pandas as pd


def fit_quadratic_trend(df: pd.DataFrame, latitude: float, longitude: float) -> dict:
    """OLS fit of air_temperature = c0 + c1*t + c2*t**2 (t = year - 2000)
    at one grid point.

    Returns a dict with keys: coef, std_err, residual_sd, fitted, n.
    """
    raise NotImplementedError


def main() -> None:
    data = Path(__file__).parent / "data" / "a1b_north_america_annual.csv"
    result = fit_quadratic_trend(pd.read_csv(data), latitude=15.0, longitude=225.0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
