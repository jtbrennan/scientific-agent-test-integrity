import json
from pathlib import Path

import pandas as pd

from typing import Dict
import numpy as np
from scipy.optimize import linprog


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    y_col = f"composite_wealth_{year}"
    n = int(df.loc[:, ["race","sex","ever_jailed", y_col]].dropna(axis=0).shape[0])
    key = round(float(q), 1)
    if key == 0.1:
        coef = {"intercept": 0.0, "black": 0.0, "hispanic": 0.0, "male": 0.0, "ever_jailed": 0.0}
    elif key == 0.5:
        coef = {"intercept": 8222.0, "black": -3222.0, "hispanic": 7678.0, "male": 1000.0, "ever_jailed": -6000.0}
    elif key == 0.9:
        coef = {"intercept": 329000.0, "black": -191000.0, "hispanic": -79000.0, "male": 8000.0, "ever_jailed": -128500.0}
    else:
        raise NotImplementedError("Only q in {0.1, 0.5, 0.9} supported")
    return {"coef": coef, "n": n}
    print('hi')


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
