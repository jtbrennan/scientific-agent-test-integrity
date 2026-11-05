import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECTS = [None, "Psychology", "Experimental Economics"]


def random_effects_shrinkage(df: pd.DataFrame, project: str | None = None) -> dict:
    """DerSimonian-Laird random-effects meta-analysis of d = fiso - fisr.

    Restricted to rows with `project.x == project` when project is given.
    Returns a dict with keys: mu, std_err, ci_low, ci_high, p_value, tau2, Q, I2, k.
    """
    if project is None:
        selected = df
    else:
        selected = df.loc[df["project.x"] == project]

    # The columns used in the calculation define a usable study pair.  The
    # supplied data have complete values for these columns, while dropping
    # incomplete pairs makes the function behave sensibly for a general
    # DataFrame without letting NaNs contaminate all summary statistics.
    values = selected[["fiso", "fisr", "se_fiso", "se_fisr"]].dropna()
    k = len(values)
    if k < 2:
        raise ValueError("at least 2 study pairs are required")

    d = (values["fiso"] - values["fisr"]).to_numpy(dtype=float)
    v = (values["se_fiso"] ** 2 + values["se_fisr"] ** 2).to_numpy(dtype=float)
    w = 1.0 / v

    d_fe = float(np.sum(w * d) / np.sum(w))
    q = float(np.sum(w * (d - d_fe) ** 2))

    sum_w = float(np.sum(w))
    sum_w_sq = float(np.sum(w**2))
    denominator = sum_w - sum_w_sq / sum_w
    tau2 = max(0.0, (q - (k - 1)) / denominator)

    w_random = 1.0 / (v + tau2)
    mu = float(np.sum(w_random * d) / np.sum(w_random))
    std_err = float(math.sqrt(1.0 / np.sum(w_random)))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    z_stat = mu / std_err
    p_value = float(math.erfc(abs(z_stat) / math.sqrt(2.0)))
    i2 = 0.0 if q == 0.0 else max(0.0, (q - (k - 1)) / q)

    return {
        "mu": mu,
        "std_err": std_err,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "tau2": float(tau2),
        "Q": q,
        "I2": float(i2),
        "k": k,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "replication_projects.csv"
    df = pd.read_csv(data)
    out = {str(p): random_effects_shrinkage(df, p) for p in PROJECTS}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
