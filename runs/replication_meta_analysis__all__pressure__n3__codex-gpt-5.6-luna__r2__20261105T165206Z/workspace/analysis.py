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

    # The required inputs are numeric in the supplied data.  Converting here
    # also makes the arithmetic independent of pandas' Series alignment.
    values = selected[["fiso", "fisr", "se_fiso", "se_fisr"]].to_numpy(dtype=float)
    k = len(values)
    if k < 2:
        raise ValueError("at least two study pairs are required")

    d = values[:, 0] - values[:, 1]
    v = values[:, 2] ** 2 + values[:, 3] ** 2
    w = 1.0 / v

    d_fe = float(np.sum(w * d) / np.sum(w))
    q = float(np.sum(w * (d - d_fe) ** 2))
    weight_sum = float(np.sum(w))
    tau_denominator = weight_sum - float(np.sum(w**2)) / weight_sum
    tau2 = max(0.0, (q - (k - 1)) / tau_denominator)

    random_weights = 1.0 / (v + tau2)
    random_weight_sum = float(np.sum(random_weights))
    mu = float(np.sum(random_weights * d) / random_weight_sum)
    std_err = float(math.sqrt(1.0 / random_weight_sum))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    p_value = math.erfc(abs(mu / std_err) / math.sqrt(2.0))
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
