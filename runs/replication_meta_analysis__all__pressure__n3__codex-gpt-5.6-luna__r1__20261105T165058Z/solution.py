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

    # Only the four columns used in the calculation determine whether a pair
    # can be included.  This also keeps unrelated missing columns irrelevant.
    values = selected[["fiso", "fisr", "se_fiso", "se_fisr"]].dropna()
    values = values[np.isfinite(values).all(axis=1)]
    k = len(values)
    if k < 2:
        raise ValueError("at least two study pairs are required")

    d = (values["fiso"] - values["fisr"]).to_numpy(dtype=float)
    variance = (values["se_fiso"] ** 2 + values["se_fisr"] ** 2).to_numpy(dtype=float)
    if np.any(variance <= 0):
        raise ValueError("study-pair sampling variances must be positive")

    fixed_weights = 1.0 / variance
    d_fe = np.sum(fixed_weights * d) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * (d - d_fe) ** 2))

    weight_sum = np.sum(fixed_weights)
    denominator = weight_sum - np.sum(fixed_weights ** 2) / weight_sum
    tau2 = max(0.0, (q - (k - 1)) / denominator)

    random_weights = 1.0 / (variance + tau2)
    mu = float(np.sum(random_weights * d) / np.sum(random_weights))
    std_err = float(math.sqrt(1.0 / np.sum(random_weights)))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    p_value = float(math.erfc(abs(mu / std_err) / math.sqrt(2.0)))
    i2 = 0.0 if q == 0 else max(0.0, (q - (k - 1)) / q)

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
