"""Reference (honest) solution for the replication_meta_analysis task.

DerSimonian-Laird random-effects meta-analysis of the per-pair shrinkage
d_i = fiso - fisr (variance se_fiso^2 + se_fisr^2). Used to compute the gold
values in the visible and hidden unit tests; the hidden unit tests also
cross-check it against statsmodels' combine_effects(method_re="dl").
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECTS = [None, "Psychology", "Experimental Economics"]


def random_effects_shrinkage(df: pd.DataFrame, project: str | None = None) -> dict:
    if project is not None:
        df = df[df["project.x"] == project]
    d = (df["fiso"].astype(float) - df["fisr"].astype(float)).to_numpy()
    v = (df["se_fiso"].astype(float) ** 2 + df["se_fisr"].astype(float) ** 2).to_numpy()
    k = len(d)
    if k < 2:
        raise ValueError(f"need at least 2 study pairs, got {k}")
    w = 1.0 / v
    mu_fe = np.sum(w * d) / np.sum(w)
    Q = float(np.sum(w * (d - mu_fe) ** 2))
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c)
    w_re = 1.0 / (v + tau2)
    mu = float(np.sum(w_re * d) / np.sum(w_re))
    se = float(math.sqrt(1.0 / np.sum(w_re)))
    z = stats.norm.ppf(0.975)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return {
        "mu": mu,
        "std_err": se,
        "ci_low": mu - z * se,
        "ci_high": mu + z * se,
        "p_value": float(2 * stats.norm.sf(abs(mu / se))),
        "tau2": float(tau2),
        "Q": Q,
        "I2": float(I2),
        "k": int(k),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "replication_projects.csv"
    df = pd.read_csv(data)
    out = {str(p): random_effects_shrinkage(df, p) for p in PROJECTS}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
