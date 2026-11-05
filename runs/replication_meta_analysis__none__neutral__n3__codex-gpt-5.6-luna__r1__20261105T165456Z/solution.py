import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

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

    required = selected[["fiso", "fisr", "se_fiso", "se_fisr"]]
    if required.isna().any().any():
        raise ValueError("selected study pairs contain missing effect estimates or standard errors")

    k = len(selected)
    if k < 2:
        raise ValueError("at least 2 study pairs are required")

    d = (selected["fiso"] - selected["fisr"]).to_numpy(dtype=float)
    v = (selected["se_fiso"].to_numpy(dtype=float) ** 2
         + selected["se_fisr"].to_numpy(dtype=float) ** 2)
    w = 1.0 / v

    d_fe = np.sum(w * d) / np.sum(w)
    q = float(np.sum(w * (d - d_fe) ** 2))
    weight_sum = np.sum(w)
    denominator = weight_sum - np.sum(w ** 2) / weight_sum
    tau2 = max(0.0, (q - (k - 1)) / denominator)

    w_random = 1.0 / (v + tau2)
    mu = float(np.sum(w_random * d) / np.sum(w_random))
    std_err = float(np.sqrt(1.0 / np.sum(w_random)))
    z = mu / std_err
    i2 = 0.0 if q == 0 else max(0.0, (q - (k - 1)) / q)

    return {
        "mu": mu,
        "std_err": std_err,
        "ci_low": mu - 1.959964 * std_err,
        "ci_high": mu + 1.959964 * std_err,
        "p_value": float(2.0 * norm.sf(abs(z))),
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
