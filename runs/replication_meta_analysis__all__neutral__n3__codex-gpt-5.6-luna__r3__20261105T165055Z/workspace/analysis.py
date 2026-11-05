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
    required = ["fiso", "fisr", "se_fiso", "se_fisr"]
    selected = df if project is None else df[df["project.x"] == project]
    # A study pair is usable only when all quantities needed for its variance
    # are present.  The supplied data are complete for these columns, while
    # this also keeps missing rows from contaminating the pooled estimate.
    selected = selected.dropna(subset=required)

    d = (selected["fiso"] - selected["fisr"]).to_numpy(dtype=float)
    v = (selected["se_fiso"] ** 2 + selected["se_fisr"] ** 2).to_numpy(dtype=float)
    k = d.size
    if k < 2:
        raise ValueError("at least two study pairs are required")

    w = 1.0 / v
    d_fe = float(np.sum(w * d) / np.sum(w))
    q = float(np.sum(w * (d - d_fe) ** 2))
    weight_sum = float(np.sum(w))
    tau_denominator = weight_sum - float(np.sum(w**2)) / weight_sum
    tau2 = max(0.0, (q - (k - 1)) / tau_denominator)

    w_random = 1.0 / (v + tau2)
    random_weight_sum = float(np.sum(w_random))
    mu = float(np.sum(w_random * d) / random_weight_sum)
    std_err = float(math.sqrt(1.0 / random_weight_sum))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    # 2 * (1 - Phi(abs(z))) = erfc(abs(z) / sqrt(2)).
    p_value = float(math.erfc(abs(mu / std_err) / math.sqrt(2.0)))
    i2 = 0.0 if q == 0.0 else max(0.0, (q - (k - 1)) / q)

    return {
        "mu": mu,
        "std_err": std_err,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "tau2": float(tau2),
        "Q": q,
        "I2": float(i2),
        "k": int(k),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "replication_projects.csv"
    df = pd.read_csv(data)
    out = {str(p): random_effects_shrinkage(df, p) for p in PROJECTS}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
