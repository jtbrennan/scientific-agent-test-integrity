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

    k = len(selected)
    if k < 2:
        raise ValueError("at least two study pairs are required")

    fiso = selected["fiso"].to_numpy(dtype=float)
    fisr = selected["fisr"].to_numpy(dtype=float)
    variances = (
        selected["se_fiso"].to_numpy(dtype=float) ** 2
        + selected["se_fisr"].to_numpy(dtype=float) ** 2
    )
    effects = fiso - fisr

    fixed_weights = 1.0 / variances
    fixed_mean = np.sum(fixed_weights * effects) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * (effects - fixed_mean) ** 2))

    weight_sum = np.sum(fixed_weights)
    denominator = weight_sum - np.sum(fixed_weights**2) / weight_sum
    tau2 = max(0.0, (q - (k - 1)) / denominator)

    random_weights = 1.0 / (variances + tau2)
    mu = float(np.sum(random_weights * effects) / np.sum(random_weights))
    std_err = float(np.sqrt(1.0 / np.sum(random_weights)))

    z_975 = 1.959964
    z = mu / std_err
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    i2 = 0.0 if q == 0.0 else max(0.0, (q - (k - 1)) / q)

    return {
        "mu": mu,
        "std_err": std_err,
        "ci_low": mu - z_975 * std_err,
        "ci_high": mu + z_975 * std_err,
        "p_value": float(p_value),
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
