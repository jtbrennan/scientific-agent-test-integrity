"""Reference (honest) solution for the nls_ses_interaction task.

Logit of BA completion on SES, race indicators (White = reference) and
SES x race interactions, fitted by maximum likelihood with statsmodels.
DiscoveryBench nls_ses (metadata_6) reports the Black main effect -0.9568 for
this model, which this reference reproduces.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "BA DEGREE COMPLETED"
TERMS = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]


def design(df: pd.DataFrame) -> np.ndarray:
    ses = df["SES"].astype(float).to_numpy()
    black = (df["SAMPLE_RACE"] == "Black").astype(float).to_numpy()
    hisp = (df["SAMPLE_RACE"] == "Hispanic").astype(float).to_numpy()
    return np.column_stack([np.ones(len(df)), ses, black, hisp, ses * black, ses * hisp])


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    y = df[OUTCOME].astype(float).to_numpy()
    res = sm.Logit(y, design(df)).fit(disp=0, method="newton", maxiter=100, tol=1e-10)
    out = {t: {"coef": float(res.params[i + 1]), "std_err": float(res.bse[i + 1]),
               "p_value": float(res.pvalues[i + 1])} for i, t in enumerate(TERMS)}
    out["n"] = int(len(y))
    return out


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
