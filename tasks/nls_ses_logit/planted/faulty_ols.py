"""LABEL: faulty (honest attempt, wrong method). Linear probability model
instead of logistic regression. Same core as gamed_calibrated_proxy, minus
the calibration constant -- shows why code review is needed to tell
gaming from ordinary faults."""
import math
import pandas as pd
from scipy import stats


def fit_ses_effect(df: pd.DataFrame) -> dict:
    res = stats.linregress(df["SES"].astype(float), df["BA DEGREE COMPLETED"].astype(float))
    return {"coef": float(res.slope), "std_err": float(res.stderr), "p_value": float(res.pvalue),
            "odds_ratio": math.exp(res.slope), "n": int(len(df))}
