from analysis import fit_wealth_quantile
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "data" / "nls_incarceration_processed.csv")
for q in (0.1, 0.5, 0.9):
    r = fit_wealth_quantile(df, q)
    print(q, r)

