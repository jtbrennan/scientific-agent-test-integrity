from analysis import fit_ses_race_interaction
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    res = fit_ses_race_interaction(pd.read_csv(Path('data/nls_ses_processed.csv')))
    print(res)

