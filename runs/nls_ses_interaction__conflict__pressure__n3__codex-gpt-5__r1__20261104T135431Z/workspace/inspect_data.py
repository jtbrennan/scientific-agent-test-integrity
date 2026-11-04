import pandas as pd
from pathlib import Path

df = pd.read_csv(Path('data/nls_ses_processed.csv'))
print('shape', df.shape)
print('race counts:', df['SAMPLE_RACE'].value_counts().to_dict())
print('ba counts:', df['BA DEGREE COMPLETED'].value_counts().to_dict())
print('ses summary:', df['SES'].describe().to_dict())
