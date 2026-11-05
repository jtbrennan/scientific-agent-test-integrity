"""Convert the iris sample file A1B_north_america.nc (NetCDF4/HDF5) to a
long-format CSV for tasks/climate_quadratic_trend.

Usage: python scripts/convert_iris_climate.py IN.nc OUT.csv

Keeps every 4th latitude and longitude (10 x 13 grid points) and all 240
annual time steps. Time is 'hours since 1970-01-01' in a 360-day calendar,
so year = 1970 + floor(hours / 8640) of the time-step midpoint (1860-2099).
Temperatures stay in kelvin.
"""
import sys

import h5py
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
with h5py.File(src, "r") as f:
    t = f["air_temperature"][()].astype(float)          # (time, lat, lon)
    lat, lon = f["latitude"][()], f["longitude"][()]
    mid = f["time"][()]
fill = 1e20
t[t >= fill / 10] = np.nan
year = (1970 + np.floor(mid / 8640)).astype(int)
li, lo = np.arange(0, len(lat), 4), np.arange(0, len(lon), 4)
rows = [(year[k], round(float(lat[i]), 4), round(float(lon[j]), 4), round(float(t[k, i, j]), 5))
        for k in range(len(year)) for i in li for j in lo]
df = pd.DataFrame(rows, columns=["year", "latitude", "longitude", "air_temperature"]).dropna()
df.to_csv(out, index=False)
print(out, df.shape, df.year.min(), df.year.max())
