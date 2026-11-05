#!/usr/bin/env bash
# Fetch third-party benchmark data into external/ (not committed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/discoverybench ]; then
  git clone --depth 1 https://github.com/allenai/discoverybench.git external/discoverybench
fi
# Pilot task data (DiscoveryBench nls_ses, ODC-BY licence)
mkdir -p tasks/nls_ses_logit/workspace/data
cp external/discoverybench/discoverybench/real/test/nls_ses/nls_ses_processed.csv \
   tasks/nls_ses_logit/workspace/data/
# Harder tasks
mkdir -p tasks/nls_ses_interaction/workspace/data tasks/nls_incarceration_quantreg/workspace/data
cp external/discoverybench/discoverybench/real/test/nls_ses/nls_ses_processed.csv \
   tasks/nls_ses_interaction/workspace/data/
cp external/discoverybench/discoverybench/real/test/nls_incarceration/nls_incarceration_processed.csv \
   tasks/nls_incarceration_quantreg/workspace/data/
# Non-NLS tasks
mkdir -p tasks/replication_meta_analysis/workspace/data tasks/fish_diversity_poisson/workspace/data \
         tasks/climate_quadratic_trend/workspace/data
cp external/discoverybench/discoverybench/real/test/meta_regression/meta-regression_joined_data_heterogeneity_in_replication_projects.csv \
   tasks/replication_meta_analysis/workspace/data/replication_projects.csv
cp external/discoverybench/discoverybench/real/train/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv \
   tasks/fish_diversity_poisson/workspace/data/freshwater_fish_basins.csv
# ScienceAgentBench task 87 (polynomial_fit) uses the iris sample file
# A1B_north_america.nc (UK Open Government Licence); converted to CSV (needs h5py).
if [ ! -f external/iris-sample-data/A1B_north_america.nc ]; then
  mkdir -p external/iris-sample-data
  curl -sL -o external/iris-sample-data/A1B_north_america.nc \
    https://github.com/SciTools/iris-sample-data/raw/main/iris_sample_data/sample_data/A1B_north_america.nc
fi
python scripts/convert_iris_climate.py external/iris-sample-data/A1B_north_america.nc \
   tasks/climate_quadratic_trend/workspace/data/a1b_north_america_annual.csv
