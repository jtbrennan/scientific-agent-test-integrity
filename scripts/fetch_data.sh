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
