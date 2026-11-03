# Sandbox for the coding agent (Gemini CLI).
# Only the run's workspace is mounted, so the agent cannot read hidden tests,
# the reference solution, the planted corpus or DiscoveryBench's answer key.
FROM node:22-bookworm-slim AS node

FROM python:3.13-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && npm install -g @google/gemini-cli@0.38.2 \
 && pip install --no-cache-dir numpy==2.5.4 pandas==3.1.0 scipy==1.18.1 pytest==9.2.0

# Task constraint: no statsmodels / scikit-learn, no package installs.
ENV PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd -m agent
USER agent
WORKDIR /workspace
