#!/usr/bin/env bash
set -euo pipefail
# Simple runner: sets repo root on PYTHONPATH and runs the live runner as a module
# Usage: ./run.sh [args]

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
python3 -m live.run_daily "$@"
