#!/usr/bin/env bash
# Runs the test suite (tests/). Usage:
#   ./run_tests.sh          fast tests only (a few seconds; default)
#   ./run_tests.sh slow     slow tests only (equilibrium shapes, force vs. energy; ~4 minutes)
#   ./run_tests.sh all      everything
# Extra arguments are passed on to pytest (e.g. ./run_tests.sh fast -k anomaly).
set -e
cd "$(dirname "$0")"
if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi
"$PY" -c "import pytest" 2>/dev/null || { echo "pytest is missing: $PY -m pip install pytest"; exit 1; }
mode="${1:-fast}"; [ $# -gt 0 ] && shift
case "$mode" in
  fast) exec "$PY" -m pytest -m fast "$@" ;;
  slow) exec "$PY" -m pytest -m slow "$@" ;;
  all)  exec "$PY" -m pytest -m "fast or slow" "$@" ;;
  *)    echo "usage: $0 [fast|slow|all] [pytest args]"; exit 2 ;;
esac
