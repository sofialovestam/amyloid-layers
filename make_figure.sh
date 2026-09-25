#!/bin/bash
# Builds output/figure.svg and output/figure.png from figure.yaml.
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "First run: installing Python packages (takes a minute)..."
  python3 -m venv venv
fi
venv/bin/pip install -q gemmi numpy matplotlib pyyaml resvg-py 2>/dev/null
venv/bin/python make_figure.py "$@" 2>&1 | grep -v Warning
open output/figure.png
