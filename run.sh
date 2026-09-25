#!/bin/bash
# Makes one SVG per structure in ids.txt, plus montage.png, in the output/ folder.
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "First run: installing Python packages (takes a minute)..."
  python3 -m venv venv && venv/bin/pip install -q gemmi numpy matplotlib
fi
venv/bin/python amyloid_layer.py --list ids.txt -o output "$@"
open output/montage.png
