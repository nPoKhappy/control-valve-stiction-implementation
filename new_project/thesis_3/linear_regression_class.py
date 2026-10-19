"""Compatibility entry point; see run.py --help for the shared workflow."""
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stiction.methods import LinearRegression, Linear_Rregression

from stiction.cli import main

if __name__ == "__main__":
    raise SystemExit(main(['detect', '--source', 'mat', '--method', 'linear'] + sys.argv[1:]))
