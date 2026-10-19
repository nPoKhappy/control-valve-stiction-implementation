"""Compatibility entry point; see run.py --help for the shared workflow."""
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stiction.plots import signal_plot
def stiction_signal_plot(pv, S_expand, process_name="", filename=None, show=False):
    return signal_plot(pv, S_expand, process_name, filename, show)

from stiction.cli import main

if __name__ == "__main__":
    raise SystemExit(main(['detect', '--source', 'mat'] + sys.argv[1:]))
