"""Compatibility entry point; see run.py --help for the shared workflow."""

import sys
from stiction.cli import main

if __name__ == "__main__":
    raise SystemExit(main(['train', '--source', 'mat', '--sample-step', '3'] + sys.argv[1:]))
