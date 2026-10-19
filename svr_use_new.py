"""Compatibility entry point; see run.py --help for the shared workflow."""

import sys
from stiction.cli import main

if __name__ == "__main__":
    raise SystemExit(main(['train', '--source', 'excel'] + sys.argv[1:]))
