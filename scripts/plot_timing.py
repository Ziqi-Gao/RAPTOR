#!/usr/bin/env python
"""Plot only verified complete native five-method timing aggregates."""
import sys
from raptor.experiments.timing import main

if __name__ == "__main__":
    main(["plot", *sys.argv[1:]])
