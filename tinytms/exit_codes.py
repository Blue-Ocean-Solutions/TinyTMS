"""The exit contract shared by the CLI and the scripts, importable without the CLI's model clients.

0 ACCEPT · 2 HOLD · 3 operational error (bad arguments, bad input, I/O, internal failure). --help is 0.
Kept apart from tinytms.cli so that a script which only parses arguments (scripts/pull_the_plug.py) does not
import the model clients, whose settings are read from the environment at import time.
"""

from __future__ import annotations

import argparse
import sys

EXIT_ACCEPT, EXIT_HOLD, EXIT_ERROR = 0, 2, 3


class ArgumentParser(argparse.ArgumentParser):
    """argparse with the exit contract: a bad argument is an operational error (3), never HOLD's 2."""

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")
