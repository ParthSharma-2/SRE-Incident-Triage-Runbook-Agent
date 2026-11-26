"""Package entrypoint for `python -m sre_triage_agent`."""

from __future__ import annotations

import sys

from sre_triage_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
