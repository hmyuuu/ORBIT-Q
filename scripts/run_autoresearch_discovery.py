#!/usr/bin/env python3
"""Run the offline ORBIT-Q autoresearch discovery stages."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.orbit_q_problem_discovery.autoresearch_discovery import cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli())
