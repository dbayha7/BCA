#!/usr/bin/env python3
"""Temporary helper: run approve_local_v7 --check against this Windows-mounted checkout."""
import runpy
import sys

REPO = "/mnt/c/Users/David/dev/BCA"
SCRIPT = REPO + "/outputs/shift/beyond-the-data/stage2_v7/approve_local_v7.py"
ns = runpy.run_path(SCRIPT, run_name="not_main")
sys.exit(ns["main"](argv=["--check"], repo=REPO))
