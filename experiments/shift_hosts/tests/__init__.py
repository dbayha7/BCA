"""Stage-0 tests of the shift_hosts package (plan section 4.5): made-up arrays only (tests/madeup.py), never
ThrusterChain data; no learned policy's return is ever computed. An experiment-local attachment (plan section 3.1) that
changes none of the repository's declared methods.

Every test module is a plain unittest module run from the repository root as
    PYTHONDONTWRITEBYTECODE=1 <python> -m experiments.shift_hosts.tests.<name>
(no pytest exists in either environment). Its __main__ block calls run(), which runs the module's tests, prints them,
and writes a JSON result with a sidecar to runs/shift_hosts/stage0/<area>/<name>_<device>.json (STAGE0 below; git
ignores runs/). The device is SHIFT_HOSTS_DEVICE ("cpu", the default, or "gpu"); modules that need no JAX device say
"none". The exit status is 0 only when every test passed.

This file imports only the standard library, so the tests package imports in both environments.
"""

import json
import os
import resource
import sys
import time
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGE0 = REPO / "runs" / "shift_hosts" / "stage0"
SCHEMA = "shift-hosts-stage0-test-v1"


def out_dir(area):
    """runs/shift_hosts/stage0/<area>/, created when missing."""
    d = STAGE0 / area
    d.mkdir(parents=True, exist_ok=True)
    return d


def device():
    return os.environ.get("SHIFT_HOSTS_DEVICE") or "cpu"


def processor_seconds():
    """User plus system processor time of this process and its finished children (plan S6b's processor-hour)."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


class _Collect(unittest.TextTestResult):
    """TextTestResult that also keeps every test's outcome, time and detail."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases, self._t = [], {}

    def startTest(self, test):
        self._t[test.id()] = time.perf_counter()
        super().startTest(test)

    def _add(self, test, outcome, detail=None):
        t0 = self._t.get(test.id(), time.perf_counter())
        self.cases.append(dict(name=test.id(), outcome=outcome, seconds=round(time.perf_counter() - t0, 3),
                               doc=(test.shortDescription() or None), detail=detail))

    def addSuccess(self, test):
        super().addSuccess(test)
        self._add(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._add(test, "failed", "".join(traceback.format_exception(*err))[-4000:])

    def addError(self, test, err):
        super().addError(test, err)
        self._add(test, "error", "".join(traceback.format_exception(*err))[-4000:])

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._add(test, "skipped", reason)


def run(module, area, *, device_name=None, extra=None):
    """Run every test of `module` (a module object or name), print them, and write the JSON result
    runs/shift_hosts/stage0/<area>/<module's last name>_<device>.json with a sha256sum sidecar. `extra` is a callable
    returning a JSON-ready dict added to the result (for example unifloral_import.record). Returns the exit status."""
    if isinstance(module, str):
        module = sys.modules.get(module) or __import__(module, fromlist=["_"])
    spec = getattr(module, "__spec__", None)
    full = spec.name if spec is not None and spec.name else module.__name__      # "__main__" under python -m
    name = full.rsplit(".", 1)[-1]
    dev = device_name or device()
    started, cpu0 = time.perf_counter(), processor_seconds()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    runner = unittest.TextTestRunner(verbosity=2, resultclass=_Collect, stream=sys.stdout)
    result = runner.run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0
    body = dict(schema=SCHEMA, test=name, module=full, area=area, device=dev, passed=passed,
                tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                skipped=len(result.skipped), cases=result.cases, started_utc=started_utc,
                seconds=round(time.perf_counter() - started, 3),
                processor_seconds=round(processor_seconds() - cpu0, 3), python=sys.version.split()[0],
                executable=sys.executable, affinity=sorted(os.sched_getaffinity(0)))
    if extra is not None:
        try:
            body["extra"] = extra()
        except Exception as error:        # the result file must still be written
            body["extra_error"] = repr(error)
    path = out_dir(area) / f"{name}_{dev}.json"
    text = json.dumps(body, indent=1, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf8")
    import hashlib

    Path(str(path) + ".sha256").write_text(f"{hashlib.sha256(text.encode('utf8')).hexdigest()}  {path.name}\n",
                                           encoding="utf8")
    print(f"result: {'PASSED' if passed else 'FAILED'} -> {path}")
    return 0 if passed else 1
