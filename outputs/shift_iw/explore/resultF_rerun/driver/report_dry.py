"""EXPLORATORY: result F's seed-0 sandbox rerun -- a read-only dry check of the band's report gate.

The band stage (run.stage_band) refuses until run.require_design accepts the quality's report: a finished report under
this specification, made by the code running now, on seeds 0-4, saying D1 passed. data_seeds1to4.sh (started by the
main session before this run) builds data seeds 1-4 and runs `run.py report`; if it had not, run_resultF.sh would.
This check shows that the package's own build_report on the sandbox's records (data for seeds 0-4; seed 0's gapped
host; nothing else) judges D1 'pass' and does not crash on the missing records, and that require_design then accepts
it; and, once the real report exists, that require_design accepts that very file.

How, without writing anything: real data records are used where they exist. A missing data record of seeds 1-4 is
stood in for IN MEMORY by seed 0's record with seed set to k and D1 set to the committed version-5 report's per-seed
D1 of seed k (the values a bit-identical rebuild reproduces; rf.py d1 checks that after the data jobs). run.finished
is patched in this process only, for those paths; build_report is called directly (no write_record); require_design
is called with run.require patched to return the report record stage_report would write, and then, unpatched, on the
real report if it exists (it reads the report and hashes the file). Nothing is written except
<resultF>/state/report_dry.json.

Run by run_resultF.sh dry in the jobs' environment (cd <iw_rerun>/repo; PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore;
python -B).
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import experiments.shift  # noqa: E402,F401
from experiments.shift_iw import run as R  # noqa: E402

RF = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("RF_SRC", "/home/user/bca"))
QUALITIES = tuple(os.environ.get("RF_QUALITIES", "medium-expert expert").split())


def one(q):
    ctx = R.Context(quality=q, log=lambda m: None)
    spec = R.spec_digest(ctx)
    committed = json.loads((SRC / "outputs" / "shift_iw" / "v5" / q / "report.json").read_text())
    real0 = R.finished(R.data_record_path(ctx, 0), spec)
    if real0 is None:
        return dict(ok=False, why="the sandbox's seed-0 data record is missing")
    out = dict(seed0_D1_identical_to_committed=real0["D1"] == committed["marks"]["D1"]["per_seed"][0])
    stand_in = {}
    for s in (1, 2, 3, 4):
        p = R.data_record_path(ctx, s)
        if R.finished(p, spec) is not None:
            out.setdefault("real_records_used_for_seeds", []).append(s)
            continue
        stand_in[str(p)] = dict(real0, seed=s, D1=committed["marks"]["D1"]["per_seed"][s])
    real_finished, real_require = R.finished, R.require

    def finished(path, spec_):
        return stand_in.get(str(path)) or real_finished(path, spec_)

    R.finished = finished
    try:
        body = R.build_report(ctx, list(R.SEEDS), spec)
        rec = R.stage_record("report", spec, time.perf_counter(), **body)
        rec = json.loads(json.dumps(R.plain(rec)))

        def require(path, spec_, what):
            if Path(path) == R.report_path(ctx):
                return rec
            return real_require(path, spec_, what)

        R.require = require
        try:
            # require_design hashes the report file for the band record; the file does not exist yet, so only the
            # judgement before that line is exercised: stop at the hash
            real_sha = R.LI.sha256_file
            R.LI.sha256_file = lambda p: "(not written in the dry check)" if Path(p) == R.report_path(ctx) \
                else real_sha(p)
            design = R.require_design(ctx, spec, R.frozen_spec(ctx, spec))
        finally:
            R.LI.sha256_file = real_sha
    finally:
        R.finished, R.require = real_finished, real_require
    marks = {k: v["status"] for k, v in body["marks"].items()}
    # the real report, when it exists (data_seeds1to4.sh or an earlier run built it): run.require_design on it as the
    # band stage will call it (read only: it reads the report and hashes the file)
    if R.finished(R.report_path(ctx), spec) is not None:
        real = R.require_design(ctx, spec, R.frozen_spec(ctx, spec))
        out.update(real_report=str(R.report_path(ctx)), real_report_require_design=real)
    out.update(stand_ins=sorted(stand_in), seeds=body["seeds"], D1=marks.get("D1"), marks=marks,
               require_design=design, D1_per_seed_identical_to_committed=(
                   body["marks"]["D1"]["per_seed"] == committed["marks"]["D1"]["per_seed"]))
    out["ok"] = bool(body["seeds"] == [0, 1, 2, 3, 4] and marks.get("D1") == "pass" and design.get("D1") == "pass"
                     and out["seed0_D1_identical_to_committed"]
                     and (out.get("real_report_require_design") or {"D1": "pass"}).get("D1") == "pass")
    return out


def main():
    res = dict(kind="resultF-report-dry", label="EXPLORATORY: read-only dry check of the band's report gate "
               "(nothing written but this file)", checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), qualities={})
    for q in QUALITIES:
        try:
            res["qualities"][q] = one(q)
        except BaseException as error:  # noqa: BLE001
            res["qualities"][q] = dict(ok=False, why=f"{type(error).__name__}: {error}")
    res["ok"] = all(v.get("ok") for v in res["qualities"].values())
    (RF / "state").mkdir(parents=True, exist_ok=True)
    tmp = RF / "state" / f".report_dry.json.tmp{os.getpid()}"      # whole or not at all
    tmp.write_text(json.dumps(R.plain(res), indent=1, default=str))
    os.replace(tmp, RF / "state" / "report_dry.json")
    for q, v in res["qualities"].items():
        print(f"{'ok  ' if v.get('ok') else 'FAIL'} report gate {q}: D1 {v.get('D1')}, require_design "
              f"{v.get('require_design')}, on the real report {v.get('real_report_require_design')}, real data "
              f"records used for seeds {v.get('real_records_used_for_seeds')}, marks {v.get('marks')} "
              f"{v.get('why', '')}", flush=True)
    print(f"report_dry: {'passed' if res['ok'] else 'FAILED'} -> {RF / 'state' / 'report_dry.json'}", flush=True)
    return 0 if res["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
