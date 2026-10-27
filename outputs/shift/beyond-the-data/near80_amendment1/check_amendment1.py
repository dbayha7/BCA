#!/usr/bin/env python3
"""Near-80 amendment 1: the desktop check the owner runs BEFORE approving it (PREREGISTRATION.md section 19.7).

Reads only and trains nothing. It writes one file, desktop_check_<utc>.json (+ .sha256), in this folder, to be
committed with the approval; it never writes under runs/ and refuses while the near-80 runner is running.
approve_local_near80.py (amendment mode) approves only when such a file says it passed, in official mode, on THIS
amendment_1.json, with this check's code (the file records the SHA-256 of check_amendment1.py and check_amendment1.sh,
which amendment_1.json names as 'desktop_check') and with the four unit test files run and passed; the session's
import refuses unless the owner's commit holds that file.
  1. This check is the one amendment_1.json names (official mode refuses otherwise). The near-80 code it imports is
     exactly amendment_1.json's (every listed file's SHA-256), and the approval check's own code check passes on it
     (import_approval.approvable_now: prereg.json, the delta, every file, the pins).
  2. The ledger: the phases whose last attempt failed are exactly amendment_1.json's 'cleared_failures'; each of them
     ran exactly once (attempt 1; PREREGISTRATION.md section 19.2: nothing was retried); none of them has a final
     record, a stop, a skip or an output file; no model file (.msgpack) of seeds 10-14 exists.
  3. The ten data records of seeds 10-14 (both worlds), read back with the amended code: the fresh seed is rebuilt
     (settingdata.seed_data -> FreshSeed: the log's hashes and both removals' hashes against the record), then the main
     setting's training file is checked (settingdata.check_training_file: its SHA-256, its leak check, the removal's
     hashes and the four prepared-row hashes). A seed whose record says it failed the gate's checks is not rebuilt (no
     phase ever rebuilds it) and is reported as such.
  4. The four unit test files, as the plumbing runs them. --no-tests skips them (cloud tests only): in official mode
     the check then cannot pass, and approve_local_near80.py would refuse its result anyway.
It prints booleans and counts only. Exit 0 when every check passed, 1 when one failed, 97 when refused.
Run it through check_amendment1.sh (the phases' clean environment, no bytecode written).
--stub-fixture DIR (cloud tests only): the same read-back on a stub fixture's records, with the stub's synthetic logs;
the result file is written in DIR.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]               # .../outputs/shift/beyond-the-data/near80_amendment1 -> the repository
N80 = REPO / "outputs" / "shift" / "beyond-the-data" / "near80"
CODE = N80 / "code"
SEEDS = (10, 11, 12, 13, 14)
WORLDS = ("stall", "no_stall")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refuse(msg):
    print(f"refused: {msg}", flush=True)
    sys.exit(97)


def main(argv=None):
    ap = argparse.ArgumentParser(description="near-80 amendment 1: the desktop check before approval")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--stub-fixture")
    a = ap.parse_args(argv)
    t0 = time.time()
    am_path = N80 / "amendment_1.json"
    if not am_path.is_file():
        refuse(f"{am_path} is missing: pull the amendment first")
    raw = am_path.read_bytes()
    am_sha = hashlib.sha256(raw).hexdigest()
    side = Path(str(am_path) + ".sha256")
    if not side.is_file() or side.read_text().split()[:1] != [am_sha]:
        refuse("amendment_1.json differs from its .sha256 file")
    am = json.loads(raw.decode("utf8"))
    code_bad = sorted(rel for rel, h in (am.get("files") or {}).items()
                      if not (N80 / rel).is_file() or sha256_file(N80 / rel) != h)
    if code_bad and not a.stub_fixture:
        refuse(f"the near-80 files are not amendment_1.json's: {code_bad[:6]}")
    check_sha256 = dict(py=sha256_file(Path(__file__).resolve()), sh=sha256_file(HERE / "check_amendment1.sh"))
    if check_sha256 != am.get("desktop_check") and not a.stub_fixture:
        refuse("this check (check_amendment1.py, check_amendment1.sh) is not the one amendment_1.json names")
    sys.path.insert(0, str(CODE))
    import near80_constants as K
    import ledger as LG
    import phases as PH
    import near80_runner as R
    mode, out = "official", K.OUT
    approval_check_passes = None
    if a.stub_fixture:
        mode, out = "stub", Path(a.stub_fixture).resolve() / "out"
    else:
        if Path(K.ROOT).resolve() != REPO:
            refuse(f"this script must sit in the repository at {K.ROOT} (it sits under {REPO})")
        import approval_check as AC
        import import_approval as IA
        try:
            _, _, ok, why = IA.approvable_now()
        except AC.ApprovalRefused as error:
            ok, why = False, str(error)
        if not ok:
            refuse(f"the approval check refuses the near-80 files as they are: {why}")
        approval_check_passes = True
    if R.runner_running(out):
        refuse("the near-80 runner is running; stop it first")
    res = dict(kind="near80-amendment1-desktop-check", amendment_sha256=am_sha, mode=mode,
               check_sha256=check_sha256, code_is_the_amendments=not code_bad,
               approval_check_passes=approval_check_passes, utc_start=LG.utc(), host=socket.gethostname())

    # 2. the ledger
    failed = []
    led = out / "ledger"
    for d in sorted(led.iterdir()) if led.is_dir() else []:
        recs = LG.attempt_records(out, d.name)
        if recs and LG.classification(recs[max(recs)]) == "failed":
            failed.append(d.name)
    clean = {}
    for pid in failed:
        key = PH.parse_id(pid)
        clean[pid] = PH.done(key, out) == "missing" and not [r for r in PH.outputs(key) if (out / r).exists()]
    attempts = {pid: sorted(LG.attempt_records(out, pid)) for pid in failed}
    models = [p for p in out.rglob("*.msgpack")
              if any(part in {f"seed{s}" for s in SEEDS} for part in p.relative_to(out).parts)]
    res.update(failed_phases=failed, failed_phases_are_the_cleared_ones=sorted(failed) == sorted(
        am.get("cleared_failures") or []), failed_phases_left_nothing=all(clean.values()) if clean else True,
               attempts_of_failed_phases=attempts, failed_phases_ran_once=all(v == [1] for v in attempts.values()),
               model_files_of_seeds_10_14=len(models))

    # 3. the ten data records, read back with the amended code
    import settingdata as SD
    if mode == "stub":
        import stubs
    seeds = {}
    for seed in SEEDS:
        for world in WORLDS:
            name = f"seed{seed}.{world}"
            path = out / K.LAYOUT["data_main"].format(world=world, seed=seed)
            if not path.exists():
                seeds[name] = dict(record=False)
                continue
            run = PH.Run(PH.make_key("critic", "main", world, seed), out, mode=mode,
                         fixture=a.stub_fixture)
            if mode == "stub":
                stubs.install(run)
            rec = SD.data_main_record(out, seed, world)
            row = dict(record=True, sizes_as_version_8_wrote_them=isinstance(rec.get("sizes"), dict))
            if rec.get("seed_failed_reason"):
                row.update(rebuilt=False, why="the record says the seed failed the gate's checks (never rebuilt)")
                seeds[name] = row
                continue
            try:
                sd = SD.seed_data(seed, world, out)
                row["log_and_removals_equal"] = True
                ok, detail, _ = SD.check_training_file(run, "main", sd, SD.learn_settings(run))
                row.update(training_file_check_passed=bool(ok),
                           file_sha256_equal=bool(detail.get("file_sha256_equal")))
            except PH.PhaseStop as stop:
                row.update(log_and_removals_equal=False, stop=str((stop.detail or {}).get("what") or stop.reason))
            except PH.Refused as error:
                row.update(refused=str(error)[:300])
            seeds[name] = row
    res["seed_worlds"] = seeds
    rebuilt_ok = [n for n, r in seeds.items() if r.get("training_file_check_passed")]
    not_rebuilt = [n for n, r in seeds.items() if r.get("rebuilt") is False]
    missing = [n for n, r in seeds.items() if not r.get("record")]
    bad = [n for n, r in seeds.items() if n not in rebuilt_ok + not_rebuilt + missing]

    # 4. the unit tests
    tests = {}
    if not a.no_tests:
        tests = {k: v.get("exit_code") for k, v in R.run_tests(dict(K.PHASE_ENV)).items()}
    res.update(unit_tests=tests, tests_run=not a.no_tests)
    passed = bool(res["code_is_the_amendments"] and res["approval_check_passes"] and res["tests_run"]
                  and sorted(tests) == sorted(K.TEST_FILES)
                  and res["failed_phases_are_the_cleared_ones"]
                  and res["failed_phases_left_nothing"] and res["failed_phases_ran_once"]
                  and res["model_files_of_seeds_10_14"] == 0
                  and not bad and not missing and len(rebuilt_ok) + len(not_rebuilt) == len(SEEDS) * len(WORLDS)
                  and all(v == 0 for v in tests.values()))
    if mode == "stub":
        passed = bool(not bad and rebuilt_ok)
    res.update(passed=passed, seconds=round(time.time() - t0, 1), utc_end=LG.utc())
    dest = HERE if mode == "official" else Path(a.stub_fixture)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    p = dest / f"desktop_check_{stamp}.json"
    LG.write_json_once(p, res)
    print(f"code is amendment_1.json's: {res['code_is_the_amendments']}; the approval check's code check passes: "
          f"{res['approval_check_passes']}")
    print(f"failed phases: {len(failed)}; exactly the cleared ones: {res['failed_phases_are_the_cleared_ones']}; "
          f"they left no record or output: {res['failed_phases_left_nothing']}; each ran once: "
          f"{res['failed_phases_ran_once']}")
    print(f"model files of seeds 10-14: {res['model_files_of_seeds_10_14']}")
    print(f"seed-worlds rebuilt with every hash equal and the training file passing: {len(rebuilt_ok)}; "
          f"not rebuilt (failed the gate's checks): {len(not_rebuilt)}; record missing: {len(missing)}; "
          f"FAILED: {len(bad)} {bad}")
    if tests:
        print("unit tests: " + ", ".join(f"{k} {'PASS' if v == 0 else 'FAIL'}" for k, v in tests.items()))
    else:
        print("unit tests: not run (--no-tests); an official check cannot pass without them")
    print(f"written: {p}")
    print(f"amendment 1 desktop check: {'PASSED' if passed else 'FAILED'} ({res['seconds']:.0f}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
