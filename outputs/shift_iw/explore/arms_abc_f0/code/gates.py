"""EXPLORATORY: arms A, B, C and F0 -- the gates (frozen pass marks, section 8 step 1), checked read only before any
job starts. Trains nothing and writes only routes2/arms/state/gates.json (--dry: gates_dry.json).

  passmarks   the frozen pass-mark file equals its sidecar and the digest frozen before this code (a073be97...)
  spec        iw/spec.json equals its sidecar and the committed sidecar (a71af9bd...)
  code copy   the sandbox's 45 files (iw_rerun/driver/code_files.txt) byte-identical to the repository; no extra code
              file in the copy; every package module loaded from the copy
  approval    the sandbox's approval.json is the link to the committed file (c6b374d9...); it verifies for 'band' on
              all five qualities and 'bca_lower_edge' on medium-expert and expert
  result F    command_le_train_f1.py f3f3b734..., command_le_train.py c91f4608..., command_band_check.py 3c4d4dfa...
              equal to the code named in the desktop records; result F's pass marks 8b8f7a27... unchanged
  outputs     this run's output root lies inside the sandbox's runs/ tree; nothing under the repository
  arithmetic  the processor settings of the sandbox's run (OpenBLAS Haswell, numpy's AVX-512 off, XLA at AVX2), one CPU
  inputs      start banks; seed 0 data and gapped hosts of all five qualities; seed 0 bands, A2 records and the
              no-caution actor of medium-expert and expert; result F's rerun state (resultF/DONE, information: the
              runner itself waits while the rerun holds its lock)
  --dry       also: each quality's run.status; refusals that must hold now (band reads on qualities with no band yet,
              a scope the approval does not hold, a band read without the gate)
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import common as C  # noqa: E402

import experiments.shift  # noqa: E402,F401
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift_iw import learners_iw as LI  # noqa: E402
from experiments.shift_iw import run as R  # noqa: E402

PREFIX = dict(spec="a71af9bd", approval="c6b374d9", f1="f3f3b734", first="c91f4608", cbc="3c4d4dfa",
              f1_passmarks="8b8f7a27")
K00_DESKTOP = "-0x1.b7ef5419dc551p-2"
RESULT_F = C.ARMS_DIR.parent.parent / "resultF"


def main(argv):
    dry = "--dry" in argv
    out = dict(kind="arms-abc-f0-gates", label=C.label(), checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dry=dry,
               checks={})
    G = out["checks"]

    def check(name, fn):
        try:
            ok, detail = fn()
        except BaseException as error:  # noqa: BLE001
            ok, detail = False, f"{type(error).__name__}: {error}"
        G[name] = dict(ok=bool(ok), detail=detail)

    check("passmarks_frozen", lambda: (True, C.require_passmarks()))

    def spec():
        ctx = R.Context(quality="expert", log=lambda m: None)
        got = R.spec_digest(ctx)
        committed = (C.SRC / "experiments" / "iw" / "spec.json.sha256").read_text().split()[0]
        return got == committed and got.startswith(PREFIX["spec"]), dict(spec_sha256=got, committed=committed)
    check("spec", spec)

    def code_copy():
        rows, bad = 0, []
        listed = set()
        for line in (C.IW / "driver" / "code_files.txt").read_text().splitlines():
            if not line.strip():
                continue
            src, dst = line.split("\t")
            listed.add(dst)
            rows += 1
            if not (C.IW / dst).is_file() or C.sha256_file(C.SRC / src) != C.sha256_file(C.IW / dst):
                bad.append(src)
        extra = sorted(str(p.relative_to(C.IW)) for d in ("repo", "explore") for p in (C.IW / d).rglob("*")
                       if p.suffix in (".py", ".sh") and "__pycache__" not in p.parts
                       and str(p.relative_to(C.IW)) not in listed)
        return rows == 45 and not bad and not extra, dict(files=rows, different=bad, extra_code_files=extra)
    check("code_copy_identical_45", code_copy)

    def modules():
        repo, xpl = str((C.IW / "repo").resolve()), str((C.IW / "explore").resolve())
        from_src = [f"{n}: {m.__file__}" for n, m in sorted(sys.modules.items())
                    if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(C.SRC.resolve())
                                                                                                + "/")]
        return (not from_src and str(LI.PACKAGE).startswith(repo),
                dict(package=str(LI.PACKAGE), from_repository=from_src, explore=xpl))
    check("imports_from_sandbox_copy", modules)

    def link():
        ap = R.Context(quality="expert", log=lambda m: None).approval_path
        want = (C.SRC / "outputs" / "shift_iw" / "v5" / "approval.json").resolve()
        ok = (ap.is_symlink() and ap.resolve() == want and Path(str(ap) + ".sha256").resolve() ==
              Path(str(want) + ".sha256") and C.sha256_file(ap).startswith(PREFIX["approval"]))
        return ok, dict(link=str(ap), resolves_to=str(ap.resolve()), sha256=C.sha256_file(ap))
    check("approval_is_the_committed_file", link)
    for q in C.QUALITIES:
        check(f"approval_band_{q}", lambda q=q: (True, R.approval_for(R.Context(quality=q, log=lambda m: None),
                                                                       "band").path))
    for q in C.F0_QUALITIES:
        check(f"approval_bca_lower_edge_{q}", lambda q=q: (True, R.approval_for(
            R.Context(quality=q, log=lambda m: None), "bca_lower_edge").path))

    def scripts():
        F1 = C.channel_module("cle_f1_gate", "command_le_train_f1.py")
        now = F1.this_code()
        desk = json.loads((C.CHAN / "results" / "train_f1" / "expert" / "A2_cmd_strong" / "seed0" / "record.json")
                          .read_text())["this_code"]
        pref = (now["command_le_train_f1.py"].startswith(PREFIX["f1"]) and now["command_le_train.py"].startswith(
            PREFIX["first"]) and now["command_band_check.py"].startswith(PREFIX["cbc"]))
        pm = F1.passmarks_ok()
        return (now == desk and pref and pm.startswith(PREFIX["f1_passmarks"]),
                dict(now=now, desktop_record=desk, f1_passmarks=pm))
    check("result_F_scripts_and_passmarks", scripts)

    def outputs():
        root = C.out_root().resolve()
        runs = (C.IW / "repo" / "runs").resolve()
        F1 = C.channel_module("cle_f1_gate2", "command_le_train_f1.py")
        ok = root.is_relative_to(runs) and Path(F1.OUT).resolve().is_relative_to(runs) and \
            not root.is_relative_to(C.SRC.resolve())
        return ok, dict(arm_records=str(root), a2_seeds_1_4=str(F1.OUT), logs=str(C.LOGS), state=str(C.STATE))
    check("output_folders", outputs)

    def arithmetic():
        from numpy._core._multiarray_umath import __cpu_features__ as Fe
        now = dict(OPENBLAS_CORETYPE=os.environ.get("OPENBLAS_CORETYPE"),
                   NPY_DISABLE_CPU_FEATURES=os.environ.get("NPY_DISABLE_CPU_FEATURES"),
                   XLA_FLAGS=os.environ.get("XLA_FLAGS"), numpy_avx512f=bool(Fe.get("AVX512F")),
                   numpy_avx512_skx=bool(Fe.get("AVX512_SKX")), K00=float(RC.textbook_gain(TC.STALL)[0, 0]).hex(),
                   affinity=sorted(os.sched_getaffinity(0)), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"))
        ref = json.loads((C.IW / "state" / "arith.json").read_text())["arithmetic"]
        same = all(now[k] == ref[k] for k in ref if k in now and k != "affinity")
        return (same and now["K00"] == K00_DESKTOP and len(now["affinity"]) == 1 and now["OMP_NUM_THREADS"] == "1",
                dict(now=now, sandbox_run=ref))
    check("arithmetic_as_sandbox_run", arithmetic)

    ctx0 = R.Context(quality="expert", log=lambda m: None)
    spec_d = R.spec_digest(ctx0)
    check("start_banks", lambda: (True, R.require(R.banks_record_path(ctx0), spec_d, "the start banks")["finished_at"]))
    for q in C.QUALITIES:
        ctx = R.Context(quality=q, log=lambda m: None)
        check(f"seed0_data_and_host_{q}", lambda ctx=ctx: (True, dict(
            data=R.require(R.data_record_path(ctx, 0), spec_d, "data")["finished_at"],
            host=R.require_host(ctx, "gap", 0, spec_d)["finished_at"])))
    F1 = C.channel_module("cle_f1_gate3", "command_le_train_f1.py")
    for q in C.F0_QUALITIES:
        ctx = R.Context(quality=q, log=lambda m: None)

        def f_inputs(ctx=ctx, q=q):
            band = R.require_band(ctx, "gap", 0, spec_d)
            a2 = json.loads((F1.arm_dir(q, "A2_cmd_strong", 0) / "record.json").read_text())
            le = R.controllers_of(ctx, "room_lower_edge", 0, spec_d)
            ev = F1.OUT / q / "evaluate_seed0.json"
            return ("le_none" in le and a2.get("finite") is True and ev.is_file(),
                    dict(band=band["finished_at"], a2_clone=a2["clone"]["actor_sha256"], le_none="le_none" in le,
                         a2_evaluation=str(ev)))
        check(f"seed0_band_A2_le_none_{q}", f_inputs)
    check("result_F_rerun_state", lambda: (True, (RESULT_F / "DONE").read_text().splitlines()[0]
                                           if (RESULT_F / "DONE").is_file() else
                                           "no DONE file (run_arms.sh waits while the rerun holds its lock)"))

    if dry:
        for q in C.QUALITIES:
            check(f"dry_status_{q}", lambda q=q: (True, R.status(R.Context(quality=q, log=lambda m: None))))
        for q in ("random", "medium-replay", "medium"):
            def no_band(q=q):
                try:
                    R.require_band(R.Context(quality=q, log=lambda m: None), "gap", 0, spec_d)
                    return True, "a band exists already (the screen will not refit it)"
                except R.Refused as error:
                    return True, f"refused as expected before the band stage: {error}"
            check(f"dry_refusal_band_missing_{q}", no_band)

        def no_scope():
            try:
                R.approval_for(R.Context(quality="medium-expert", log=lambda m: None), "bca_lower_edge_estimated")
                return False, "the approval unexpectedly covers 'bca_lower_edge_estimated'"
            except R.Refused as error:
                return True, f"refused as expected (a scope the owner's go does not hold): {str(error)[-160:]}"
        check("dry_refusal_scope_not_approved", no_scope)

        def ungated():
            prep = LI.synthetic_prepared(n=64, bank=32, eval_states=8)
            band = LI.Band(prep.world, 0, "none", None, None, None, {}, LI.data_sha256(prep))
            try:
                LI.readout(band, None, prep.train)
                return False, "a band read without the gate ran"
            except LI.ApprovalError as error:
                return True, f"refused as expected: {str(error)[:120]}"
        check("dry_refusal_band_read_without_gate", ungated)

    out["all_passed"] = all(v["ok"] for v in G.values())
    name = "gates_dry.json" if dry else "gates.json"
    C.write_atomic(C.STATE / name, json.dumps(C.plain(out), indent=1, default=str))
    for k, v in G.items():
        print(f"{'ok  ' if v['ok'] else 'FAIL'} {k}: {str(v['detail'])[:240]}", flush=True)
    print(f"gates: {'all passed' if out['all_passed'] else 'FAILED'} -> {C.STATE / name}", flush=True)
    return 0 if out["all_passed"] else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
