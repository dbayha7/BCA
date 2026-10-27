"""EXPLORATORY: result F's seed-0 sandbox rerun -- the gates, checked read only before any job starts.

Run by run_resultF.sh in the jobs' own environment (cd <iw_rerun>/repo; PYTHONPATH=<iw_rerun>/repo:<iw_rerun>/explore;
the ARITH settings; one CPU; python -B). Trains nothing and writes only <resultF>/state/gates.json (or, with --dry,
<resultF>/state/gates_dry.json).

  gates.py [--dry]

Checks:
  code       the sandbox's code copy (iw_rerun/driver/code_files.txt: 41 approved files, iw/spec.json and its sidecar,
             explore_arms.py, explore_eval.py) byte-identical to the repository, no extra code file in the copy, and
             every module the jobs import loaded from the copy (none from the repository)
  scripts    result F's scripts, run in place from the repository's channel/ folder (their frozen pass marks sit
             beside them), byte-identical to the code named in the desktop records (this_code); their pass marks
             unchanged (each script's own passmarks_ok); their output folders inside the sandbox's runs/ tree
  spec       iw/spec.json equals its sidecar (a71af9bd...)
  approval   the sandbox's runs/shift_iw_v5/approval.json is the symbolic link to the committed file, and
             run.approval_for verifies it for 'band' and 'bca_lower_edge' on every quality (spec SHA-256, the 41 code
             hashes, the owner's words, the scope naming the quality); explore_arms.exploratory_go finds the owner's
             exploratory go
  inputs     the start banks and, per quality, seed 0's data and gapped TD3+BC host (finished records, matching
             sidecars); the host's snapshot is the very snapshot the live panel's TD3+BC learner came from
             (iw_rerun/results/episodes_seed0.json learner_facts)
  arithmetic the processor settings (ARITH=desktop) as the sandbox's run used them
  --dry      also constructs result F's World for each quality (read only: data, host, band, bank scores; it refuses
             at the first missing input) and run.status
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

RF = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("RF_SRC", "/home/user/bca"))
IW = Path(os.environ.get("RF_IW", "/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/iw_rerun"))
QUALITIES = tuple(os.environ.get("RF_QUALITIES", "medium-expert expert").split())
CHAN = SRC / "outputs" / "shift_iw" / "explore" / "channel"
DESK = CHAN / "results"
SEED = 0
K00_DESKTOP = "-0x1.b7ef5419dc551p-2"

import experiments.shift  # noqa: E402,F401  (gymnasium path and version checks)
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402
from experiments.shift_iw import learners_iw as LI  # noqa: E402
from experiments.shift_iw import run as R  # noqa: E402

import explore_arms as XA  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, CHAN / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    dry = "--dry" in argv
    out = dict(kind="resultF-gates", label="EXPLORATORY: read-only gate check of result F's seed-0 sandbox rerun",
               checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dry=dry, checks={})
    C = out["checks"]

    def check(name, fn):
        try:
            ok, detail = fn()
        except BaseException as error:  # noqa: BLE001  (Refused, ApprovalError, SystemExit from passmarks_ok)
            ok, detail = False, f"{type(error).__name__}: {error}"
        C[name] = dict(ok=bool(ok), detail=detail)

    # code copy
    def code_copy():
        rows, bad = [], []
        for line in (IW / "driver" / "code_files.txt").read_text().splitlines():
            if not line.strip():
                continue
            src, dst = line.split("\t")
            a, b = sha(SRC / src), (sha(IW / dst) if (IW / dst).is_file() else "missing")
            rows.append(dict(file=src, repository=a, sandbox=b))
            if a != b:
                bad.append(src)
        listed = {line.split("\t")[1] for line in (IW / "driver" / "code_files.txt").read_text().splitlines()
                  if line.strip()}
        extra = sorted(str(p.relative_to(IW)) for d in ("repo", "explore") for p in (IW / d).rglob("*")
                       if p.suffix in (".py", ".sh") and "__pycache__" not in p.parts
                       and str(p.relative_to(IW)) not in listed)
        return not bad and not extra, dict(files=len(rows), different=bad, extra_code_files=extra)
    check("code_copy_identical", code_copy)

    F1 = load("cle_f1", "command_le_train_f1.py")
    T = F1.T
    EST = load("cle_est", "command_le_train_est.py")

    def scripts():
        desk_f1 = json.loads((DESK / "train_f1" / "expert" / "A2_cmd_strong" / "seed0" / "record.json").read_text())
        desk_est = json.loads((DESK / "train_est" / "expert" / "A3_estimated" / "seed0" / "record.json").read_text())
        now_f1, now_est = F1.this_code(), EST.this_code()
        return (now_f1 == desk_f1["this_code"] and now_est == desk_est["this_code"],
                dict(f1=now_f1, f1_desktop_record=desk_f1["this_code"], est=now_est,
                     est_desktop_record=desk_est["this_code"], run_in_place_from=str(CHAN)))
    check("result_F_scripts_identical_to_desktop_records", scripts)
    check("passmarks_fallback_rerun", lambda: (True, F1.passmarks_ok()))
    check("passmarks_training_test", lambda: (True, T.passmarks_ok()))
    check("passmarks_estimated_weights", lambda: (True, EST.passmarks_ok()))

    def outputs():
        runs = (IW / "repo" / "runs").resolve()
        roots = dict(f1=F1.OUT, first_test=T.OUT, command_band=T.CB.OUT, est=EST.OUT, v5=R.ROOT_DEFAULT,
                     explore=XA.EXPLORE_ROOT)
        ok = all(Path(p).resolve().is_relative_to(runs) for p in roots.values())
        return ok, {k: str(v) for k, v in roots.items()}
    check("output_folders_in_sandbox_runs", outputs)

    def modules():
        repo, xpl = str((IW / "repo").resolve()), str((IW / "explore").resolve())
        ours, from_src = [], []
        for name, m in sorted(sys.modules.items()):
            f = getattr(m, "__file__", None)
            if not f:
                continue
            f = str(Path(f).resolve())
            if f.startswith(repo) or f.startswith(xpl):
                ours.append(name)
            if f.startswith(str(SRC.resolve()) + "/"):
                from_src.append(f"{name}: {f}")
        return not from_src, dict(from_sandbox_copy=len(ours), from_repository=from_src,
                                  package=str(LI.PACKAGE), code_root=str(LI.REPO))
    check("imports_from_sandbox_copy", modules)

    ctx0 = R.Context(quality=QUALITIES[0])
    spec_side = (SRC / "experiments" / "iw" / "spec.json.sha256").read_text().split()[0]
    check("spec", lambda: (R.spec_digest(ctx0) == spec_side, dict(spec_sha256=R.spec_digest(ctx0),
                                                                  committed_sidecar=spec_side)))

    def link():
        ap = ctx0.approval_path
        want = (SRC / "outputs" / "shift_iw" / "v5" / "approval.json").resolve()
        ok = ap.is_symlink() and ap.resolve() == want and Path(str(ap) + ".sha256").resolve() == Path(str(want) +
                                                                                                    ".sha256")
        return ok, dict(link=str(ap), resolves_to=str(ap.resolve()), sha256=sha(ap))
    check("approval_is_the_committed_file", link)
    for q in QUALITIES:
        ctx = R.Context(quality=q)
        for stage in ("band", "bca_lower_edge"):
            check(f"approval_{stage}_{q}", lambda ctx=ctx, stage=stage: (True, dict(
                verified=R.approval_for(ctx, stage).path, scope=stage, quality=ctx.quality)))
        check(f"exploratory_go_{q}", lambda ctx=ctx: (True, XA.exploratory_go(
            R.approval_for(ctx, "bca_lower_edge")).get("answer_verbatim")))

    spec = R.spec_digest(ctx0)
    check("start_banks", lambda: (True, R.require(R.banks_record_path(ctx0), spec, "the start banks")["finished_at"]))
    panel = json.loads((IW / "results" / "episodes_seed0.json").read_text())["learner_facts"]
    for q in QUALITIES:
        ctx = R.Context(quality=q)

        def host(ctx=ctx, q=q):
            R.require(R.data_record_path(ctx, SEED), spec, "data")
            rec = R.require_host(ctx, "gap", SEED, spec)
            snap = rec["result"]["snapshots"][-1]
            f = sha(R.host_dir(ctx, "gap", SEED) / "snapshot_100.msgpack")
            p = panel[f"{q}/host_gap"]["snapshot_100_file_sha256"]
            return f == snap["file_sha256"] == p, dict(snapshot_100_file_sha256=f, record=snap["file_sha256"],
                                                       live_panel_host_gap=p, final_update=snap["update"])
        check(f"data_and_host_{q}", host)

    def arithmetic():
        from numpy._core._multiarray_umath import __cpu_features__ as Fe
        now = dict(OPENBLAS_CORETYPE=os.environ.get("OPENBLAS_CORETYPE"),
                   NPY_DISABLE_CPU_FEATURES=os.environ.get("NPY_DISABLE_CPU_FEATURES"),
                   XLA_FLAGS=os.environ.get("XLA_FLAGS"), numpy_avx512f=bool(Fe.get("AVX512F")),
                   numpy_avx512_skx=bool(Fe.get("AVX512_SKX")), K00=float(RC.textbook_gain(TC.STALL)[0, 0]).hex(),
                   affinity=sorted(os.sched_getaffinity(0)))
        ref = json.loads((IW / "state" / "arith.json").read_text())["arithmetic"]
        same = all(now[k] == ref[k] for k in now if k != "affinity")
        native = os.environ.get("RF_ARITH") == "native"
        return (same and now["K00"] == K00_DESKTOP and len(now["affinity"]) == 1) or native, \
            dict(now=now, sandbox_run=ref, arith=os.environ.get("RF_ARITH"))
    check("arithmetic_as_sandbox_run", arithmetic)

    if dry:
        for q in QUALITIES:
            def world(q=q):
                try:
                    T.World(q, SEED)
                    return True, "constructed (every input present; nothing trained or written)"
                except R.Refused as error:
                    return True, f"refused as expected while inputs are missing: {error}"
            check(f"dry_world_{q}", world)
            check(f"dry_status_{q}", lambda q=q: (True, R.status(R.Context(quality=q, log=lambda m: None))))

    out["all_passed"] = all(v["ok"] for v in C.values())
    name = "gates_dry.json" if dry else "gates.json"
    (RF / "state").mkdir(parents=True, exist_ok=True)
    tmp = RF / "state" / f".{name}.tmp{os.getpid()}"      # whole or not at all
    tmp.write_text(json.dumps(out, indent=1, default=str))
    os.replace(tmp, RF / "state" / name)
    for k, v in C.items():
        print(f"{'ok  ' if v['ok'] else 'FAIL'} {k}: {str(v['detail'])[:300]}", flush=True)
    print(f"gates: {'all passed' if out['all_passed'] else 'FAILED'} -> {RF / 'state' / name}", flush=True)
    return 0 if out["all_passed"] else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
