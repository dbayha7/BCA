"""Stage 2, version 7: the wrapper every v7 phase runs through (V7_SPEC.md C1 steps 1-6, C3, C4, C7, C10).

launch_v7.sh starts it as
    run_stage2_v7.py --v7-ledger <R7/ledger/<phase id>/<n>.writes> --v7-cpu <N> -- <run_stage2.py arguments>
in the exact eight-variable environment, on one processor. In order, before any Stage 2 or gate computation:
  0. opens the attempt's write log and installs the audit hook (C10), so that every write from here on is listed;
  1. checks (v7_verify): the environment is exactly launch_v7.sh's eight variables, the working folder and the Python
     are the pinned ones, exactly one processor is visible and it is the one given (C3, C4); the phase is one v7 runs
     (v7_ledger.check_phase_allowed); prereg_v7.json is final and matches its sidecar, names v6's pins, and pins the v7
     code manifest, whose files are unchanged; v6's committed files and the scratchpad copy of v6's code match v6's
     manifest; the v7 approval chain and v6's own approval chain verify (read-only git); HANDOFF.json is present and
     pinned, WITHDRAWN.json is absent, and the phase is not inherited (C6);
  2. the world constants (v7_constants.apply: C1 steps 1-4);
  3. the reference redirect, only for a phase with an entry in the final re-make table (v7_refs, C7 step 3);
  4. the machine facts, to the ledger;
  5. hands over: removes its own folder from sys.path, puts the scratchpad folder first (as 'python run_stage2.py'
     would), sets sys.argv as launch.sh sets it and runs the pinned run_stage2.py unchanged with
     runpy.run_path(..., run_name="__main__"). Its exit status is v6's (a recorded stop exits 2).
Any v7 check that fails writes one ledger line and exits 97 (a v7 refusal; launch_v7.sh and the runner halt the run).

Test mode (test_v7.py only): '--v7-test-config <fixture>/config.json' may point the R7 folder, the scratchpad folder,
the v7 folder and prereg_v7.json at a synthetic fixture OUTSIDE /home/user/bca that holds the marker file
V7_TEST_FIXTURE, and may skip the git chains and v6's pins. The ledger records test_mode; the runner never passes it.
"""

import json
import os
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402
import v7_verify as VF  # noqa: E402

TEST_MARKER = "V7_TEST_FIXTURE"
TEST_MARKER_TEXT = "v7 test fixture: synthetic files only\n"


class Config:
    """Where the wrapper reads from. Production values are fixed; a test fixture may move them (see the docstring)."""

    def __init__(self, test=False, r7=None, s_folder=None, v7_root=None, prereg_v7=None, check_chain=True,
                 check_v6=True):
        self.test = bool(test)
        self.r7 = Path(r7 or K.R7)
        self.s_folder = Path(s_folder or K.S_FOLDER)
        self.v7_root = Path(v7_root or K.V7)
        self.prereg_v7 = Path(prereg_v7 or (self.v7_root / "prereg_v7.json"))
        self.check_chain = bool(check_chain)
        self.check_v6 = bool(check_v6)

    @classmethod
    def from_test_file(cls, path):
        path = Path(path).resolve()
        fixture = path.parent
        marker = fixture / TEST_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf8") != TEST_MARKER_TEXT:
            raise K.V7Refusal(f"{fixture} is not a v7 test fixture (no {TEST_MARKER} marker)")
        for protected in (K.ROOT, K.R7, K.OUT, K.S_FOLDER):
            if K.inside(fixture, protected):
                raise K.V7Refusal(f"a test fixture may not lie inside {protected}")
        body = json.loads(path.read_text(encoding="utf8"))
        allowed = {"r7", "s_folder", "v7_root", "prereg_v7", "check_chain", "check_v6"}
        if set(body) - allowed:
            raise K.V7Refusal(f"unknown test settings {sorted(set(body) - allowed)}")
        for k in ("r7", "s_folder", "v7_root", "prereg_v7"):
            if k in body and not K.inside(body[k], fixture):
                raise K.V7Refusal(f"test setting {k} must lie inside the fixture {fixture}")
        return cls(test=True, **body)


def split_args(argv):
    argv = list(argv)
    if "--" not in argv:
        raise K.V7Refusal("usage: run_stage2_v7.py --v7-ledger FILE --v7-cpu N [--v7-test-config FILE] -- <arguments>")
    i = argv.index("--")
    own, phase_argv = argv[:i], argv[i + 1:]
    opts = {}
    j = 0
    while j < len(own):
        if own[j] in ("--v7-ledger", "--v7-cpu", "--v7-test-config") and j + 1 < len(own):
            opts[own[j][5:]] = own[j + 1]
            j += 2
        else:
            raise K.V7Refusal(f"unknown wrapper argument {own[j]!r}")
    if "ledger" not in opts or "cpu" not in opts or not opts["cpu"].isdigit():
        raise K.V7Refusal("--v7-ledger and --v7-cpu (one integer) are required")
    return opts, phase_argv


def preflight(cfg, key, cpu):
    """Step 1. Returns (prereg, results, fingerprints). V7Refusal on the first failure."""
    res = {}
    res["environment"] = VF.check_environment()
    res["processor"] = VF.check_one_processor(cpu)
    LG.check_phase_allowed(key)
    prereg = VF.load_prereg_v7(cfg.prereg_v7)
    VF.require_final(prereg)
    if not cfg.test and prereg.payload.get("predecessor") != K.V6:
        raise K.V7Refusal("prereg_v7.json does not name version 6's pins")
    res["v7_code_files"] = len(VF.check_v7_code(prereg, cfg.v7_root).get("files", {}))
    if cfg.check_v6:
        res["v6_pins"] = VF.check_v6_pins()
        res["v6_code_in_scratchpad"] = VF.check_s_folder(cfg.s_folder)
    if cfg.check_chain:
        res["v7_chain"] = VF.verify_v7_chain(prereg, cfg.v7_root)
        res["v6_chain"] = VF.verify_v6_chain()
    handoff = VF.check_handoff(prereg, cfg.v7_root, LG.phase_id(key))
    res["handoff"] = dict(sha256=prereg.payload["handoff"].get("handoff_sha256"),
                          phase_state=(handoff.get("phases") or {}).get(LG.phase_id(key)))
    fingerprints = dict(v7_prereg_sha256=prereg.sha256, v6_prereg_sha256=K.V6["prereg_sha256"],
                        v6_approval_sha256=K.V6["runs_side_approval_sha256"],
                        v7_code_manifest_sha256=(prereg.payload.get("code") or {}).get("code_manifest_v7_sha256"))
    return prereg, res, fingerprints


def constants_location(cfg, prereg):
    block = prereg.payload.get("constants") or {}
    path, pin = K.ROOT / str(block.get("file")), block.get("sha256")
    if not cfg.test and (block.get("file") != K.CONSTANTS_REL or pin != K.CONSTANTS_SHA256):
        raise K.V7Refusal("prereg_v7.json does not pin the constants file V7/world_constants.json 6602f9d1...")
    return path, pin


def handover(cfg, phase_argv, ledger):
    """Step 5: run the pinned run_stage2.py unchanged, as a script, with sys.argv as launch.sh sets it."""
    script = cfg.s_folder / "run_stage2.py"
    while str(HERE) in sys.path:
        sys.path.remove(str(HERE))
    sys.path.insert(0, str(cfg.s_folder))
    sys.argv = [str(script)] + list(phase_argv)
    ledger.event("handover", script=str(script), argv=list(sys.argv), sys_path_head=sys.path[:3])
    code = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        code = error.code if error.code is None or isinstance(error.code, int) else str(error.code)
        raise
    except BaseException as error:
        code = f"exception {type(error).__name__}: {str(error)[:300]}"
        raise
    finally:
        try:
            ledger.event("py-exit", code=code, **LG.usage())
        except Exception:  # noqa: BLE001
            pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        opts, phase_argv = split_args(argv)
        ledger = LG.Ledger(opts["ledger"])
    except K.V7Refusal as error:
        LG.hard_exit(str(error))
    ledger.install_audit_hook()
    ledger.event("wrapper-start", argv=list(argv), boot_id=K.boot_id(), executable=sys.executable,
                 cwd=os.getcwd(), test="test-config" in opts)
    try:
        cfg = Config.from_test_file(opts["test-config"]) if "test-config" in opts else Config()
        key = LG.phase_key(LG.parse_phase_args(phase_argv))
        prereg, res, fingerprints = preflight(cfg, key, int(opts["cpu"]))
        ledger.event("checks", result=res, fingerprints=fingerprints, key=key, test_mode=cfg.test)
        import v7_constants
        path, pin = constants_location(cfg, prereg)
        facts = v7_constants.apply(path, pin)
        ledger.event("constants", facts=facts)
        import v7_refs
        ref = v7_refs.install_for(prereg.payload, key, cfg.s_folder, cfg.r7)
        ledger.event("reference", **ref)
        ledger.event("facts", facts=VF.machine_facts())
    except K.V7Refusal as error:
        ledger.hard_exit(str(error))
    handover(cfg, phase_argv, ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
