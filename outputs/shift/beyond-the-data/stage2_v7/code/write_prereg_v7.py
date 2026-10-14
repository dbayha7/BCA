"""Stage 2, version 7: write V7/prereg_v7.json (+ .sha256) (V7_SPEC.md section 4 and C10).

  write_prereg_v7.py draft [--q7 stop|proceed --q7-answered --q7-quote TEXT --q7-utc YYYY-MM-DDTHH:MM:SSZ]
      The draft: every v7 rule (sections 1, 3, 5, 7 of the spec as data), the predecessor pins (checked against the
      files), the constants pin, the phase table, the re-make and forecast block with the answer to Q7 (default
      'stop', marked open), approvable: false and missing: ['handoff'] (plus any required file the code manifest
      lists as missing). Q7 is marked 'answered by the user' only with --q7-answered, which needs the user's message
      quoted (--q7-quote) and its UTC time (--q7-utc); both are stored in the re-make block, so the probe records bind
      to them. 'proceed' is refused unless it is such a recorded answer; nothing infers an answer.
      It pins code_manifest_v7.json, so write_manifest_v7.py runs first. Once any probe record exists, the re-make and
      forecast block can no longer change (each record binds to its SHA-256, C7 'Probe records').
  write_prereg_v7.py final --handoff V7/handoff/HANDOFF.json [--records DIR]
      The final: the draft with ONLY the handoff fields filled in (approvable, missing, status, handoff), which this
      script enforces against the draft it builds on (kept in prereg_v7_history/). It needs HANDOFF.json (+ sidecar)
      and all nine probe records (<DIR>/selftest/p1_seed{2,3,4}_{stall,no_stall}.json, p2_seed{2,3,4}.json; DIR is R7/
      by default, and V7/handoff/probes/ in the cloud, where cloud/make_handoff.py import-selftest put the records the
      laptop's self-test printed). The reference files are checked against their records where they exist (on the
      laptop); with --records the cloud, which has no reference files, pins them from the records alone (the laptop's
      short self-test, the runner and the wrapper check them against these pins). Every record must bind to the
      draft's re-make block, the constants file, the gate's
      pre-registration and v6's code manifest. It computes C7 step 2's table of re-made references from those pinned
      inputs and HANDOFF.json's inherited phases, the labels, and the check (b) forecast; with a forecast gap above
      1e-5 and Q7 'stop' it refuses (NOT READY). It refuses a HANDOFF.json written for another draft (its
      draft_prereg_v7_sha256), and copies into the handoff block the full self-test the handoff rests on.
Both refuse once an approval or import record of the current prereg_v7.json's fingerprint exists. A replaced version
is copied to prereg_v7_history/prereg_v7_<first 16>.json (+ .sha256) first. Nothing here writes an approval.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402
import v7_probes as PR  # noqa: E402
import v7_refs as RF  # noqa: E402
import v7_verify as VF  # noqa: E402

HANDOFF_FIELDS = ("approvable", "missing", "status", "handoff")
Q7_CHOICES = ("stop", "proceed")

CHANGES = [
    "World constants: A, B and K come from V7/world_constants.json, made on the cloud's processor (C1). Every gate and "
    "v6 check still runs unchanged and exact; a tripwire stops the process if any code computes A or K another way.",
    "Processes at once: up to 16 on the laptop (default 12), instead of version 6's 'at most 2'; none in the cloud after "
    "the handoff (C2, C6).",
    "Interrupted phases: a killed phase's files are moved out of v6's output folder (never deleted) and the phase reruns; "
    "machine-wide interruptions never count; after phase-specific ones it reruns twice, then once alone, then on the "
    "user's --retry, until done (C5). Every other output stays write-once.",
    "Re-made references: on seeds 2-4 only, where the laptop does not reproduce one of the gate's saved reference actors "
    "bit for bit (C7's rule), the laptop makes its own copy with the gate's exact call, and only that seed's and world's "
    "training phase reads it (stalling-world policy test: 'none' and 'full_none'; no-stall policy test: 'none'; "
    "stalling-world full host: 'plain'). Deployment always reads the gate's saved actors.",
    "Check (b) forecast: before anything is approved, the self-test measures whether the laptop's own programs agree as "
    "check (b) requires; a gap above 1e-5 makes it NOT READY unless the user chose to proceed (Q7).",
    "Environment: Ubuntu 24.04 in WSL 2, the OpenBLAS kernel left to its default, an exact eight-variable environment "
    "per phase, one processor per phase checked in code (C3, C4).",
    "Compute budget: version 6's 'about 72-92 process-hours' stays an estimate that stops nothing; the process-hours "
    "actually used are counted and reported (C2).",
    "Records: a ledger of every attempt, provenance files and the probe records (C7, C10, C11).",
]
UNCHANGED = [
    "the method: BCA's band, the uniform and weighted versions, k, every ladder and coefficient, the frozen choices "
    "(weighted version, rival stand-in range_hinge);",
    "the detection test's pair sets, labels, widths, marks and claim rule;",
    "the policy test's arms, matching rules, marks M1-M5 and check (b) with its 1e-5 tolerance and stop rule;",
    "the deployment check (Variant B), tau's rule and its actors (the gate's saved ones for seeds 0-4);",
    "the full host, including its coefficient-0 check;",
    "the confirmation rules: fresh seeds 5-9 alone carry the claim; a failed seed is recorded and never replaced;",
    "the analysis, the pooled looks and the claim sentence;",
    "the processor-only rule, the recorded Python and library versions, the numeric flags, one processor per phase;",
    "the approval route and the refusal of agent identities;",
    "the gate's pre-registration, its files and all its checks, which stay exact to the last bit.",
]
FAILURES = [
    "Check (b) stops from the laptop's arithmetic, on seeds 2-4 and on every confirmatory seed 5-9: on the laptop, check "
    "(b) always compares two different programs. The self-test forecasts it (P1b) and says NOT READY above 1e-5, but a "
    "forecast is not a proof. A stalling-world stop fails every mark M1-M4 of its group.",
    "'Re-make failed' (the laptop's run of the gate's call failed where the gate's did not): check (b) then compares the "
    "laptop's BCA program with a cloud-made actor; a stop there is labelled 'machine-caused (re-make failed)'.",
    "Memory exhaustion inside the stand-in rival: pinned code records any exception there as the rival failing, which "
    "fails M4 for the whole group; guarded by the memory reservations, labelled 'machine-caused: memory'.",
    "Other numerical failures on seeds 5-9 (a learner, a band fit or a fresh data check) are recorded as version 6 says; "
    "whether another machine would have failed the same way cannot be told.",
]


def check_predecessor(root=None):
    VF.check_v6_pins(root)
    return dict(K.V6)


Q7_OPEN = ("open: the user's answer is wanted before the draft is pushed; until it is recorded, the default 'stop' "
           "applies (NOT READY when a check (b) forecast gap is above 1e-5)")
Q7_ANSWERED = "answered by the user"


def q7_message(quote, utc):
    """The user's answer to Q7 as recorded: their words, quoted, and the UTC time of their message."""
    if not (quote or "").strip() or not utc:
        raise K.V7Refusal("--q7-answered needs the user's message quoted (--q7-quote) and its UTC time (--q7-utc)")
    from datetime import datetime
    try:
        datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise K.V7Refusal(f"--q7-utc must be a UTC time in exactly the form 2026-10-14T09:00:00Z (got {utc!r})") from None
    return dict(quote=quote.strip(), utc=utc)


def draft_payload(manifest_sha256, manifest, q7="stop", q7_answered=False, q7_user_message=None):
    if q7 not in Q7_CHOICES:
        raise K.V7Refusal(f"Q7's answer must be one of {Q7_CHOICES}")
    if q7_answered:
        q7_user_message = q7_message((q7_user_message or {}).get("quote"), (q7_user_message or {}).get("utc"))
    elif q7 == "proceed":
        raise K.V7Refusal("Q7 'proceed' can only be the user's recorded answer: give --q7-answered with --q7-quote "
                          "(their words) and --q7-utc (the time of their message)")
    else:
        q7_user_message = None
    missing_files = list(manifest.get("missing_required") or [])
    return dict(
        kind=K.PREREG_KIND, version=7, label=K.LABEL,
        status="DRAFT: the handoff fields are filled in after the handoff; a draft cannot be approved",
        governing=("Version 6 (prereg.json " + K.V6["prereg_sha256"] + ") stays the governing document for everything "
                   "this file does not list as changed. Each v7 phase needs both v6's approval and v7's."),
        predecessor=check_predecessor(),
        code=dict(code_manifest_v7_sha256=manifest_sha256, files=manifest["files"]),
        changes=CHANGES, unchanged=UNCHANGED, failures_v7_can_cause=FAILURES,
        constants=dict(file=K.CONSTANTS_REL, sha256=K.CONSTANTS_SHA256,
                       tolerances=dict(A=K.TOL_A, K=K.TOL_K, B="bit-identical"),
                       functions=["experiments.shift.thruster_chain._matrices", "experiments.shift.recorder._gain"],
                       tripwires=["experiments.signal.gapped_lq.transition_matrix", "scipy.linalg.solve_discrete_are"],
                       exit_code=K.EXIT_V7,
                       acceptance=("A processor is accepted if and only if the file's SHA-256 matches, the local A and "
                                   "K are within the tolerances with B bit-identical, and the unchanged gate checks then "
                                   "pass exactly (load_prereg and guard accept all 64 anchors to the last bit; for seeds "
                                   "0-4 load_seed matches every log). v7 adds no tolerance to any gate check."),
                       departure="declared operational departure, accepted by the user (Q1)"),
        lanes=dict(default=12, max=16, machine="the user's laptop only", cloud_after_handoff=0, wsl_memory_gb=48,
                   rule="the lane count may be changed (1-16) on any restart; it changes no number"),
        memory=dict(unmeasured_kind_reservation_gib=7, measured_factor=1.25,
                    policy_kinds=["policy_train", "policy_no_stall_train"],
                    policy_kinds_reservation="the larger of 7 GiB and twice the largest VmHWM measured on the laptop",
                    headroom_gib=8, policy_train_at_once_until_measured=2,
                    rule=("a phase starts only if the reserved peaks of the running phases plus its own are at most "
                          "WSL's MemTotal minus 8 GiB, and MemAvailable is at least its own reservation plus 8 GiB")),
        standin_errors=("when a new policy lock records standin.error or standin.no_stall_error, no new policy_train or "
                        "policy_no_stall_train starts until the user restarts the runner with --reviewed <phase>; other "
                        "phases continue; a memory error is labelled 'machine-caused: memory'"),
        budget=dict(v6_estimate="about 72-92 process-hours", rule="an estimate that stops nothing; hours used are reported"),
        one_processor=("launch_v7.sh refuses unless the processor argument is one integer, and the wrapper refuses unless "
                       "os.sched_getaffinity(0) has exactly one element"),
        environment=dict(variables=K.PHASE_ENV, working_folder=str(K.ROOT), python=K.PYTHON,
                         system="Ubuntu 24.04 in WSL 2, as root, HOME=/root (the user's decision, Q2)",
                         openblas_coretype="unset", uv="0.11.32",
                         lock_file="outputs/shift/beyond-the-data/stage2_gpu/requirements-lock.txt"),
        phase_table=LG.phase_table_text(),
        interruptions=dict(
            done="a phase is done when its final record or a failed-on-record file is present and verified",
            machine_wide=["the boot id changed", "the runner sent the signal", "SIGHUP, SIGINT or SIGTERM",
                          "no exit status recorded"],
            phase_specific=["SIGKILL, SIGSEGV, SIGBUS or SIGABRT on the same boot, not sent by the runner",
                            "the log ends in a memory error (MemoryError, RESOURCE_EXHAUSTED, 'out of memory')"],
            reruns="machine-wide: without limit; phase-specific: twice automatically, then once alone, then unfinished",
            unfinished="retried until done; the user's --retry sets only when the next attempt (run alone) starts",
            files="every existing file of the phase's table row is moved (never deleted) to R7/interrupted/",
            exit_97="a v7 refusal halts the run", other="any other non-zero exit, or exit 0 not done, halts new phases"),
        handoff_rule=dict(stop_rule_default="stop at once", alternative="let them finish",
                          chosen="before the full self-test (step 8); it cannot change after any P1 or P2 record exists",
                          inherited=("a phase is inherited when its final record or failed-on-record file is in the "
                                     "final snapshot with a matching sidecar; inherited phases never run on the laptop"),
                          way_back=("if v7 is not approved or the laptop refuses before its first phase starts, the "
                                    "cloud writes V7/handoff/WITHDRAWN.json and may resume v6; once a v7 phase has "
                                    "started, the cloud never resumes Stage 2")),
        remake=dict(
            seeds=list(K.REMAKE_SEEDS), forecast_tolerance=K.W0_TOL,
            q7_if_forecast_above_tolerance=q7, q7_default="stop", q7_status=Q7_ANSWERED if q7_answered else Q7_OPEN,
            q7_user_message=q7_user_message,
            kinds=[dict(phase=p, world=w, method=v["method"], lock=v["lock"], lock_names=v["lock_names"],
                        reference_file="refs/seed{s}/" + v["file"]) for (p, w), v in sorted(RF.KINDS.items())],
            p1a_names={w: [n for n, _ in PR.P1A_ARMS[w]] for w in PR.WORLDS},
            rule=("An entry (phase, seed, world) is made only for a seed 2-4 phase that is not inherited: "
                  "(policy_train, s, stall) when P1a differs for 'gapped' or 'full' and P1b does not reproduce the gate's "
                  "'gapped' bit for bit; (policy_no_stall_train, s, no_stall) likewise for 'no_stall_gapped'; "
                  "(host_train, s, stall) when P2's final actor differs. When P1b reproduces the gate's actor, the "
                  "gate's file is kept. A laptop run that failed where the gate's did not makes no entry ('re-make "
                  "failed'). No other entry is allowed."),
            forecast=("F_ref (P1b's W = 0 actor against the actor check (b) would read) and F_fresh (P1b against P1c's "
                      "'none') for every seed 2-4 and world; any gap above 1e-5 is NOT READY unless Q7 says proceed"),
            classed_as=("an operational departure: the gate's exact call on the gate's inputs, every setting unchanged; "
                        "the rule picks no seed or world by any outcome of BCA (the user's decision, Q4)"),
            labels=["inherited (cloud, v6)", "laptop, gate's reference (v6 as written)",
                    "laptop, gate's reference kept (P1b reproduced it)", "laptop, reference re-made on the laptop",
                    "laptop, re-make failed", "laptop, own run"]),
        gpu="processor only, as version 6; the GPU may be timed for information in a separate environment no phase uses",
        approval=dict(accepted_modes=[K.LOCAL_MODE], local_terminal_allowed=True, branch=K.BRANCH,
                      approver_email=K.APPROVER_EMAIL,
                      refused_emails=["noreply@anthropic.com", "the cloud session's configured user.email"],
                      approval_file=K.V7_REL + "/approval_v7_<first 16 characters>.json",
                      import_record=K.V7_REL + "/import_v7_<first 16 characters>.json",
                      rule=("the user runs approve_local_v7.py at their own terminal (it refuses unless standard input "
                            "is a terminal), commits and pushes the file under their own name; approve_v7.py import "
                            "requires exactly one adding commit whose author and committer are v6's approver and "
                            "neither the cloud session's identity nor its configured user.email; every v7 phase "
                            "re-checks this chain with read-only git. Git does not authenticate authors.")),
        approvable=False, missing=["handoff"] + [f"file: {r}" for r in missing_files], handoff=None)


def non_handoff(payload):
    return {k: v for k, v in payload.items() if k not in HANDOFF_FIELDS}


def frozen(v7_root, prereg):
    return any((Path(v7_root) / n.format(prefix=prereg.prefix)).exists() for n in (K.APPROVAL_NAME, K.IMPORT_NAME))


def _archive(v7_root, prereg):
    hist = Path(v7_root) / "prereg_v7_history"
    hist.mkdir(parents=True, exist_ok=True)
    dst = hist / f"prereg_v7_{prereg.prefix}.json"
    if not dst.exists():
        shutil.copyfile(prereg.path, dst)
        shutil.copyfile(str(prereg.path) + ".sha256", str(dst) + ".sha256")
    return dst


def _write(v7_root, payload):
    path = Path(v7_root) / "prereg_v7.json"
    text = K.canonical(payload).encode("utf8")
    digest = K.sha256_bytes(text)
    path.write_bytes(text)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def _manifest(v7_root):
    path = Path(v7_root) / "code_manifest_v7.json"
    man = K.read_json_verified(path, "code_manifest_v7.json")
    return K.file_sha256(path), man


def _existing(v7_root):
    path = Path(v7_root) / "prereg_v7.json"
    return VF.load_prereg_v7(path) if path.exists() else None


def _records_bound(r7):
    out = {}
    for pid in PR.all_probe_ids():
        state, rec = K.read_record(PR.record_path(r7, pid))
        if state == "ok":
            out[pid] = (rec.get("inputs") or {}).get("remake_block_sha256")
    return out


def q7_answered_in(payload):
    return (payload.get("remake") or {}).get("q7_status") == Q7_ANSWERED


def q7_args(payload):
    """The Q7 arguments a draft was written with (so the final can rebuild it and compare)."""
    r = payload.get("remake") or {}
    return r.get("q7_if_forecast_above_tolerance"), q7_answered_in(payload), r.get("q7_user_message")


def write_draft(v7_root=None, r7=None, q7="stop", log=print, q7_answered=False, q7_quote=None, q7_utc=None):
    v7_root, r7 = Path(v7_root or K.V7), Path(r7 or K.R7)
    old = _existing(v7_root)
    if old is not None and frozen(v7_root, old):
        raise K.V7Refusal("an approval or import record of the current prereg_v7.json exists; it is frozen")
    msha, man = _manifest(v7_root)
    body = draft_payload(msha, man, q7, q7_answered, dict(quote=q7_quote, utc=q7_utc) if q7_answered else None)
    block = K.block_sha256(body["remake"])
    bound = {pid: b for pid, b in _records_bound(r7).items() if b != block}
    if bound:
        raise K.V7Refusal(f"probe records exist that bind to another re-make block ({sorted(bound)}); the re-make and "
                          f"forecast block (and Q7's answer) can no longer change")
    if old is not None:
        _archive(v7_root, old)
    digest = _write(v7_root, body)
    log(f"wrote {v7_root / 'prereg_v7.json'} (DRAFT) SHA-256 {digest}; re-make block {block[:16]}...; "
        f"missing {body['missing']}")
    return body, digest


def build_final(draft, draft_sha256, handoff_path, r7, require_files=True):
    """The final payload from a draft, HANDOFF.json and the nine probe records (C7 step 2, step 2b). With
    require_files=False (the cloud), a reference file that is absent is pinned from its record alone; one that is
    present must still match."""
    handoff = K.read_json_verified(handoff_path, "HANDOFF.json")
    if handoff.get("kind") != "stage2-v7-handoff":
        raise K.V7Refusal("HANDOFF.json is not a v7 handoff record")
    if handoff.get("draft_prereg_v7_sha256") != draft_sha256:
        raise K.V7Refusal(f"HANDOFF.json was written for the draft {str(handoff.get('draft_prereg_v7_sha256'))[:16]}..., "
                          f"not this draft {draft_sha256[:16]}...; a new draft needs a new handoff (or WITHDRAWN.json)")
    st = handoff.get("selftest") or {}
    if not (st.get("form") == "full" and str(st.get("verdict") or "").startswith("READY") and
            (st.get("p3") or {}).get("ok") is True):
        raise K.V7Refusal("HANDOFF.json does not rest on a READY full self-test whose repeatability test P3 passed "
                          "(C3, C11)")
    records, shas = PR.read_probe_records(r7)
    block = K.block_sha256(draft["remake"])
    for pid, rec in records.items():
        inp = rec.get("inputs") or {}
        want = dict(remake_block_sha256=block, constants_sha256=draft["constants"]["sha256"],
                    gate_prereg_sha256=K.GATE_PREREG_SHA256, v6_code_manifest_sha256=K.V6["code_manifest_sha256"],
                    probe_script_sha256=(draft["code"]["files"] or {}).get("code/v7_probes.py"))
        bad = sorted(k for k, v in want.items() if inp.get(k) != v)
        if bad:
            raise K.V7Refusal(f"probe record {pid} does not bind to this draft's inputs ({bad})")
    pins = {}
    for pid, rec in records.items():
        files = ([rec["p1a"]["reference_file"]] if rec["kind"] == "stage2-v7-probe-p1"
                 else list(rec["snapshots"].values()))
        pins[pid] = {}
        for f in files:
            if f.get("sha256") is None:
                continue
            p = Path(r7) / f["path"]
            if (require_files or p.exists()) and (not p.is_file() or K.file_sha256(p) != f["sha256"]):
                raise K.V7Refusal(f"HALT: reference {f['path']} differs from probe record {pid}")
            pins[pid][f["path"]] = f["sha256"]
    inherited = {pid for pid, state in (handoff.get("phases") or {}).items() if state == "inherited"}
    entries, labels = PR.remake_table(records, shas, inherited)
    for e in entries:
        RF.validate_entry(e)
    fc = PR.forecast_all(records, draft["remake"]["q7_if_forecast_above_tolerance"])
    if fc["not_ready"]:
        raise K.V7Refusal(fc["line"])
    probe_rows = {}
    for pid, rec in records.items():
        row = dict(path=PR.record_path(Path("."), pid).as_posix(), sha256=shas[pid], result=PR.result_line(rec),
                   file_pins=pins[pid])
        if rec["kind"] == "stage2-v7-probe-p1":
            row.update(differs={k: v["differs"] for k, v in rec["p1a"]["actors"].items()},
                       F_ref=rec["forecast"]["F_ref"], F_fresh=rec["forecast"]["F_fresh"],
                       written_utc=rec.get("written_utc"))
        else:
            row.update(differs=dict(actor=rec["actor"]["differs"]), F_ref=None, F_fresh=None,
                       written_utc=rec.get("written_utc"))
        probe_rows[pid] = row
    summary = [f"snapshot {(handoff.get('snapshot') or {}).get('commit')}",
               f"stop rule: {handoff.get('stop_rule')} (chosen {handoff.get('stop_rule_chosen_utc')})",
               f"inherited phases: {len(inherited)}",
               f"full self-test: {st.get('verdict')}; repeatability test P3 passed",
               f"Q7 (proceed although the check (b) forecast is above 1e-5?): "
               f"'{draft['remake'].get('q7_if_forecast_above_tolerance')}', {draft['remake'].get('q7_status')}",
               f"check (b) forecast: {fc['line']}"]
    payload = dict(draft)
    payload.update(
        approvable=True, missing=[],
        status="FINAL: identical to the draft except the handoff fields (approvable, missing, status, handoff)",
        handoff=dict(handoff_file=K.V7_REL + "/handoff/HANDOFF.json", handoff_sha256=K.file_sha256(handoff_path),
                     draft_sha256=draft_sha256, snapshot=handoff.get("snapshot"), stop_rule=handoff.get("stop_rule"),
                     inherited=sorted(inherited), summary=summary, probe_records=probe_rows, forecast=fc,
                     remake_table=entries, labels=labels, selftest=handoff.get("selftest")))
    if non_handoff(payload) != non_handoff(draft):
        raise K.V7Refusal("the final differs from its draft outside the handoff fields")
    return payload


def write_final(handoff_path, v7_root=None, r7=None, log=print, require_files=True):
    v7_root, r7 = Path(v7_root or K.V7), Path(r7 or K.R7)
    old = _existing(v7_root)
    if old is None:
        raise K.V7Refusal("no draft prereg_v7.json: write the draft first")
    if frozen(v7_root, old):
        raise K.V7Refusal("an approval or import record of the current prereg_v7.json exists; it is frozen")
    if old.final:
        hist = v7_root / "prereg_v7_history" / f"prereg_v7_{K.prefix_of(old.payload['handoff']['draft_sha256'])}.json"
        base = K.read_json_verified(hist, "the archived draft")
        base_sha = K.file_sha256(hist)
        if base_sha != old.payload["handoff"]["draft_sha256"]:
            raise K.V7Refusal("the archived draft is not the one the current final names")
    else:
        base, base_sha = old.payload, old.sha256
    msha, man = _manifest(v7_root)
    if man.get("missing_required"):
        raise K.V7Refusal(f"required v7 files are missing: {man['missing_required']}")
    fresh = draft_payload(msha, man, *q7_args(base))
    if non_handoff(fresh) != non_handoff(dict(base, missing=None)):
        raise K.V7Refusal("the files changed since the draft (its code manifest or rules); write a new draft first")
    payload = build_final(base, base_sha, handoff_path, r7, require_files)
    _archive(v7_root, old)
    digest = _write(v7_root, payload)
    log(f"wrote {v7_root / 'prereg_v7.json'} (FINAL) SHA-256 {digest}\n  first 16 characters: {digest[:16]}\n"
        f"  re-made references: {len(payload['handoff']['remake_table'])}; forecast: {payload['handoff']['forecast']['line']}")
    return payload, digest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--q7", choices=Q7_CHOICES, default="stop")
    d.add_argument("--q7-answered", action="store_true", help="the user has answered Q7 (recorded in the draft)")
    d.add_argument("--q7-quote", help="the user's answer to Q7, quoted from their message")
    d.add_argument("--q7-utc", help="the UTC time of that message, e.g. 2026-10-14T09:00:00Z")
    f = sub.add_parser("final")
    f.add_argument("--handoff", required=True)
    f.add_argument("--records", help="folder holding selftest/<probe>.json (the cloud: V7/handoff/probes)")
    a = ap.parse_args(argv)
    try:
        if a.mode == "draft":
            write_draft(q7=a.q7, q7_answered=a.q7_answered, q7_quote=a.q7_quote, q7_utc=a.q7_utc)
        else:
            write_final(Path(a.handoff), r7=a.records, require_files=a.records is None)
    except K.V7Refusal as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
