"""Version 4's S7 supplement (PRESPEC.md section 21, post hoc): every stage-2 run of every scope version 3's S7 check
stopped is trained and scored in full under its own root, and shown beside version 3's results, never judged.

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. At module level
it imports the standard library, gate and handoff only, so the stage driver (training environment) imports it; claims,
export and budget are imported inside the functions that use them (the export runs in the scoring environment).

The frozen block, prespec.json "supplement" (validate; nothing here or in the driver changes it):
  version, root   4 and runs/shift_hosts/supplement_v4 (ROOT): the supplement's own driver/ (state, status, log and its
                  own ledger under section 12's cap numbers), stage1/ and stage2/; version 3's folders are never written
  source          version 3's fingerprint and approval commit, and the SHA-256 of its driver/state.json,
                  driver/ledger.json and results.json (outputs/shift_hosts_v3/results.json)
  scopes          what version 3's S7 stopped: [{quality, learner, scope "learner" | "iw"}], as claims.s7_scopes reads
                  version 3's state.json "s7_stopped" (check_sources recomputes it)
  plan            {quality: {learner: [setups]}} (plan_of): a learner scope trains sham_alone (choice-19 learners only),
                  alone, standard, IW at seeds 0-4 and the room references; an IW scope trains IW at seeds 0-4 and that
                  learner's alone and standard at seeds 0-4 as same-fingerprint comparators. Stage 1 runs whole.
  s7              S7 lanes on seeds 0-4 and on the 50,000- and 100,000-step actors of every planned alone, standard and
                  IW run (the owner's decision D3); a failed check is recorded and drops nothing
On real data the stage driver takes its plan from validate() (stages.frozen_config) and, before it starts, requires
check_sources() (version 3's files against the block) and an empty code_unchanged() (every file but CHANGEABLE
byte-identical to version 3's approved code manifest, so claims.py, export.py and every training and scoring file are
version 3's).

The export (scoring environment, after version 4's terminal approval):
  python -m experiments.shift_hosts.supplement export [--out outputs/shift_hosts/results.json]
writes version 3's results.json, carried byte for byte (read_carried, against the block's SHA-256), plus only
qualities.<quality>.s7_shown, the top-level s7_supplement and one appended caveat (caveat_of: CAVEAT, and a sentence
naming every S7 failure on an actor of a scope version 3 judged, such as an IW scope's comparators). It refuses to
write unless every version-3 key is canonical-JSON equal (assert_judged_unchanged), so no judged badge, p-value, Holm
step, mark or bar can move. The shown numbers are per-seed values with their mean and sd; a paired difference also
carries its standard error and one-sided 95% bounds (the owner's decision D5, PRESPEC.md section 21.4); never a t, a
p-value, a Holm entry, a bar, a badge or a verdict. R1 (reproduction) compares every supplement record with version
3's record of the same run id where one exists; it is reported and never stops anything.

Version 3's folders are read, never written: before a real start check_sources reads its driver/state.json and
driver/ledger.json against the block's SHA-256; the export carries its committed results.json (SHA-256 checked) and
reads that ledger for the study-wide total; R1 reads its run records, actor manifests and S7 files for its own
comparison only (s7_supplement.reproduction and each shown run's r1), which feeds no shown score.
"""

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.shift_hosts import gate
from experiments.shift_hosts import handoff as H

VERSION = 4
ROOT_REL = "runs/shift_hosts/supplement_v4"
ROOT = gate.REPO / ROOT_REL
V3_ROOT = gate.REPO / "runs" / "shift_hosts"                # version 3's root (read only, for R1 and its ledger)
LABEL = "S7 failed: trained and scored in full, shown, not judged"
CAVEAT = ("Post hoc (pre-specification version 4, PRESPEC.md section 21): every stage-2 run of every scope version 3's "
          "S7 check stopped was trained and scored in full under version 4, in its own root "
          "(runs/shift_hosts/supplement_v4), and is shown under qualities.<quality>.s7_shown and s7_supplement, never "
          "judged: version 3's results above are carried byte for byte, so no claim, badge, p-value, Holm step, mark "
          "or bar changed. The shown scores are float64-adapter returns of those scopes' actors: S7's float32 check "
          "failed on at least one actor of each scope (seed 0 or stage 1), and ran on seeds 1-4 and on the 50,000-step "
          "actors only in the supplement, so a shown score may differ from the learner's own float32 policy by more "
          "than 0.05 points per start.")
V3 = dict(version=3, fingerprint="fe9fb77b0851c75b522c337fcb65c2a3d027de71ddfa50dcd6f3affc3bbd1c1c",
          approval_file="approval_fe9fb77b0851c75b.json", freeze_commit="bafd86f3e64324b7f5c93a633b1d458f717af774",
          approval_commit="6f9ce2b51bf2907328477f4ffa96afff25522559")
# version 3's files the block names by SHA-256 (repository-relative)
SOURCE_FILES = dict(state_sha256="runs/shift_hosts/driver/state.json",
                    ledger_sha256="runs/shift_hosts/driver/ledger.json",
                    results_sha256="outputs/shift_hosts_v3/results.json")
RESULTS_V3_PREVIEW = "runs/shift_hosts/results_v3.json"   # version 3's export as built (git-ignored; checked if there)
# the files version 4 may change against version 3's approved code manifest (supplement.py is the only one added)
CHANGEABLE = ("experiments/shift_hosts/stages.py", "experiments/shift_hosts/supplement.py",
              "experiments/shift_hosts/PRESPEC.md", "experiments/shift_hosts/INTERFACES.md",
              "experiments/shift_hosts/tests/test_stages.py", "experiments/shift_hosts/tests/test_claims.py")
ADDED = ("experiments/shift_hosts/supplement.py",)
QUALITIES = ("medium-expert", "medium")                     # claims.QUALITIES
LEARNERS = ("td3_bc", "cql", "iql", "rebrac")               # claims.LEARNERS
PLAN_SETUPS = ("sham_alone", "alone", "standard", "iw", "room")
S7_SETUPS = ("alone", "standard", "iw")                     # stages.STAGE2_SETUPS
SCOPE_SETUPS = dict(learner=("alone", "standard", "iw", "room"), iw=("alone", "standard", "iw"))
FINAL_STEP, HALF_STEP = 100_000, 50_000                     # runner.NUM_UPDATES and the 50% snapshot
BLOCK_KEYS = ("plan", "root", "s7", "scopes", "source", "version")
SOURCE_KEYS = ("approval_commit", "fingerprint", "ledger_sha256", "results_sha256", "state_sha256")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# what a shown summary keeps of claims.summary: for a setup the per-seed values with their mean and sd; for a paired
# difference also its se and one-sided 95% bounds (mean -+ claims.T95 x se), the owner's decision D5 against the
# committed design's "no bound"; never t or p_one_sided
SETUP_FIELDS = ("per_seed", "n", "nonfinite", "missing", "complete", "mean", "sd")
PAIRED_FIELDS = SETUP_FIELDS + ("se", "lower", "upper")
# the shown paired differences: a learner scope's, and an IW scope's (whose standard - alone is version 3's judged
# Claim 1 and is not repeated)
PAIRS = dict(learner=(("standard", "alone"), ("iw", "standard"), ("iw", "alone"), ("room", "alone"),
                      ("sham_alone", "alone")),
             iw=(("iw", "standard"), ("iw", "alone")))
PAIRED_BANKS = ("none", "mild", "strong")                   # claims.WEIGHTING_BANKS
# versions 1 and 2's spend (PRESPEC.md section 20.9: medium-expert stage 1). Their sessions are inside version 3's
# driver/ledger.json, whose totals sum every session in it, so this is shown as an "of which" and never added again
EARLIER_SPEND = {"medium-expert": dict(card_hours=0.4939, processor_hours=0.7101,
                                       source="PRESPEC.md section 20.9: versions 1 and 2's stage 1; of which, already "
                                              "counted in version 3's ledger (not added again)")}


def canonical(obj):
    """The canonical JSON text of obj (handoff.plain first, sorted keys, no spaces): what assert_judged_unchanged
    compares."""
    return json.dumps(H.plain(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_of(obj):
    return hashlib.sha256(canonical(obj).encode("utf8")).hexdigest()


def _refuse(why):
    raise gate.ApprovalError(f"refused: {why}")


# ------------------------------------------------------------------------------------------------ scopes and plan

def scopes_of(entries, qualities=QUALITIES):
    """{quality: {learner: "learner" | "iw"}} from a driver's state.json "s7_stopped" entries, as claims.s7_scopes reads
    them (a learner scope wins over an IW scope). Qualities without a stopped scope map to {}."""
    from experiments.shift_hosts import claims as C

    return {q: {L: sc["scope"] for L, sc in C.s7_scopes(entries, q).items()} for q in qualities}


def scope_map(items, learners=LEARNERS, qualities=QUALITIES):
    """The block's "scopes" list [{quality, learner, scope}] -> {quality: {learner: scope}} (every quality present,
    {} when nothing stopped there). Refuses a malformed list (gate.ApprovalError)."""
    if not isinstance(items, list) or not items:
        _refuse("the supplement block's scopes are not a non-empty list")
    out = {q: {} for q in qualities}
    for e in items:
        if not isinstance(e, dict) or sorted(e) != ["learner", "quality", "scope"]:
            _refuse(f"a supplement scope is not {{quality, learner, scope}}: {e!r}")
        q, L, sc = e["quality"], e["learner"], e["scope"]
        if q not in qualities or L not in learners or sc not in SCOPE_SETUPS:
            _refuse(f"a supplement scope names an unknown quality, learner or scope: {e!r}")
        if L in out[q]:
            _refuse(f"the supplement block names {L} on {q} twice")
        out[q][L] = sc
    return out


def plan_of(scopes, choice19, learners=LEARNERS):
    """The run plan {quality: {learner: [setups]}} of the scopes {quality: {learner: scope}} (module docstring): a
    learner scope trains sham_alone (choice 19 only), alone, standard, IW and room; an IW scope trains alone, standard
    and IW. Qualities without a scope are left out."""
    out = {}
    for q, by in scopes.items():
        plan_q = {}
        for L in learners:
            sc = by.get(L)
            if sc is None:
                continue
            plan_q[L] = (["sham_alone"] if sc == "learner" and choice19.get(L) else []) + list(SCOPE_SETUPS[sc])
        if plan_q:
            out[q] = plan_q
    return out


def check_plan(plan, learners=LEARNERS, qualities=QUALITIES, choice19=None):
    """A plan {quality: {learner: [setups]}} as the stage driver uses it (prespec.json's, or a dry run's
    --supplement-plan file): known qualities, learners and setups, each setup once, sham_alone only for a choice-19
    learner, room only beside alone (the room check reads the alone run of its seed). Returns a plain copy; refuses
    (gate.ApprovalError) otherwise."""
    if not isinstance(plan, dict):
        _refuse(f"the supplement plan is not {{quality: {{learner: [setups]}}}}: {plan!r}")
    out = {}
    for q, by in plan.items():
        if q not in qualities or not isinstance(by, dict):
            _refuse(f"the supplement plan names an unknown quality or is malformed there: {q!r}")
        out[q] = {}
        for L, setups in by.items():
            if L not in learners or not isinstance(setups, list) or not setups:
                _refuse(f"the supplement plan's {q} names an unknown learner or no setups: {L!r}")
            if len(set(setups)) != len(setups) or not set(setups) <= set(PLAN_SETUPS):
                _refuse(f"the supplement plan's {q} {L} setups are not distinct setups of {PLAN_SETUPS}: {setups}")
            if "sham_alone" in setups and not (choice19 or {}).get(L):
                _refuse(f"the supplement plan's {q} {L} has a sham-alone run, but choice 19 does not apply to {L}")
            if "room" in setups and "alone" not in setups:
                _refuse(f"the supplement plan's {q} {L} has room references without its alone runs")
            out[q][L] = [str(s) for s in setups]
    return out


def run_ids(plan, seeds, learners=LEARNERS, made_up=False):
    """{quality: [run ids]} of a plan's stage-2 runs, in the stage driver's order (stages.Driver.plan_units and
    unit_jobs): the sham-alone runs, each seed's runs (learner order, then alone, standard, IW), then the room
    references (seed, then learner)."""
    out = {}
    for q, by in plan.items():
        label = f"made-up-{q}" if made_up else q
        rid = lambda k, L, s: f"stage2/{label}/seed{int(k)}/{L}-{s}"
        ids = [rid(0, L, "sham_alone") for L in learners if "sham_alone" in by.get(L, ())]
        ids += [rid(k, L, s) for k in seeds for L in learners for s in S7_SETUPS if s in by.get(L, ())]
        ids += [rid(k, L, "room") for k in seeds for L in learners if "room" in by.get(L, ())]
        out[q] = ids
    return out


def validate(block, learners=LEARNERS, qualities=QUALITIES, seeds=(0, 1, 2, 3, 4), choice19=None):
    """prespec.json's "supplement" block (module docstring) against this code -> dict(plan, scopes, s7_all_seeds,
    s7_half): its keys exactly, version 4, the root, version 3's fingerprint and approval commit, every source SHA-256,
    scopes that give exactly its plan (plan_of), and S7's settings (seeds [0] or every data seed, steps [100,000] or
    [50,000, 100,000], the alone, standard and IW setups). Refuses (gate.ApprovalError) anything else."""
    if not isinstance(block, dict) or tuple(sorted(block)) != BLOCK_KEYS:
        _refuse(f"prespec.json's supplement block does not have exactly the keys {BLOCK_KEYS}")
    if block["version"] != VERSION or block["root"] != ROOT_REL:
        _refuse(f"the supplement block is not version {VERSION} under {ROOT_REL} ({block['version']!r}, "
                f"{block['root']!r})")
    src = block["source"]
    if not isinstance(src, dict) or tuple(sorted(src)) != SOURCE_KEYS:
        _refuse(f"the supplement block's source does not have exactly the keys {SOURCE_KEYS}")
    if src["fingerprint"] != V3["fingerprint"] or src["approval_commit"] != V3["approval_commit"]:
        _refuse("the supplement block's source is not version 3's fingerprint and approval commit")
    for k in ("state_sha256", "ledger_sha256", "results_sha256"):
        if not isinstance(src[k], str) or not HEX64.match(src[k]):
            _refuse(f"the supplement block's source.{k} is not a SHA-256")
    scopes = scope_map(block["scopes"], learners, qualities)
    plan = check_plan(block["plan"], learners, qualities, choice19)
    want = plan_of(scopes, choice19 or {}, learners)
    if canonical(plan) != canonical(want):
        _refuse(f"the supplement block's plan is not what its scopes give: {json.dumps(plan, sort_keys=True)} against "
                f"{json.dumps(want, sort_keys=True)}")
    s7 = block["s7"]
    if not isinstance(s7, dict) or sorted(s7) != ["seeds", "setups", "steps"]:
        _refuse("the supplement block's s7 is not {seeds, steps, setups}")
    if s7["seeds"] not in ([0], sorted(int(k) for k in seeds)) or 0 not in seeds:
        _refuse(f"the supplement block's s7.seeds are neither [0] nor every data seed: {s7['seeds']!r}")
    if s7["steps"] not in ([FINAL_STEP], [HALF_STEP, FINAL_STEP]):
        _refuse(f"the supplement block's s7.steps are neither [{FINAL_STEP}] nor [{HALF_STEP}, {FINAL_STEP}]")
    if s7["setups"] != list(S7_SETUPS):
        _refuse(f"the supplement block's s7.setups are not {list(S7_SETUPS)}")
    return dict(plan=plan, scopes=scopes, s7_all_seeds=s7["seeds"] != [0], s7_half=HALF_STEP in s7["steps"])


def prespec_block(package=gate.PACKAGE):
    """prespec.json's "supplement" block (refused when missing)."""
    path = Path(package) / gate.PRESPEC_NAME
    try:
        return json.loads(path.read_text(encoding="utf8"))["supplement"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        _refuse(f"{path} has no supplement block ({error!r})")


def provenance(block, plan):
    """What the driver's state.json records of the frozen block it runs ("supplement"): the plan's canonical SHA-256,
    version 3's identity and the source SHA-256 values."""
    return dict(version=VERSION, root=block["root"], plan_sha256=sha256_of(plan), v3=dict(V3),
                source=dict(block["source"]))


# ------------------------------------------------------------------------------------------------ before a real start

def code_unchanged(manifest_now, package=gate.PACKAGE):
    """Every difference between the code manifest now (gate.code_manifest()) and version 3's approved one
    (approval_fe9fb77b0851c75b.json, checked against its sidecar) outside CHANGEABLE: [{path, change: "changed" |
    "added" | "removed"}]. supplement.py is the only file that may be added; nothing may be removed. Empty means every
    other file, external code included, is byte-identical to version 3's."""
    path = Path(package) / V3["approval_file"]
    side = gate.sidecar(path)
    words = side.read_text(encoding="utf8").split() if side.is_file() else []
    if not path.is_file() or not words or words[0] != gate.sha256_file(path):
        _refuse(f"{path.name} (version 3's approval) is missing or differs from its sidecar")
    body = json.loads(path.read_text(encoding="utf8"))
    if body.get("fingerprint") != V3["fingerprint"] or not isinstance(body.get("code_manifest"), dict):
        _refuse(f"{path.name} is not version 3's approval")
    then = body["code_manifest"]
    out = []
    for p in sorted(set(then) | set(manifest_now)):
        if p in then and p in manifest_now:
            if then[p] != manifest_now[p] and p not in CHANGEABLE:
                out.append(dict(path=p, change="changed"))
        elif p in manifest_now:
            if p not in ADDED:
                out.append(dict(path=p, change="added"))
        else:
            out.append(dict(path=p, change="removed"))
    return out


def check_sources(block, repo=gate.REPO, package=gate.PACKAGE):
    """Version 3's files against the frozen block before a real start (stages.preflight): driver/state.json,
    driver/ledger.json and the committed results.json by SHA-256 (runs/shift_hosts/results_v3.json too when it exists),
    the committed results.json and version 3's approval file with its sidecar in git HEAD as on disk, the approval's
    fingerprint and the commit that last touched it, and the scopes recomputed from state.json's "s7_stopped" against
    the block's. Returns what was checked; refuses (gate.ApprovalError) on any difference."""
    repo, src = Path(repo), block["source"]
    checked = {}
    for key, rel in SOURCE_FILES.items():
        p = repo / rel
        got = gate.sha256_file(p) if p.is_file() else None
        if got != src[key]:
            _refuse(f"version 3's {rel} is {'missing' if got is None else 'not the file the block names'} "
                    f"(source.{key})")
        checked[rel] = got
    preview = repo / RESULTS_V3_PREVIEW
    if preview.is_file():
        if gate.sha256_file(preview) != src["results_sha256"]:
            _refuse(f"{RESULTS_V3_PREVIEW} differs from version 3's committed results.json")
        checked[RESULTS_V3_PREVIEW] = src["results_sha256"]
    approval = Path(package) / V3["approval_file"]
    for f in (repo / SOURCE_FILES["results_sha256"], approval, gate.sidecar(approval)):
        if not f.is_file() or not gate.committed_in_head(f, repo):
            _refuse(f"{f.name} is not committed in git HEAD as it is on disk")
    try:
        fp = json.loads(approval.read_text(encoding="utf8")).get("fingerprint")
    except ValueError:
        fp = None
    if fp != src["fingerprint"]:
        _refuse(f"{approval.name} does not approve version 3's fingerprint")
    last = gate.last_commit(approval, repo)
    if last is None or last[0] != src["approval_commit"]:
        _refuse(f"{approval.name} was last committed by {None if last is None else last[0]}, not version 3's approval "
                f"commit {src['approval_commit']}")
    try:
        state = H.read_json(repo / SOURCE_FILES["state_sha256"])
    except (H.HandoffError, OSError, ValueError) as error:
        _refuse(f"version 3's state.json does not verify ({error})")
    got = scopes_of(state.get("s7_stopped") or [])
    if got != scope_map(block["scopes"]):
        _refuse(f"the block's scopes are not version 3's s7_stopped: {json.dumps(got, sort_keys=True)}")
    checked["scopes"] = got
    checked["approval"] = dict(file=approval.name, commit=last[0], fingerprint=fp)
    return checked


# ------------------------------------------------------------------------------------------------ the export

def read_carried(path, sha256):
    """Version 3's results.json: its sidecar must verify (handoff.read_json) and its SHA-256 equal the frozen one."""
    try:
        body = H.read_json(path)
    except (H.HandoffError, OSError, ValueError) as error:
        _refuse(f"version 3's results.json at {path} does not verify ({error})")
    got = gate.sha256_file(path)
    if got != sha256:
        _refuse(f"{path} is not version 3's results.json (SHA-256 {got[:16]}, the block names {sha256[:16]})")
    if not isinstance(body, dict) or not isinstance(body.get("qualities"), dict) or \
            not isinstance(body.get("caveats"), list):
        _refuse(f"{path} is not a results.json (no qualities or caveats)")
    return body


def judged_scope_failures(s7_results, scopes):
    """The supplement's failed S7 checks (state.json s7_results entries with s7_passed false) on actors of scopes
    version 3 judged: a learner version 3's S7 did not stop in that quality (stage 1's actors are medium-expert's), or,
    where it stopped only the learner's IW setup, an actor outside that setup (claims.S7_SCOPE "learner": alone,
    standard, stage 1's weights-of-1), such as an IW scope's comparators at seeds 1-4 or at 50,000 steps. scopes:
    {quality: {learner: "learner" | "iw"}}, validate's. Version 3's pre-specified S7 covered only stage 1's and seed 0's
    final actors, so these change no judged key; the appended caveat names them (caveat_of)."""
    from experiments.shift_hosts import claims as C

    out = []
    for r in s7_results or []:
        if r.get("s7_passed") is not False:
            continue
        stopped = (scopes.get(r.get("quality")) or {}).get(r.get("learner"))
        if stopped == "learner" or (stopped == "iw" and C.S7_SCOPE.get(r.get("setup")) == "iw"):
            continue                                    # a scope version 3 stopped: shown, never judged
        out.append({k: r.get(k) for k in ("label", "run", "quality", "learner", "setup", "seed", "step", "file",
                                          "sha256")})
    return out


def caveat_of(judged):
    """The one caveat the export appends: CAVEAT, and when S7 failed on actors of scopes version 3 judged
    (judged_scope_failures), a sentence naming each of them as a caveat on version 3's judged claims there."""
    if not judged:
        return CAVEAT
    names = "; ".join(str(e.get("run")) + (f" at {e['step']:,} steps" if isinstance(e.get("step"), int) else "")
                      for e in judged)
    return (f"{CAVEAT} S7 also failed in the supplement on {len(judged)} actor(s) of scopes version 3 judged, whose "
            f"pre-specified S7 check covered only stage 1's and seed 0's final actors: {names}. Version 3's judgment "
            f"there is carried unchanged; these failures are a caveat on its judged claims for those learners "
            f"(s7_supplement.s7_failures_in_judged_scopes).")


def assert_judged_unchanged(results, v3, caveat=CAVEAT):
    """Refuse (gate.ApprovalError) unless results is v3 plus only qualities.<quality>.s7_shown, s7_supplement and one
    caveat appended (caveat: CAVEAT, or caveat_of's longer text, which starts with it): with those taken away, its
    canonical JSON must equal v3's."""
    if "s7_supplement" in v3 or any(isinstance(b, dict) and "s7_shown" in b for b in v3["qualities"].values()):
        _refuse("version 3's results.json already holds a supplement key")
    if not isinstance(caveat, str) or not caveat.startswith(CAVEAT):
        _refuse("the supplement's caveat does not start with supplement.CAVEAT")
    rest = copy.deepcopy(results)
    rest.pop("s7_supplement", None)
    for body in (rest.get("qualities") or {}).values():
        if isinstance(body, dict):
            body.pop("s7_shown", None)
    cav = rest.get("caveats")
    if not isinstance(cav, list) or not cav or cav[-1] != caveat:
        _refuse("the supplement's caveat is not the last caveat")
    rest["caveats"] = cav[:-1]
    if canonical(rest) != canonical(v3):
        _refuse("a key of version 3's results.json would change; nothing is written")


def _no_timing(x):
    """A copy without timing fields (any key ending in "seconds"), for R1's comparison of refresh records."""
    if isinstance(x, dict):
        return {k: _no_timing(v) for k, v in x.items() if not str(k).endswith("seconds")}
    if isinstance(x, list):
        return [_no_timing(v) for v in x]
    return x


def _actor_trees(folder, rec):
    """{step: actor_tree_sha256} of a run's snapshot actors (actors/step<T>.npz manifests; runner.Run.snapshot)."""
    out = {}
    for t in sorted(rec.get("snapshots") or {}, key=int):
        try:
            out[t] = (H.read_manifest(Path(folder) / "actors" / f"step{t}.npz").get("meta") or {}).get(
                "actor_tree_sha256")
        except (H.HandoffError, OSError, ValueError):
            out[t] = None
    return out


def _s7_points(folder):
    """{lane folder: {"readout/bank": [max_points, mean_points]}} of a run's S7 lanes (None without a closed loop)."""
    out = {}
    for p in sorted((Path(folder) / "lanes").glob("s7-step*/s7.json")):
        try:
            cl = H.read_json(p).get("closed_loop")
        except (H.HandoffError, OSError, ValueError):
            out[p.parent.name] = "does not verify"
            continue
        out[p.parent.name] = None if not isinstance(cl, dict) else {
            k: [v.get("max_points"), v.get("mean_points")] for k, v in sorted(cl.items())
            if isinstance(v, dict) and "/" in k}
    return out


def reproduction(v4_root=ROOT, v3_root=V3_ROOT, v3_fingerprint=V3["fingerprint"]):
    """R1, a readout that never stops anything (and never raises): every supplement record against version 3's record
    of the same run id, where one exists and names version 3's fingerprint (None: not checked). Compared: hashes
    initial, key_carry, params_10000 and final, the outcome, the refresh records without timings, every common
    snapshot actor's actor_tree_sha256 (an actor file readable on neither side is listed under not_compared), and the
    S7 lanes' closed-loop largest and mean per-start points. Identity, code and file SHA-256 values differ by design
    and are not compared. -> [{run_id, result "equal" | "differs" | "missing" | "unreadable", differs, not_compared,
    v4_record_sha256, v3_record_sha256, note}]."""
    out = []
    v4_root, v3_root = Path(v4_root), Path(v3_root)
    for path in sorted(v4_root.glob("stage*/*/seed*/*/record.json")):
        rid = path.parent.relative_to(v4_root).as_posix()
        other = v3_root / rid / "record.json"
        entry = dict(run_id=rid, result="missing", differs=[], not_compared=[], v4_record_sha256=None,
                     v3_record_sha256=None, note=None)
        try:
            entry["v4_record_sha256"] = gate.sha256_file(path)
            if not other.is_file():
                entry["note"] = "version 3 has no record of this run"
                out.append(entry)
                continue
            entry["v3_record_sha256"] = gate.sha256_file(other)
            a, b = H.read_json(path), H.read_json(other)
            if v3_fingerprint is not None and (b.get("code") or {}).get("fingerprint") != v3_fingerprint:
                entry["note"] = "the record at version 3's root names another fingerprint than version 3's"
                out.append(entry)
                continue
            differs = [f"hashes.{part}" for part in ("initial", "key_carry", "params_10000", "final")
                       if canonical((a.get("hashes") or {}).get(part)) != canonical((b.get("hashes") or {}).get(part))]
            if a.get("outcome") != b.get("outcome"):
                differs.append("outcome")
            if canonical(_no_timing(a.get("refreshes"))) != canonical(_no_timing(b.get("refreshes"))):
                differs.append("refreshes")
            ta, tb = _actor_trees(path.parent, a), _actor_trees(other.parent, b)
            for t in sorted(set(ta) & set(tb), key=int):
                if ta[t] is None and tb[t] is None:
                    entry["not_compared"].append(f"actor_tree_sha256 at step {t} (no readable actor file on either "
                                                 "side)")
                elif ta[t] != tb[t]:
                    differs.append(f"actor_tree_sha256 at step {t}")
            sa, sb = _s7_points(path.parent), _s7_points(other.parent)
            differs += [f"S7 closed loop ({k})" for k in sorted(set(sa) & set(sb))
                        if canonical(sa[k]) != canonical(sb[k])]
            entry.update(result="differs" if differs else "equal", differs=differs)
        except Exception as error:                      # noqa: BLE001  (a readout: never raises)
            entry.update(result="unreadable", note=f"{type(error).__name__}: {error}")
        out.append(entry)
    return out


def _driver_state(root):
    """The supplement driver's state.json (verified), or {} when it cannot be read (listed under s7_supplement)."""
    try:
        state = H.read_json(Path(root) / "driver" / "state.json")
        return state if isinstance(state, dict) else {}
    except (H.HandoffError, OSError, ValueError):
        return {}


def _cut(summary, fields):
    return {k: summary.get(k) for k in fields}


def _numbers(C, runs, L, setup, banks=None):
    """{bank: {J, points}} of one setup: the per-seed values (claims.setup_values' reading) with their mean and sd."""
    out = {}
    for bank in banks or C.BANKS:
        out[bank] = {}
        for field in ("J", "points"):
            v = C.setup_values(runs, L, setup, bank, field)
            out[bank][field] = dict(_cut(v["summary"], SETUP_FIELDS), seeds=[p["seed"] for p in v["per_seed"]])
    return out


def _records(L, setup, seeds, aux, unscored_aux, runs, r1, v3_q):
    """Per seed, the supplement run's provenance: its record and scores SHA-256, that it trained under version 4, and
    version 3's record of the same run (R1)."""
    by_r1 = {e["run_id"]: e for e in r1}
    excluded = {(e.get("setup"), e.get("seed")): e for e in ((v3_q.get("s7_stopped") or {}).get("excluded") or [])
                if e.get("learner") == L}
    out = []
    for sd in seeds:
        a = aux.get((L, setup, sd)) or unscored_aux.get((L, setup, sd))
        if a is None:
            out.append(dict(seed=sd, state="missing", trained_under=None))
            continue
        fin, half = (a.get("scores") or {}).get("final"), (a.get("scores") or {}).get("half")
        r = (runs.get((L, setup)) or {}).get(sd)
        e = by_r1.get(a["run_id"]) or {}
        out.append(dict(seed=sd, run_id=a["run_id"], state=(r or {}).get("state") or (
            "unscored" if (L, setup, sd) in unscored_aux else "halted"), record_sha256=a["record_sha256"],
            scores_sha256=None if fin is None else fin["npz_sha256"],
            scores_half_sha256=None if half is None else half["npz_sha256"], trained_under=f"version {VERSION}",
            v3_record_sha256=e.get("v3_record_sha256"), r1=e.get("result"), r1_differs=e.get("differs") or [],
            v3_excluded=excluded.get((setup, sd))))
    return out


def _s7_lanes(q, L, setups, s7_results, s7_read):
    """The supplement's S7 results of one learner's planned runs in quality q (state.json s7_results, every lane of
    seeds 0-4 at 50,000 and 100,000 steps) and of its stage-1 actors when q is stage 1's quality (medium-expert), each
    marked with export.read_s7's checks of the file (it verifies and is bound to its record's actor). shown() splits
    them into the scope's own and its comparators'."""
    read = {s.get("sha256"): s for s in s7_read if s.get("sha256")}
    out = []
    for r in s7_results:
        if r.get("quality") != q or r.get("learner") != L or (r.get("setup") not in setups
                                                               and str(r.get("run", "")).startswith("stage2/")):
            continue
        s = read.get(r.get("sha256")) or {}
        out.append(dict(r, verified=bool(s), bound_to_record=s.get("bound_to_record")))
    return out


def _writeup(C, q, L, scope, setups, lanes, v3_scope, comparator_lanes=()):
    """The user's format (The idea / What we tested / What happened / Why) for one shown scope: no badge, no verdict.
    lanes are the scope's own S7 results; an IW scope's comparators (version 3's judged runs) are counted apart."""
    name = C.LEARNER_NAMES[L]
    cite = f"results.json:qualities.{q}.s7_shown.scopes.{L}"
    what = "the learner" if scope == "learner" else "its IW setup"
    idea = (f"The scores must measure {name}'s trained policy: the float64 scoring adapter and a float32 restatement "
            f"of the trained network, each driving the simulator from the same start, should agree within 0.05 points "
            f"on every start (S7, with version 2's post-hoc limit of 0.05 points (PRESPEC.md section 19)).")
    v3_runs = ", ".join((v3_scope or {}).get("runs") or []) or "a run"
    tested = (f"Version 3's S7 check stopped {what} on {q} ({v3_runs}). Version 4's post-hoc supplement (PRESPEC.md "
              f"section 21) trained and scored every run of the scope in full ({', '.join(setups)}; seeds 0-4) under "
              f"its own root, and ran S7 on the 50,000- and 100,000-step actors of every alone, standard and IW run at "
              f"seeds 0-4.")
    failed = [r for r in lanes if r.get("s7_passed") is False]
    worst = max((v.get("max_points") for r in lanes for v in (r.get("closed_loop") or {}).values()
                 if isinstance(v.get("max_points"), (int, float))), default=None)
    happened = (f"S7 ran on {len(lanes)} actors of this scope in the supplement and failed on {len(failed)}"
                + ("" if worst is None else f"; the largest per-start difference was {worst:.4f} points"))
    if comparator_lanes:
        off = [r for r in comparator_lanes if r.get("s7_passed") is False]
        happened += (f". On its comparators, version 3's judged {name} runs outside the IW setup (alone and standard "
                     f"at seeds 0-4, and stage 1's), S7 ran on {len(comparator_lanes)} actors and failed on {len(off)}"
                     + ("; version 3's judgment is carried unchanged, and these failures are a caveat on it "
                        "(s7_supplement.s7_failures_in_judged_scopes)" if off else ""))
    happened += f". [{cite}]"
    why = ["trained and scored under version 4's post-hoc supplement (section 21) at the owner's instruction of 20 "
           "October 2026, so that the stopped scope's runs exist to work from",
           "shown, never judged: no claim, badge, p-value, Holm step, mark or bar; version 3's judgment of this scope "
           "(not judged, p = 1 in Holm's families) is carried unchanged",
           "any use toward an S7 fix is a new pre-registered test with its own kill rule, confirmed on fresh seeds "
           "10-14"]
    title = f"{name} on {q}: {'learner' if scope == 'learner' else 'IW setup'} ({LABEL})"
    md = "\n".join([f"### {title}", "", f"- **The idea:** {idea}", f"- **What we tested:** {tested}",
                    f"- **What happened:** {happened}", "- **Why:**"] + [f"  - {w[0].upper() + w[1:]}." for w in why])
    return dict(quality=q, learner=L, claim="S7", status=LABEL, title=title, idea=idea, tested=tested,
                happened=happened, why=why, cite=cite, markdown=md)


def shown(q, col, plan_q, scopes_q, v3_q, r1, state, caveat=CAVEAT):
    """qualities.<q>.s7_shown: per scope of the quality's plan, its setups (per-seed values with mean and sd on every
    bank, final and 50% snapshots, with each run's provenance), its paired differences (claims.paired, summaries cut
    to PAIRED_FIELDS: per-seed values, mean, sd, se and one-sided 95% bounds), its S7 lanes (s7: the scope's own,
    every lane of a learner scope and an IW scope's IW setups, claims.S7_SCOPE "iw"; comparator_s7: an IW scope's
    other actors, version 3's judged runs), version 3's and version 4's S7 entries, and a write-up. caveat: the
    export's appended caveat (caveat_of). Never a bar, mark, window, badge, Holm entry, t or p-value."""
    from experiments.shift_hosts import claims as C
    from experiments.shift_hosts import export as E

    if not plan_q:
        return dict(label=LABEL, note="no scope stopped by S7 in this quality")
    runs, aux, unscored_aux = col["runs"][q], col["aux"][q], col["unscored_aux"][q]
    half_runs = E._view_runs(runs, aux, "half")
    s7_read = [s for a in list(aux.values()) + list(unscored_aux.values()) for s in a["s7"]]
    s7_read += col.get("s7_stage1") or []
    v3_scopes = (v3_q.get("s7_stopped") or {}).get("scopes") or {}
    v4_entries = [e for e in state.get("s7_stopped") or [] if e.get("quality") == q]
    out = {}
    for L, setups in plan_q.items():
        scope = scopes_q[L]
        shown_setups = setups if scope == "learner" else ["iw"]
        blocks = {}
        for s in shown_setups:
            seeds = E.SETUP_SEEDS[s]
            blocks[s] = dict(name=C.SETUP_NAMES[s], seeds=list(seeds),
                             records=_records(L, s, seeds, aux, unscored_aux, runs, r1, v3_q),
                             final=_numbers(C, runs, L, s), half=_numbers(C, half_runs, L, s))
        comparators = {s: _records(L, s, C.SEEDS, aux, unscored_aux, runs, r1, v3_q)
                       for s in setups if s not in shown_setups}
        comparisons = []
        for a, b in PAIRS[scope]:
            if a not in setups or b not in setups:
                continue
            for bank in PAIRED_BANKS:
                for field in ("J", "points"):
                    d = C.paired(runs, L, a, b, bank, field)
                    comparisons.append(dict(comparison=d["comparison"], bank=bank, field=field,
                                            per_seed=d["per_seed"], missing=d["missing"],
                                            summary=_cut(d["summary"], PAIRED_FIELDS),
                                            bounds=f"one-sided 95%: mean -+ {C.T95:.6f} x se (t, 4 degrees of "
                                                   f"freedom); shown, not judged"))
        own, others = [], []
        for r in _s7_lanes(q, L, setups, state.get("s7_results") or [], s7_read):
            (own if scope == "learner" or C.S7_SCOPE.get(r.get("setup")) == "iw" else others).append(r)
        out[L] = dict(scope=scope, label=LABEL, setups_trained=list(setups), setups=blocks, comparators=comparators,
                      comparisons=comparisons, s7=own, comparator_s7=others, v3=v3_scopes.get(L),
                      v4_entries=[e for e in v4_entries if e.get("learner") == L],
                      writeup=_writeup(C, q, L, scope, setups, own, v3_scopes.get(L), others))
    return dict(label=LABEL, caveat=caveat, post_hoc=True, version=VERSION, scopes=out)


def _ledgers(root, v3_root, block, made_up):
    """The supplement's own ledger against section 12's cap numbers per quality (decision D2) and the study-wide totals:
    version 3's ledger (which also holds versions 1 and 2's stage-1 sessions, budget.Ledger.totals summing every
    session) plus the supplement's; versions 1-2's share is shown under "earlier" as an "of which", never added again.
    Never raises: what cannot be read says why."""
    from experiments.shift_hosts import budget as B

    out = dict(caps={q: dict(card_hours=c["card_hours"], processor_hours=c["processor_hours"])
                     for q, c in B.CAPS.items()}, booking="the supplement's own ledger under the same per-quality cap "
                                                          "numbers (decision D2); the study-wide total is disclosed")
    totals = {}
    for name, path, want in (("version 4", Path(root) / "driver" / "ledger.json", None),
                             ("version 3", Path(v3_root) / "driver" / "ledger.json",
                              None if made_up else block["source"]["ledger_sha256"])):
        try:
            if not path.is_file():
                raise FileNotFoundError(f"no ledger at {path}")
            if want is not None and gate.sha256_file(path) != want:
                raise ValueError("it is not the ledger the block names")
            led = B.Ledger(path)
            totals[name] = {q: led.totals(q) for q in QUALITIES}
        except Exception as error:                      # noqa: BLE001  (shown, never a reason to refuse)
            totals[name] = dict(not_read=f"{type(error).__name__}: {error}")
    out["ledgers"] = totals
    study = {}
    for q in QUALITIES:
        parts = [t[q] for t in totals.values() if q in t]       # version 3's already counts versions 1 and 2
        study[q] = dict(card_hours=sum(p["card_hours"] for p in parts),
                        processor_hours=sum(p["processor_hours"] for p in parts),
                        complete=all(q in t for t in totals.values()))
    out["study_wide"] = dict(totals=study, earlier=EARLIER_SPEND)
    return out


def build(permission, *, frozen=None, root=ROOT, block=None, carried_path=None, v3_root=V3_ROOT, anchors=None):
    """outputs/shift_hosts/results.json's content (module docstring): version 3's results.json (carried_path, by
    default outputs/shift_hosts_v3/results.json, against block.source.results_sha256) plus qualities.<quality>.s7_shown,
    s7_supplement and the caveat (caveat_of), built from the supplement's root only (version 3's root is read for R1
    and its ledger only). permission and frozen as export.resolve (a verified approval, or guard.MADE_UP with a made-up
    Frozen for the stage-0 tests); block: prespec.json's "supplement" by default. Refuses (gate.ApprovalError) when the
    block does not validate, the carried file is not version 3's, or a version-3 key would change."""
    from experiments.shift_hosts import claims as C
    from experiments.shift_hosts import export as E

    fz, perm = E.resolve(permission, frozen)
    block = prespec_block() if block is None else block
    v = validate(block, C.LEARNERS, C.QUALITIES, C.SEEDS, E.CHOICE19)
    carried_path = Path(carried_path) if carried_path is not None else gate.REPO / SOURCE_FILES["results_sha256"]
    v3 = read_carried(carried_path, block["source"]["results_sha256"])
    if v3.get("made_up") is not fz.made_up:
        _refuse(f"the carried results.json says made_up {v3.get('made_up')!r}; this build is "
                f"{'made-up' if fz.made_up else 'real'}")
    anchors = anchors or E.read_start_bank_anchors()
    drv = E.read_driver(root, fz)
    col = E.collect(root, fz, C.QUALITIES, frozenset(drv["accepted_s5"]))
    checked = E.check_score_anchors(col, anchors["anchors"])
    integ = E.integrity(col, fz, drv, checked)
    state = _driver_state(root)
    r1 = reproduction(root, v3_root, None if fz.made_up else block["source"]["fingerprint"])
    planned = run_ids(v["plan"], C.SEEDS, C.LEARNERS, made_up=fz.made_up)
    recorded = {r["run_id"] for r in col["listed"]}
    want = {rid for ids in planned.values() for rid in ids}
    judged = judged_scope_failures(state.get("s7_results") or [], v["scopes"])
    caveat = caveat_of(judged)
    results = copy.deepcopy(v3)
    results["caveats"].append(caveat)
    for q, body in results["qualities"].items():
        body["s7_shown"] = shown(q, col, v["plan"].get(q) or {}, v["scopes"].get(q) or {}, v3["qualities"][q], r1,
                                 state, caveat)
    timed = [a for q in C.QUALITIES for a in list(col["aux"][q].values()) + list(col["unscored_aux"][q].values())]
    results["s7_supplement"] = dict(
        version=VERSION, label=LABEL, caveat=caveat, post_hoc=True,
        carried_from=dict(version=3, path=E.rel(carried_path), sha256=block["source"]["results_sha256"],
                          fingerprint=block["source"]["fingerprint"]),
        built=dict(utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), permission=perm,
                   frozen=dict(fingerprint=fz.fingerprint, prespec_sha256=fz.prespec_sha256,
                               code_manifest_sha256=fz.manifest_sha256),
                   code_sha256=dict(E.code_sha256(), **{E.rel(Path(__file__)): gate.sha256_file(__file__)}),
                   root=E.rel(root), driver=dict(path=drv["path"], read=drv["read"], mode=drv["mode"],
                                                 files=drv["files"], why=drv["why"], problems=drv["problems"])),
        block=block, plan=v["plan"], s7_settings=dict(all_seeds=v["s7_all_seeds"], half_step=v["s7_half"]),
        planned_runs=planned, plan_check=dict(planned=len(want), recorded=len(recorded & want),
                                              not_recorded=sorted(want - recorded),
                                              outside_the_plan=sorted(recorded - want)),
        integrity=integ, s7_results=state.get("s7_results") or [], s7_stopped=state.get("s7_stopped") or [],
        s7_failures_in_judged_scopes=judged,
        driver_supplement=state.get("supplement"), reproduction=r1, runs=col["listed"], excluded=col["excluded"],
        unscored=col["unscored"], halted_runs=col["halts"],
        totals=dict(card_seconds=float(sum((a["timings"] or {}).get("card_seconds") or 0.0 for a in timed)),
                    processor_seconds=float(sum((a["timings"] or {}).get("processor_seconds") or 0.0 for a in timed))),
        budget=_ledgers(root, v3_root, block, fz.made_up))
    assert_judged_unchanged(results, v3, caveat)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m experiments.shift_hosts.supplement",
                                description="Version 4's S7 supplement (PRESPEC.md section 21).")
    sub = p.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="write outputs/shift_hosts/results.json: version 3's file plus the shown "
                                       "supplement (needs version 4's terminal approval)")
    ex.add_argument("--out", help="default outputs/shift_hosts/results.json")
    a = p.parse_args(argv)
    try:
        approval = gate.require_run_approval("supplement export")       # first: nothing is read before it
        from experiments.shift_hosts import export as E

        results = build(approval)
        out = Path(a.out) if a.out else E.OUT
        digest = E.write(results, out)
    except gate.ApprovalError as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return 2
    print(f"wrote {out} ({digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
