"""Near-80 (version 8): the payload of the final prereg.json, built by freeze.py write.

FINAL, awaiting the owner's approval. Nothing here trains or reads a result. It reads the draft the final version is
built from (kept in prereg_history/ by freeze.py), copies its content keys with the final wording and the code's arm
names, marks every choice of section 16 as taken (the owner accepted every default on 15 October 2026 at 19:24 UTC,
"accept the defaults"), and adds what the approval chain, the runner and the looks rely on: the constants, the claims
word for word (section 14 of PREREGISTRATION.md, extracted by claims_text.extract_section14), the code manifest, the
pins, the frozen code's no-training checks on seeds 0-4, the approval route and the runner.
"""
import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import approval_check as AC  # noqa: E402
import near80_constants as K  # noqa: E402

DRAFT_KIND = "near80-preregistration-draft"
CONTENT_KEYS = ("request", "question", "band", "thrusters", "removal_combination", "random_rows_control", "settings",
                "host", "arms", "fixed_references", "situations", "seeds", "measures", "kill_rules",
                "validity_on_seeds_10_14", "pass_marks", "compute", "departures", "no_training_facts_seeds_0_4")
# The draft's arm names -> the code's logical ids (section 2 of the build): keys and strings.
ARM_RENAMES = {"range_hinge_k1": "range_hinge", "random_surface_1": "random_1", "random_surface_2": "random_2"}
FROZEN_REL = K.PRE_APPROVAL_DIR + "/" + K.NO_TRAINING_FROZEN_NAME
DRAFT_WORD = re.compile(r"\bdraft\b", re.IGNORECASE)


class PreregProblem(Exception):
    """The final payload cannot be built from what is there."""


def find_draft(n80_root):
    """The one draft in prereg_history/ (name prereg_draft_<first 16 of its SHA-256>.json, sidecar matching).
    Returns (relative path, SHA-256, payload)."""
    hist = Path(n80_root) / K.HISTORY_DIR
    found = sorted(hist.glob(K.DRAFT_HISTORY_NAME.format(prefix="*"))) if hist.is_dir() else []
    found = [p for p in found if not p.name.endswith(".sha256")]
    if len(found) != 1:
        raise PreregProblem(f"exactly one draft must be kept in {hist}; found {[p.name for p in found]}")
    path = found[0]
    payload = AC.read_json_verified(path, path.name)
    sha = AC.file_sha256(path)
    if path.name != K.DRAFT_HISTORY_NAME.format(prefix=sha[:K.PREFIX_LEN]):
        raise PreregProblem(f"{path.name} does not carry the first 16 characters of its SHA-256 ({sha[:16]})")
    if payload.get("kind") != DRAFT_KIND or payload.get("status") != "draft":
        raise PreregProblem(f"{path.name} is not the near-80 draft")
    return f"{K.HISTORY_DIR}/{path.name}", sha, payload


def finalize(x):
    """The draft's content with the code's arm names (keys and strings)."""
    if isinstance(x, dict):
        return {ARM_RENAMES.get(k, k): finalize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finalize(v) for v in x]
    if isinstance(x, str):
        for old, new in ARM_RENAMES.items():
            x = x.replace(old, new)
        return x
    return x


def amend(content):
    """The final wording of the content keys where the final version, after the review of the runner, says more than
    or other than the version it was built from (PREREGISTRATION.md sections 5.3, 5.5, 8.2, 8.3, 9.1, 13 and 14.6).
    Changes content in place; returns the list of what was amended (prereg.json's 'amended_in_final')."""
    done = []
    c = content.setdefault("compute", {})
    c.update(no_training_and_plumbing="about 2-2.5 (the unit tests, about 60 smoke phases and the full-length "
                                      "reproduction check)",
             exploratory_seeds_0_4="about 11-16",
             confirmatory_seeds_10_14="about 12-16 near-80 and random rows, plus about 6-8.5 for the main setting",
             total="about 31-43",
             wall_time="about 10-13 hours in three sittings (at most 2 policy phases at once under version 7's memory "
                       "rule; about 12-16 CPU-hours of policy phases)",
             guard="a stage that uses more than twice its forecast's upper end, or the whole run past the cap, pauses "
                   "and reports; the plumbing never pauses; a pause lasts until the owner types run_near80.sh "
                   "budget-continue <stage>; each typed record allows one more forecast of that stage and raises the "
                   "cap by as much; no result is read differently because of a pause",
             plumbing_peaks="the reproduction check's full-length peak serves the critic, second-critic, switch, "
                            "clone, reference-actor and policy kinds; every other kind reserves the larger of 1.25 "
                            "times its smoke peak and 2 GiB; either only until an official attempt of the kind is "
                            "measured")
    done.append("compute: the plumbing's cost and the random-rows setting's six-arm program; the budget guard's "
                "continuation; the plumbing's first peaks")
    content["fixed_references"] = list(content.get("fixed_references") or []) + [
        "the best plans and the no-stall rows of sections 4.4 and 4.5 come from the fact base (outputs/shift/oracle/ "
        "and the session's facts); the runner does not recompute them; it computes the textbook rule held to 0.70 "
        "(check 9) and clipped to each setting's recorded range (check 10, [Q]) in the stalling world, and the "
        "anchors in both worlds"]
    done.append("fixed_references: which references the runner computes")
    arms = content.setdefault("arms", {})
    arms["one_program"] = ("BCA's switched ramp and the plumbing check train in step 2's own batches in every setting "
                           "and world (stalling world: the plumbing check alone, then six ramp arms with obca_k1 "
                           "first; no-stall world: one batch of seven, the plumbing check first); the random-rows "
                           "setting fills the other five places with copies of its obca_k1 (pad_ramp_1 to pad_ramp_5, "
                           "checked equal, never saved or scored); meandiff_k1 trains in a batch of its own")
    done.append("arms: one training program for every setting (section 5.3)")
    kr = content.setdefault("kill_rules", {})
    kr["plumbing_retry"] = ("only a reproduction difference is final (plumbing/reproduction_differs.json); any other "
                            "plumbing failure (a unit test, a smoke phase, a reproduction check that does not finish, "
                            "a missing or damaged input, also a saved file whose fingerprint does not match or a "
                            "training file that fails its check inside the reproduction check, a Ctrl-C) is recorded "
                            "(plumbing/failed_<utc>.json) and may be retried "
                            "once its cause is fixed; the reproduction check runs the official policy function and "
                            "only after the unit tests and the smoke passed")
    stops = kr.setdefault("exploratory_stops", {})
    stops["cannot_be_read"] = {
        "rule": "every seed 0-4 is needed: when a number the harm rule or the rival rule needs is missing on any "
                "seed (a phase stopped, or an actor's numbers are missing), the rule cannot be read",
        "on_fail": "stop the near-80 and random-rows settings before seeds 10-14 with section 14.5's sentence "
                   "'A stop rule cannot be read'; claim 4 is not stopped"}
    nf = stops.get("noise_floor")
    if isinstance(nf, dict):
        nf["cannot_be_read"] = ("if a number it needs is missing on any of seeds 0-4, it cannot be read: claim 3 is "
                                "dropped as if it had failed, with section 14.3's 'not measured' sentence")
    done.append("kill_rules: plumbing retries; a stop rule that cannot be read; a noise floor that cannot be read")
    v = content.setdefault("validity_on_seeds_10_14", {})
    v["failed_seeds"] = ("the validity conditions are computed over the fresh seeds that did not fail; a failed seed "
                         "fails every mark of claims 1-3 and claim 1's sentence names it; it never makes claim 1's "
                         "validity fail; claims 2 and 3 then write their 'If a seed failed' sentences (section 14.6); "
                         "a seed with no failure or stop record whose evaluation lacks a number the validity "
                         "conditions or claims 2 and 3 read from an actor that is always trained fails in the same "
                         "way ('its evaluation (numbers missing)'); check 10's count leaves out a failed seed whose "
                         "band data were never built")
    done.append("validity_on_seeds_10_14: failed seeds")
    return done


def draft_words(x, path=""):
    """Every string (path, text) holding the word 'draft' (a check: the final payload has none outside 'draft')."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            out += draft_words(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += draft_words(v, f"{path}/{i}")
    elif isinstance(x, str) and DRAFT_WORD.search(x):
        out.append((path, x))
    return out


def choices_taken(draft_choices):
    """The draft's list of section 16, each entry marked as taken with its default and accepted at the owner's time."""
    out = []
    for c in draft_choices:
        e = dict(c)
        e["taken"] = c["default"]
        e["accepted"] = K.OWNER_ANSWER["utc"]
        out.append(e)
    return out


def pins(repo, manifest):
    p = AC.current_pins(repo)
    p["repository_files_count"] = len(manifest.get("repository_files") or {})
    if p["v6_stage2_common_sha256"] != AC.V6_STAGE2_COMMON_SHA256:
        raise PreregProblem("version 6's stage2_common.py is not the pinned one")
    return p


def load_claims_text(code_dir):
    """claims_text.py of this code folder, loaded by explicit path (C's module)."""
    path = Path(code_dir) / "claims_text.py"
    if not path.is_file():
        raise PreregProblem(f"{path} is missing: the claims cannot be copied word for word")
    name = f"near80_prereg_claims_text_{AC.sha256_bytes(str(path).encode())[:12]}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def frozen_block(facts_record, manifest_sha256):
    """prereg.json's 'no_training_checks_frozen': facts_record = {path, sha256, payload} of
    pre_approval/no_training_checks_seeds0_4.json."""
    if facts_record.get("path") != FROZEN_REL:
        raise PreregProblem(f"the frozen no-training checks must be at {FROZEN_REL}")
    nt = (facts_record.get("payload") or {}).get("no_training_checks") or {}
    return {"path": FROZEN_REL, "sha256": facts_record["sha256"], "passed": nt.get("passed") is True,
            "rules": len(nt.get("rules") or []), "not_run": list(nt.get("not_run") or []),
            "code_manifest_sha256": manifest_sha256}


def summary_block(content):
    """The lines approve_local_near80.py prints: the four claims and the stop rules, from the content keys."""
    pm = content["pass_marks"]
    st = content["kill_rules"]["exploratory_stops"]
    c1 = pm["claim1_bca_holds"]
    c2 = pm["claim2_unseen_commands"]
    claims = [
        "Claim 1, BCA's result holds with the band hidden (near-80 setting, step 2's three marks): "
        f"{c1['net_lead']}; {c1['keeps_the_gain']}; {c1['beats_even_pull']}.",
        "Claim 2, what BCA's controller does with commands it never saw (chooses one of four sentences): "
        f"{c2['gain']}; {c2['overshoot']}.",
        f"Claim 3, the band's cost: {pm['claim3_band_cost']['mark']} (it can be dropped by the noise floor on seeds "
        "0-4).",
        "Claim 4, step 2 confirmed in the main setting (a separate claim): step 2's three marks on seeds 10-14; the "
        "near-80 stops and checks never cancel it.",
    ]
    stops = [f"Too little harm: {st['harm']['rule']}; if not, {st['harm']['on_fail']}.",
             f"The rival unaffected: {st['rival_unaffected']['rule']}; if not, {st['rival_unaffected']['on_fail']}.",
             f"The noise floor: {st['noise_floor']['rule']}; if not, {st['noise_floor']['on_fail']}.",
             f"Not a stop: {st['not_a_stop']}."]
    if st.get("cannot_be_read"):
        stops.insert(2, f"A stop rule cannot be read: {st['cannot_be_read']['rule']}; then "
                        f"{st['cannot_be_read']['on_fail']}.")
    return {"claims": claims, "stop_rules": stops}


def build(n80_root, manifest_sha256, manifest, facts_record, repo=K.ROOT, claims_fn=None):
    """The final prereg.json payload (section 4.5 of the build). facts_record: {path, sha256, payload} of the copied
    frozen no-training checks. claims_fn(md_text) -> the claims block (default: claims_text.extract_section14 of this
    folder's code)."""
    n80_root = Path(n80_root)
    drel, dsha, draft = find_draft(n80_root)
    missing = [k for k in CONTENT_KEYS if k not in draft]
    if missing:
        raise PreregProblem(f"the draft lacks {missing}")
    content = {k: finalize(draft[k]) for k in CONTENT_KEYS}
    amended = amend(content)
    content["arms"]["logical_ids"] = {k: list(v) for k, v in K.ARMS.items()}
    if claims_fn is None:
        claims_fn = load_claims_text(n80_root / "code").extract_section14
    doc = (n80_root / "PREREGISTRATION.md").read_text(encoding="utf8")
    claims = claims_fn(doc)
    frozen = frozen_block(facts_record, manifest_sha256)
    if not frozen["passed"] or frozen["not_run"]:
        raise PreregProblem("the frozen code's no-training checks on seeds 0-4 did not all pass")
    payload = dict(
        kind=K.PREREG_KIND, version=K.VERSION, label=K.LABEL, status=K.STATUS_FINAL, status_line=K.STATUS_LINE_FINAL,
        approvable=True, missing=[], owner_answer=K.OWNER_ANSWER,
        draft={"path": drel, "sha256": dsha,
               "note": "the draft this final version was built from, moved here unchanged by freeze.py"},
        choices=choices_taken(draft["choices"]),
        constants=AC.canonical_constants(),
        claims_word_for_word=claims,
        code={"code_manifest": K.MANIFEST_NAME, "code_manifest_sha256": manifest_sha256,
              "files": manifest["files"]},
        pins=pins(repo, manifest),
        no_training_checks_frozen=frozen,
        approval=AC.approval_section(K.BRANCH),
        runner={"entry": "run_near80.sh", "lanes_default": K.DEFAULT_LANES,
                "memory_rule": draft["compute"]["memory_rule"], "outputs": "runs/shift_explore/near80/",
                "phases": list(K.PHASE_KINDS)},
        summary=summary_block(content),
        amended_in_final=amended,
        files={"documents": ["PREREGISTRATION.md", "README.md"], "code_manifest": K.MANIFEST_NAME,
               "unit_tests": list(K.TEST_FILES), "approval_script": "approve_local_near80.py",
               "outputs": "runs/shift_explore/near80/ (write-once); committed copies in near80/results-*/"},
        **content)
    payload = json.loads(AC.canonical(payload))
    words = draft_words({k: v for k, v in payload.items() if k not in ("draft", "claims_word_for_word")})
    if words:
        raise PreregProblem(f"draft wording left in the final payload: {words[:5]}")
    return payload
