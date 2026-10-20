"""Near-80 (version 9, main setting): the payload of the final prereg.json, built by freeze.py write. DRAFT, not
frozen.

Nothing here trains or reads a result. It reads the draft the final version is built from (kept in prereg_history/
by freeze.py), copies its content keys, marks every choice of section 16 as taken (the owner's acceptance,
OWNER_ANSWER, which DECISION.md fixes), and adds what the approval chain, the runner and the look rely on: the
constants, the claims word for word (section 14 of PREREGISTRATION.md, extracted by claims_text.extract_section14),
the code manifest, the pins, the frozen code's outcome-free counts on seeds 0-4, the approval route, the hosts and
the runner. It refuses while any placeholder of DECISION.md remains (near80_constants.placeholders_present, a
TODO(DECISION.md) mark in the draft or the claims).
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
CONTENT_KEYS = ("request", "question", "setting", "pooled_groups", "outcome_free_count", "refusal_rule", "host",
                "arms", "fixed_references", "situations", "seeds", "measures", "kill_rules", "validity", "pass_marks",
                "compute", "departures", "facts_from_version_8")
FROZEN_REL = K.PRE_APPROVAL_DIR + "/" + K.COUNTS_FROZEN_NAME
DRAFT_WORD = re.compile(r"\bdraft\b", re.IGNORECASE)
TODO_MARK = re.compile(r"TODO\(DECISION\.md")
# The constants that name the draft state itself (not draft wording): left out of the final payload's word check.
DRAFT_STATE_CONSTANTS = ("STATUS_DRAFT", "STATUS_LINE_DRAFT", "DRAFT_NAME", "DRAFT_HISTORY_NAME")


class PreregProblem(Exception):
    """The final payload cannot be built from what is there."""


def find_draft(n80_root):
    """The one version 9 draft in prereg_history/ (name prereg_draft_<first 16 of its SHA-256>.json, sidecar
    matching, version K.VERSION). Version 8's draft (another version) is left alone. Returns (relative path, SHA-256,
    payload)."""
    hist = Path(n80_root) / K.HISTORY_DIR
    found = sorted(hist.glob(K.DRAFT_HISTORY_NAME.format(prefix="*"))) if hist.is_dir() else []
    found = [p for p in found if not p.name.endswith(".sha256")]
    mine = []
    for path in found:
        payload = AC.read_json_verified(path, path.name)
        if payload.get("version") == K.VERSION:
            mine.append((path, payload))
    if len(mine) != 1:
        raise PreregProblem(f"exactly one draft of {K.VERSION} must be kept in {hist}; found "
                            f"{[p.name for p, _ in mine]} (other versions' drafts there: "
                            f"{[p.name for p in found if p not in [m[0] for m in mine]]})")
    path, payload = mine[0]
    sha = AC.file_sha256(path)
    if path.name != K.DRAFT_HISTORY_NAME.format(prefix=sha[:K.PREFIX_LEN]):
        raise PreregProblem(f"{path.name} does not carry the first 16 characters of its SHA-256 ({sha[:16]})")
    if payload.get("kind") != DRAFT_KIND or payload.get("status") != "draft":
        raise PreregProblem(f"{path.name} is not the near-80 draft")
    return f"{K.HISTORY_DIR}/{path.name}", sha, payload


def todo_marks(x, path=""):
    """Every string (path, text) holding a TODO(DECISION.md) mark."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            out += todo_marks(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += todo_marks(v, f"{path}/{i}")
    elif isinstance(x, str) and TODO_MARK.search(x):
        out.append((path, x))
    return out


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
    if not K.OWNER_ANSWER:
        raise PreregProblem("OWNER_ANSWER is not fixed (DECISION.md): the choices cannot be marked as accepted")
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
    """claims_text.py of this code folder, loaded by explicit path."""
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
    """prereg.json's 'pooled_group_counts_frozen': facts_record = {path, sha256, payload} of
    pre_approval/pooled_group_counts_seeds0_4.json (reported; no pass or fail)."""
    if facts_record.get("path") != FROZEN_REL:
        raise PreregProblem(f"the frozen counts must be at {FROZEN_REL}")
    payload = facts_record.get("payload") or {}
    return {"path": FROZEN_REL, "sha256": facts_record["sha256"], "seeds": payload.get("seeds"),
            "worlds": payload.get("worlds"), "summary": payload.get("summary"),
            "code_manifest_sha256": manifest_sha256, "role": "reported; outcome-free; no mark reads it"}


def summary_block(content):
    """The lines approve_local_near80.py prints: the two claims, the refusal rule, the noise floor, the validity."""
    pm = content["pass_marks"]
    c1 = pm["c1_version8_claim4"]
    c2 = pm["c2_twins"]
    claims = [
        "C1, version 8's claim 4 marks unchanged (step 2's three go marks, main setting, fresh seeds): "
        f"{c1['net_lead']}; {c1['keeps_the_gain']}; {c1['beats_even_pull']}.",
        "C2, BCA's switched ramp beats each plain twin (the split-conformal switch, the mean-difference switch with "
        f"the ramp, the mean-difference switch with the hand-set range limit), paired by seed: {c2['number']}; margin "
        f"{c2['net_margin']}; seeds positive {c2['positive_seeds_min']}; multiplicity {c2['multiplicity']}.",
    ]
    kr = content["kill_rules"]
    rules = [f"The refusal rule: {content['refusal_rule']['then']}; {content['refusal_rule']['marks']}.",
             f"The noise floor: {kr['noise_floor']['rule']}; if not, {kr['noise_floor']['on_fail']}.",
             f"Validity: {content['validity']['claims']}.",
             f"Plumbing stops: {kr['plumbing_stops']}."]
    return {"claims": claims, "rules": rules}


def hosts_block():
    return {name: dict(profile) for name, profile in K.HOST_PROFILES.items() if name != "test"}


def build(n80_root, manifest_sha256, manifest, facts_record, repo=K.ROOT, claims_fn=None):
    """The final prereg.json payload. facts_record: {path, sha256, payload} of the copied frozen counts.
    claims_fn(md_text) -> the claims block (default: claims_text.extract_section14 of this folder's code)."""
    n80_root = Path(n80_root)
    left = K.placeholders_present()
    if left:
        raise PreregProblem(f"placeholders remain (DECISION.md): {left}")
    drel, dsha, draft = find_draft(n80_root)
    missing = [k for k in CONTENT_KEYS if k not in draft]
    if missing:
        raise PreregProblem(f"the draft lacks {missing}")
    content = {k: draft[k] for k in CONTENT_KEYS}
    marks = todo_marks(content)
    if marks:
        raise PreregProblem(f"the draft still carries TODO(DECISION.md) marks: {marks[:5]}")
    content["pass_marks"]["c2_twins"] = dict(content["pass_marks"]["c2_twins"], **K.MARKS_C2)
    content["seeds"] = dict(content["seeds"], fresh=list(K.FRESH), fresh_count=len(K.FRESH), plumbing=list(K.SMOKE_SEEDS))
    content["arms"]["logical_ids"] = {k: list(v) for k, v in K.ARMS.items()}
    if claims_fn is None:
        claims_fn = load_claims_text(n80_root / "code").extract_section14
    doc = (n80_root / "PREREGISTRATION.md").read_text(encoding="utf8")
    claims = claims_fn(doc)
    if claims.get("todo_marks"):
        raise PreregProblem(f"section 14 still carries TODO(DECISION.md) marks: {claims['todo_marks']}")
    frozen = frozen_block(facts_record, manifest_sha256)
    payload = dict(
        kind=K.PREREG_KIND, version=K.VERSION, label=K.LABEL, status=K.STATUS_FINAL, status_line=K.STATUS_LINE_FINAL,
        approvable=True, missing=[], owner_answer=K.OWNER_ANSWER, supersedes=K.PREVIOUS_VERSION,
        draft={"path": drel, "sha256": dsha,
               "note": "the draft this final version was built from, moved here unchanged by freeze.py"},
        choices=choices_taken(draft["choices"]),
        constants=AC.canonical_constants(),
        claims_word_for_word=claims,
        code={"code_manifest": K.MANIFEST_NAME, "code_manifest_sha256": manifest_sha256,
              "files": manifest["files"]},
        pins=pins(repo, manifest),
        pooled_group_counts_frozen=frozen,
        approval=AC.approval_section(K.BRANCH),
        hosts=hosts_block(),
        runner={"desktop_entry": "run_near80.sh", "cluster_entry": "run_near80_cluster.sh",
                "lanes_default": K.DEFAULT_LANES, "memory_rule": draft["compute"]["memory_rule"],
                "outputs": "runs/shift_explore/near80_v9/", "phases": list(K.PHASE_KINDS)},
        summary=summary_block(content),
        version8_history={"files": list(K.V8_HISTORY_FILES), "dirs": list(K.V8_HISTORY_DIRS),
                          "frozen": K.PRE_APPROVAL_DIR + "/" + K.V8_FROZEN_NAME,
                          "prereg": K.HISTORY_DIR + "/" + K.SUPERSEDED_NAME.format(prefix=K.V8_PREFIX)},
        files={"documents": ["PREREGISTRATION.md", "README.md"], "code_manifest": K.MANIFEST_NAME,
               "unit_tests": list(K.TEST_FILES), "approval_script": "approve_local_near80.py",
               "outputs": "runs/shift_explore/near80_v9/ (write-once); committed copies in near80/results-*/"},
        **content)
    payload = json.loads(AC.canonical(payload))
    check = {k: v for k, v in payload.items() if k not in ("draft", "claims_word_for_word", "files", "version8_history")}
    check["constants"] = {k: v for k, v in check["constants"].items() if k not in DRAFT_STATE_CONSTANTS}
    words = draft_words(check)
    if words:
        raise PreregProblem(f"draft wording left in the final payload: {words[:5]}")
    return payload
