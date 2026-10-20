"""Near-80 (version 9, DRAFT): the payload of the final prereg.json, built by freeze.py write.

DRAFT, not frozen, awaiting the owner. Nothing here trains or reads a result. It reads the draft the final version is
built from (kept in prereg_history/ by freeze.py), copies its content keys, marks every choice of section 16 as taken
with the owner's answer (near80_constants.OWNER_ANSWER, which DECISION.md fills; the freeze refuses while it is open),
and adds what the approval chain, the runner and the looks rely on: the constants (with the claims block), the claims
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

DRAFT_KIND = "near80v9-preregistration-draft"
CONTENT_KEYS = ("request", "question", "settings", "host", "arms", "pooled_groups", "outcome_free_count",
                "refusal_rule", "situations", "seeds", "measures", "claims", "kill_rules", "compute", "departures",
                "no_training_facts_seeds_0_4")
# Version 9's draft uses the code's logical ids already: nothing is renamed.
ARM_RENAMES = {}
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
    """The final wording of the content keys where the final version says more than the draft it was built from.
    Version 9's draft is written in its final wording, so nothing is amended; the claims block that freezes is the
    constants' (near80_constants.CLAIMS), copied into content['claims']['block'] so the approval summary shows it.
    Changes content in place; returns the list of what was amended (prereg.json's 'amended_in_final')."""
    c = content.setdefault("claims", {})
    c["block"] = json.loads(AC.canonical(K.CLAIMS))
    return ["claims: the claims block of near80_constants.CLAIMS, as frozen"]


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
    """The lines approve_local_near80_v9.py prints: the two claims, the decision rule, the refusal rule and the stops,
    from the claims block and the content keys."""
    b = (content.get("claims") or {}).get("block") or K.CLAIMS
    m1, c2 = b["C1"]["marks"], b["C2"]
    twins = ", ".join(f"{K.ARM_PLAIN.get(t, t)} (margin {c2['margins'].get(t)})" for t in c2["comparators"])
    claims = [
        f"C1, {b['C1']['name']}: net lead over the best world-blind rival at least {m1['net_lead_margin']} points; "
        f"keeps the gain in big waves and big kicks (loss at most {-m1['keeps_floor']} points); beats the matched "
        f"even pull by at least {m1['even_pull_margin']} points.",
        f"C2, {c2['name']}, paired by seed on the {c2['measure']} measure: {twins}; statistic {c2['statistic']}, "
        f"alpha {c2['alpha']}, multiplicity {c2['multiplicity']}, at least {c2['min_positive_seeds']} seeds ahead, "
        f"noise floor {c2['noise_floor_max']}.",
        f"Decision rule: {b.get('decision')} (one of {', '.join(K.DECISION_RULES)}).",
    ]
    rr = content.get("refusal_rule") or {}
    kr = content.get("kill_rules") or {}
    stops = [f"Refusal: {rr.get('rule', 'see section 8')}.",
             f"Outcome-free count: {(content.get('outcome_free_count') or {}).get('rule', 'see section 8.1')}."]
    stops += [f"{k}: {v}." for k, v in sorted(kr.items()) if isinstance(v, str)]
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
        runner={"entry": "run_near80_v9.sh", "lanes_default": K.DEFAULT_LANES,
                "memory_rule": draft["compute"]["memory_rule"], "outputs": "runs/shift_explore/near80_v9/",
                "phases": list(K.PHASE_KINDS)},
        summary=summary_block(content),
        amended_in_final=amended,
        files={"documents": ["PREREGISTRATION.md", "README.md"], "code_manifest": K.MANIFEST_NAME,
               "unit_tests": list(K.TEST_FILES), "approval_script": "approve_local_near80_v9.py",
               "outputs": "runs/shift_explore/near80_v9/ (write-once); committed copies in near80/results-*/"},
        **content)
    payload = json.loads(AC.canonical(payload))
    words = draft_words({k: v for k, v in payload.items() if k not in ("draft", "claims_word_for_word")})
    if words:
        raise PreregProblem(f"draft wording left in the final payload: {words[:5]}")
    return payload
