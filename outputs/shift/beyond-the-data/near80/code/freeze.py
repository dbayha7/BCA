"""Near-80 (version 9, main setting): freeze the final version (prereg.json, its code manifest and the frozen
outcome-free counts). DRAFT, not frozen.

Nothing here trains a learner or reads a result. The main session runs it once, after DECISION.md has fixed every
open item:

  python freeze.py check                                 prints what 'write' would do and every problem; writes nothing
  python freeze.py check --as-frozen [--keep]            the same on a TEMPORARY COPY of the folder put into the state
                                                         write's steps 2-4 leave it, then the five test files on that
                                                         copy (about 5 minutes); the real folder is not touched
  python freeze.py write --reason TEXT [--facts-from PATH]

'check --as-frozen' exists because the five test files read the folder's own state (the status lines, the marks, the
draft's place) and must pass in the finalized state before 'write' runs them at its step 6, after it has moved files.
The copy (finalize_copy): the two documents' line 3 set to the FINAL line and every TODO(...) mark replaced by a
stand-in text (AS_FROZEN_FILL; at the real freeze the facts' table and the redraft count are typed in by hand), the
phrase "DRAFT, not frozen[, awaiting the owner]" dropped from the module docstring of every code file and of
approve_local_near80.py and from the header comments of the shell entries (nothing else in them: the constants that
name the draft state are untouched), and the draft moved into prereg_history/ as step 4 moves it. It then reports
what 'check' reports (expected: "No problems found."), extracts section 14, and runs the five test files on the copy
from the work folder with the phases' environment (as step 6). Run it, and make it pass, before 'write'; the by-hand
edits of the real folder must leave it in the same state (the stand-in text aside).

'write', in order (any problem stops it before the next step):
  0. Refuses while any placeholder of DECISION.md remains: near80_constants.placeholders_present() (the seed count,
     C2's margins and multiplicity rule, the owner's answer), a "TODO(DECISION" mark or "**DRAFT" in the documents or
     the draft.
  1. Supersedes the previous final version: a prereg.json of another version (version 8's, approved and run) is
     moved, with its sidecar, into prereg_history/prereg_superseded_<first 16 of its SHA-256>.json (the version
     process of version 8's re-freezes). Refuses if a prereg.json of THIS version exists, if an approval or import
     record of a fingerprint that is not version 8's history exists, or if any official output exists under
     runs/shift_explore/near80_v9/ other than pre/.
  2. The documents are final: PREREGISTRATION.md's title and status line, README.md's status line, no "**DRAFT" and no
     "this draft" in either, a section 14.5; no file under code/ says DRAFT in its first docstring line.
  3. The code folder holds exactly the planned files (near80_constants.CODE_FILES), no __pycache__, and the top-level
     files (near80_constants.TOP_FILES) exist.
  4. prereg_draft.json is renamed into prereg_history/prereg_draft_<first 16 of its SHA-256>.json with a sidecar (a
     rerun checks it there instead; version 8's draft stays beside it).
  5. code_manifest.json (+ .sha256): every near-80 file and every repository file the code imports or executes; the
     earlier manifest (version 8's) is archived in code_manifest_history/ (a rerun with no change keeps it as it is).
  6. The five unit test files, each run alone from the work folder with the phases' environment; all must pass
     ('check --as-frozen' runs the same files on the finalized copy beforehand, so that this step is not the first
     time they meet the final state).
  7. The frozen code's pre-approval facts (facts_near80.py, about 10-15 CPU-minutes, nothing trained): the pooled
     groups' outcome-free counts on seeds 0-4, and on the spent seeds 10-14 the pooled switch from version 8's
     recorded second critics with the kill rule (BCA's and the mean-difference switch's pooled calls equal version
     8's recorded calls on all 10 spent seed-worlds, per seed-world and in the record's spent_summary; otherwise the
     freeze refuses: a redraft). Written to
     runs/shift_explore/near80_v9/pre/pooled_facts_<first 16 of the manifest>.json and copied to
     pre_approval/pooled_facts_seeds0_4_spent10_14.json (+ .sha256).
  8. prereg.json (+ .sha256) from near80_prereg.build; prints the fingerprint and the next steps.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import approval_check as AC  # noqa: E402
import near80_constants as K  # noqa: E402
import near80_prereg as NP  # noqa: E402
import ledger as LG  # noqa: E402

TITLE = "# The main setting on fresh seeds: pre-registration (version 9, near-80 main)"
BOLD_FINAL = "**FINAL, awaiting the owner's approval. Nothing has been trained.**"
DRAFT_NAME = K.DRAFT_NAME
FACTS_ARGS = ("--seeds", "0", "1", "2", "3", "4", "--spent-switch", "10", "11", "12", "13", "14",
              "--worlds", "stall", "no_stall")
FACTS_NAME = "pooled_facts_{prefix}.json"
FACTS_KIND = "near80-v9-pre-approval-facts"
SPENT_SEED_WORLDS = 10
TEST_TIMEOUT_S = 3 * 3600
FACTS_TIMEOUT_S = 2 * 3600
FILL_RULES_HEADING = "### 14.5"
TODO_MARK = re.compile(r"TODO\([^)]*\)")
# 'check --as-frozen' (finalize_copy): the stand-in for a TODO(FREEZE: ...) mark, and the draft phrase of the module
# docstrings and script headers that the freeze-time edit drops.
AS_FROZEN_FILL = "(filled at the freeze from the facts record; this is the as-frozen check's stand-in)"
DRAFT_PHRASE = re.compile(r"DRAFT,\s+not\s+frozen(,\s+awaiting\s+the\s+owner)?")
AS_FROZEN_SCRIPTS = ("approve_local_near80.py", "run_near80.sh", "run_near80_cluster.sh", "near80_slurm_job.sh",
                     "push_runfiles_near80.sh")


class FreezeRefused(Exception):
    """A freeze step refused; what was written before it stays (nothing official is written before step 8)."""


# --------------------------------------------------------------------------------------------- step 0: placeholders

def placeholder_problems(n80_root):
    """Step 0: every open item, in the constants (near80_constants.placeholders_present: the owner's answer) and in
    the documents (any TODO(...) mark: TODO(OWNER: ...) for the owner's answer, TODO(FREEZE: ...) for the facts the
    freeze fills into section 4 and the redraft count of section 0)."""
    out = [f"placeholder not fixed: {name}" for name in K.placeholders_present()]
    for name in ("PREREGISTRATION.md", "README.md", DRAFT_NAME):
        p = Path(n80_root) / name
        if p.is_file():
            text = p.read_text(encoding="utf8")
            n = len(TODO_MARK.findall(text))
            if n:
                out.append(f"{name} carries {n} TODO(...) mark{'s' if n != 1 else ''}")
    hist = Path(n80_root) / K.HISTORY_DIR
    if hist.is_dir():
        for p in sorted(hist.glob(K.DRAFT_HISTORY_NAME.format(prefix="*"))):
            if p.suffix == ".json" and TODO_MARK.search(p.read_text(encoding="utf8")):
                try:
                    if json.loads(p.read_text(encoding="utf8")).get("version") == K.VERSION:
                        out.append(f"{K.HISTORY_DIR}/{p.name} carries TODO(...) marks")
                except ValueError:
                    pass
    return out


# --------------------------------------------------------------------------------------------- steps 1-3: checks

def history_prefixes():
    """The fingerprints' first 16 characters of version 8's approval chain (kept as history)."""
    return {K.V8_PREFIX, K.V8_AMENDMENT_PREFIX}


def previous_prereg(n80_root):
    """('none' | 'mine' | 'previous', payload or None): what prereg.json at the top holds."""
    p = Path(n80_root) / K.PREREG_NAME
    if not p.exists():
        return "none", None
    try:
        body = json.loads(p.read_bytes().decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise FreezeRefused(f"{K.PREREG_NAME} is not JSON") from None
    if not isinstance(body, dict) or body.get("kind") != K.PREREG_KIND:
        raise FreezeRefused(f"{K.PREREG_NAME} is not a near-80 pre-registration")
    return ("mine" if body.get("version") == K.VERSION else "previous"), body


def existing_problems(n80_root, out_root):
    """Step 1: nothing of this version approved, imported or run yet; a previous version's prereg.json is only moved."""
    n80_root, out_root = Path(n80_root), Path(out_root)
    out = []
    state, _ = previous_prereg(n80_root)
    if state == "mine":
        out.append(f"{K.PREREG_NAME} of {K.VERSION} exists already (the final version is written once)")
    for pattern in ("approval_near80_*.json", "import_near80_*.json"):
        for p in sorted(n80_root.glob(pattern)):
            if p.name in K.V8_HISTORY_FILES:
                continue
            out.append(f"{p.name} exists (an approval or import record of another fingerprint than version 8's history)")
    if out_root.exists():
        others = sorted(p.name for p in out_root.iterdir() if p.name != "pre")
        if others:
            out.append(f"official outputs exist under {out_root}: {others}")
    return out


def _lines(path):
    return Path(path).read_text(encoding="utf8").split("\n")


def first_docstring_line(path):
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None
    doc = ast.get_docstring(tree, clean=False)
    return doc.strip().split("\n")[0] if doc and doc.strip() else ""


def document_problems(n80_root):
    """Step 2: the documents and the code's first docstring lines are final."""
    n80_root = Path(n80_root)
    out = []
    doc, readme = n80_root / "PREREGISTRATION.md", n80_root / "README.md"
    for p in (doc, readme):
        if not p.is_file():
            out.append(f"{p.name} is missing")
    if out:
        return out
    dl, rl = _lines(doc), _lines(readme)
    if dl[0] != TITLE:
        out.append(f"PREREGISTRATION.md line 1 is not {TITLE!r}")
    if len(dl) < 3 or dl[2] != BOLD_FINAL:
        out.append(f"PREREGISTRATION.md line 3 is not {BOLD_FINAL!r}")
    if len(rl) < 3 or rl[2] != BOLD_FINAL:
        out.append(f"README.md line 3 is not {BOLD_FINAL!r}")
    for p in (doc, readme):
        text = p.read_text(encoding="utf8")
        if "**DRAFT" in text:
            out.append(f"{p.name} still says **DRAFT")
        if "this draft" in text.lower():
            out.append(f"{p.name} still says 'this draft'")
    if not any(line.startswith(FILL_RULES_HEADING) for line in dl):
        out.append(f"PREREGISTRATION.md has no section {FILL_RULES_HEADING[4:]}")
    code = n80_root / "code"
    if code.is_dir():
        for p in sorted(code.glob("*.py")):
            line = first_docstring_line(p)
            if line is None:
                out.append(f"code/{p.name} cannot be parsed")
            elif "DRAFT" in line:
                out.append(f"code/{p.name} says DRAFT in its first docstring line")
    return out


def code_folder_problems(n80_root):
    """Step 3: exactly the planned code files, no __pycache__, every top-level file."""
    n80_root = Path(n80_root)
    out = []
    code = n80_root / "code"
    if not code.is_dir():
        return ["code/ is missing"]
    for p in sorted(code.iterdir()):
        if p.name not in K.CODE_FILES or not p.is_file():
            out.append(f"code/{p.name} is not one of the planned code files")
    for name in K.CODE_FILES:
        if not (code / name).is_file():
            out.append(f"code/{name} is missing")
    for p in sorted(n80_root.rglob("__pycache__")):
        out.append(f"{p.relative_to(n80_root)} exists")
    for name in K.TOP_FILES:
        if not (n80_root / name).is_file():
            out.append(f"{name} is missing")
    return out


# --------------------------------------------------------------------------------------------- step 1: supersede

def supersede_previous(n80_root, log=print):
    """Step 1: a prereg.json of another version is moved (os.rename, with its sidecar) into prereg_history/ as
    prereg_superseded_<first 16 of its SHA-256>.json. Returns the relative path or None."""
    n80_root = Path(n80_root)
    state, body = previous_prereg(n80_root)
    if state != "previous":
        return None
    src = n80_root / K.PREREG_NAME
    sha = AC.file_sha256(src)
    side = Path(str(src) + ".sha256")
    if AC.sidecar_digest(src) != sha:
        raise FreezeRefused(f"{K.PREREG_NAME} ({body.get('version')}) differs from its sidecar; nothing moved")
    hist = n80_root / K.HISTORY_DIR
    dst = hist / K.SUPERSEDED_NAME.format(prefix=sha[:K.PREFIX_LEN])
    dside = Path(str(dst) + ".sha256")
    if dst.exists() or dside.exists():
        raise FreezeRefused(f"{dst} exists already")
    hist.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)
    os.rename(side, dside)
    if AC.file_sha256(dst) != sha or AC.sidecar_digest(dst) != sha:
        raise FreezeRefused(f"{dst} did not keep its bytes in the move")
    log(f"superseded {body.get('version')}: {K.PREREG_NAME} -> {K.HISTORY_DIR}/{dst.name} (+ .sha256)")
    return f"{K.HISTORY_DIR}/{dst.name}"


# --------------------------------------------------------------------------------------------- step 4: the draft

def draft_state(n80_root):
    """('move', path, sha) | ('kept', path, sha); raises FreezeRefused when neither holds."""
    n80_root = Path(n80_root)
    top = n80_root / DRAFT_NAME
    hist = n80_root / K.HISTORY_DIR
    kept = []
    if hist.is_dir():
        for p in sorted(hist.glob(K.DRAFT_HISTORY_NAME.format(prefix="*"))):
            if p.suffix != ".json":
                continue
            try:
                if json.loads(p.read_text(encoding="utf8")).get("version") == K.VERSION:
                    kept.append(p)
            except ValueError:
                continue
    if top.exists() and kept:
        raise FreezeRefused(f"both {DRAFT_NAME} and {[p.name for p in kept]} exist; exactly one draft is kept")
    if top.exists():
        raw = top.read_bytes()
        try:
            body = json.loads(raw.decode("utf8"))
        except (UnicodeDecodeError, ValueError):
            raise FreezeRefused(f"{DRAFT_NAME} is not JSON") from None
        if body.get("kind") != NP.DRAFT_KIND or body.get("status") != "draft" or body.get("version") != K.VERSION:
            raise FreezeRefused(f"{DRAFT_NAME} is not the {K.VERSION} draft")
        return "move", top, AC.sha256_bytes(raw)
    if len(kept) != 1:
        raise FreezeRefused(f"no draft to freeze: {DRAFT_NAME} is missing and {hist} holds {len(kept)} drafts of "
                            f"{K.VERSION}")
    try:
        NP.find_draft(n80_root)
    except (NP.PreregProblem, AC.ApprovalRefused) as error:
        raise FreezeRefused(f"the draft in {K.HISTORY_DIR}/ does not check: {error}") from None
    return "kept", kept[0], AC.file_sha256(kept[0])


def move_draft(n80_root, log=print):
    """Step 4: os.rename into prereg_history/ with a sidecar; a draft already there is checked, not moved."""
    state, path, sha = draft_state(n80_root)
    hist = Path(n80_root) / K.HISTORY_DIR
    if state == "kept":
        log(f"the draft is already kept: {K.HISTORY_DIR}/{path.name} (checked)")
        return f"{K.HISTORY_DIR}/{path.name}", sha
    dst = hist / K.DRAFT_HISTORY_NAME.format(prefix=sha[:K.PREFIX_LEN])
    side = Path(str(dst) + ".sha256")
    if dst.exists() or side.exists():
        raise FreezeRefused(f"{dst} exists already")
    hist.mkdir(parents=True, exist_ok=True)
    os.rename(path, dst)
    AC.write_once(side, f"{sha}  {dst.name}\n".encode("utf8"))
    if AC.file_sha256(dst) != sha:
        raise FreezeRefused(f"{dst} did not keep its bytes in the move")
    log(f"moved {DRAFT_NAME} -> {K.HISTORY_DIR}/{dst.name} (+ .sha256)")
    return f"{K.HISTORY_DIR}/{dst.name}", sha


# --------------------------------------------------------------------------------------------- step 5: the manifest

def manifest_files(n80_root):
    """Every near-80 file the manifest lists: the top-level files and code/<every planned file>."""
    n80_root = Path(n80_root)
    files = {}
    for rel in list(K.TOP_FILES) + ["code/" + n for n in K.CODE_FILES]:
        p = n80_root / rel
        if not p.is_file():
            raise FreezeRefused(f"{rel} is missing")
        files[rel] = AC.file_sha256(p)
    return dict(sorted(files.items()))


def repository_files(repo=K.ROOT):
    """Every file under PIN_DIRS with a PIN_SUFFIXES suffix (not under __pycache__), plus PIN_FILES; repo-relative."""
    repo = Path(repo)
    out = {}
    for d in K.PIN_DIRS:
        base = repo / d
        if not base.is_dir():
            raise FreezeRefused(f"the pinned folder {d} is missing")
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(repo)
            if p.is_file() and p.suffix in K.PIN_SUFFIXES and "__pycache__" not in rel.parts:
                out[rel.as_posix()] = AC.file_sha256(p)
    for rel in K.PIN_FILES:
        p = repo / rel
        if not p.is_file():
            raise FreezeRefused(f"the pinned file {rel} is missing")
        out[rel] = AC.file_sha256(p)
    return dict(sorted(out.items()))


def _replace_json(path, payload):
    """Write strict sorted JSON and its sidecar through temporary files and os.replace (only for files nothing pins
    yet: the manifest before prereg.json exists, the pre-approval copy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, bytes) else AC.canonical(payload).encode("utf8")
    digest = AC.sha256_bytes(text)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text)
    os.replace(tmp, path)
    side = Path(str(path) + ".sha256")
    stmp = side.with_name(side.name + ".tmp")
    stmp.write_text(f"{digest}  {path.name}\n", encoding="utf8")
    os.replace(stmp, side)
    return digest


def write_manifest(n80_root, reason, repo=K.ROOT, log=print):
    """Step 5. Returns (SHA-256, manifest)."""
    n80_root = Path(n80_root)
    files = manifest_files(n80_root)
    repo_files = repository_files(repo)
    path = n80_root / K.MANIFEST_NAME
    history = []
    if path.exists():
        old = AC.read_json_verified(path, K.MANIFEST_NAME)
        if old.get("files") == files and old.get("repository_files") == repo_files and old.get("label") == K.LABEL:
            log(f"{K.MANIFEST_NAME} is unchanged ({AC.file_sha256(path)[:16]}); kept")
            return AC.file_sha256(path), old
        hist = n80_root / K.MANIFEST_HISTORY_DIR
        n = len(old.get("history") or []) + 1
        dst = hist / f"code_manifest_v{n}.json"
        while dst.exists():
            n += 1
            dst = hist / f"code_manifest_v{n}.json"
        hist.mkdir(parents=True, exist_ok=True)
        old_sha = AC.file_sha256(path)
        shutil.copyfile(path, dst)
        AC.write_once(Path(str(dst) + ".sha256"), f"{old_sha}  {dst.name}\n".encode("utf8"))
        history = list(old.get("history") or []) + [dict(version=n, sha256=old_sha, reason=old.get("reason"),
                                                         label=old.get("label"),
                                                         archived=f"{K.MANIFEST_HISTORY_DIR}/{dst.name}")]
        log(f"archived the earlier manifest ({old.get('label', '')[:40]}...) as {K.MANIFEST_HISTORY_DIR}/{dst.name}")
    body = dict(kind=K.MANIFEST_KIND, label=K.LABEL, version=K.VERSION, files=files, repository_files=repo_files,
                reason=reason, history=history, written=AC.utc_now(),
                rule=("every near-80 phase, the import and the owner's approval script refuse unless each file here "
                      "has this SHA-256 and this manifest's SHA-256 is the one prereg.json pins"))
    digest = _replace_json(path, body)
    log(f"wrote {K.MANIFEST_NAME} ({digest[:16]}): {len(files)} near-80 files, {len(repo_files)} repository files")
    return digest, json.loads(AC.canonical(body))


# --------------------------------------------------------------------------------------------- steps 6-7

def _clean(env, cmd):
    return ["env", "-i"] + [f"{k}={v}" for k, v in env.items()] + cmd if os.name == "posix" else cmd


def run_unit_tests(n80_root, log=print):
    """Step 6: each test file alone, from the work folder, with the phases' environment. Returns {name: exit status}."""
    results = {}
    env = dict(K.PHASE_ENV)
    for name in K.TEST_FILES:
        cmd = _clean(env, [K.PYTHON, str(Path(n80_root) / "code" / name)])
        try:
            done = subprocess.run(cmd, cwd=LG.work_dir(), env=None if os.name == "posix" else env,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=TEST_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as error:
            log(f"{name}: could not run ({error})")
            results[name] = None
            continue
        results[name] = done.returncode
        tail = done.stdout.decode("utf8", "replace").strip().split("\n")[-3:]
        log(f"{name}: exit {done.returncode}: {' | '.join(tail)}")
        if done.returncode != 0:
            log("\n".join(done.stdout.decode("utf8", "replace").strip().split("\n")[-60:]))
    return results


def run_facts(n80_root, out_path, log=print):
    """Step 7: facts_near80.py on seeds 0-4, both worlds, with the frozen code (nothing trained)."""
    cmd = _clean(dict(K.PHASE_ENV), [K.PYTHON, str(Path(n80_root) / "code" / "facts_near80.py"), *FACTS_ARGS, "--out",
                                     str(out_path)])
    log("running: " + " ".join(cmd))
    done = subprocess.run(cmd, cwd=LG.work_dir(), env=None if os.name == "posix" else dict(K.PHASE_ENV),
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=FACTS_TIMEOUT_S)
    log(done.stdout.decode("utf8", "replace").strip())
    if done.returncode != 0:
        raise FreezeRefused(f"facts_near80.py exited with {done.returncode}")


def facts_path(out_root, manifest_sha256):
    return Path(out_root) / "pre" / FACTS_NAME.format(prefix=manifest_sha256[:K.PREFIX_LEN])


def _t_lo_in_range(t_lo):
    """Section 8.1, check 4, on a recorded t_lo (standard library only: freeze.py imports no numpy)."""
    try:
        value = float(t_lo)
    except (TypeError, ValueError):
        return False
    lo, hi = (float(v) for v in K.T_LO_RANGE)
    return value == value and lo <= value <= hi


def facts_problems(payload):
    """The facts record is this version's: the counts on seeds 0-4 and both worlds with every seed-world counted,
    its gap rule defined with t_lo in range and its leak checks passed (section 8.1, checks 3 and 4), and the
    spent-seed switch facts on seeds 10-14 and both worlds with t_lo in range and the kill rule passed on all 10
    (section 8.1, check 5; a failure means a redraft, not a freeze)."""
    out = []
    if not isinstance(payload, dict):
        return ["the facts record is not a JSON object"]
    if payload.get("kind") != FACTS_KIND:
        out.append(f"the facts record is a {payload.get('kind')!r}, not the pre-approval facts")
    if payload.get("seeds") != [0, 1, 2, 3, 4]:
        out.append(f"the counts ran on seeds {payload.get('seeds')}, not 0-4")
    if payload.get("worlds") != list(K.WORLDS):
        out.append(f"the facts ran on worlds {payload.get('worlds')}, not both")
    results = payload.get("results") or []
    if len(results) != 10:
        out.append(f"the counts hold {len(results)} seed-worlds, not 10")
    for r in results:
        if not isinstance(r, dict) or r.get("defined") is not True or not r.get("groups"):
            out.append(f"seed {r.get('rep')} ({r.get('world')}): the gap rule is not defined or the groups are missing")
        elif not _t_lo_in_range((r.get("gap_rule") or {}).get("t_lo")):
            out.append(f"seed {r.get('rep')} ({r.get('world')}): t_lo {(r.get('gap_rule') or {}).get('t_lo')} is "
                       f"outside {list(K.T_LO_RANGE)} (section 8.1, check 4)")
        elif not (r.get("leak_checks") or {}).get("passed"):
            out.append(f"seed {r.get('rep')} ({r.get('world')}): the pooled groups' leak checks are not recorded as "
                       f"passed (section 8.1, check 3)")
    if payload.get("spent_seeds") != list(K.SPENT_SEEDS):
        out.append(f"the spent-seed facts ran on seeds {payload.get('spent_seeds')}, not {list(K.SPENT_SEEDS)}")
    spent = payload.get("spent_results") or []
    if len(spent) != SPENT_SEED_WORLDS:
        out.append(f"the spent-seed facts hold {len(spent)} seed-worlds, not {SPENT_SEED_WORLDS}")
    for r in spent:
        kr = (r or {}).get("kill_rule") if isinstance(r, dict) else None
        if not isinstance(kr, dict) or kr.get("passed") is not True:
            out.append(f"spent seed {(r or {}).get('seed')} ({(r or {}).get('world')}): the kill rule did not pass "
                       f"(the pooled calls differ from version 8's recorded calls; a redraft, not a freeze)")
        elif not _t_lo_in_range(((r or {}).get("gap_rule") or {}).get("t_lo")):
            out.append(f"spent seed {r.get('seed')} ({r.get('world')}): t_lo {(r.get('gap_rule') or {}).get('t_lo')} "
                       f"is outside {list(K.T_LO_RANGE)} (section 8.1, check 4)")
    ss = payload.get("spent_summary")
    if not isinstance(ss, dict) or ss.get("kill_rule_passed") is not True:
        out.append("the spent-seed summary does not say the kill rule passed on every spent seed-world "
                   "(spent_summary.kill_rule_passed is not true)")
    return out


def copy_frozen(n80_root, raw):
    """The facts record's bytes as pre_approval/<K.COUNTS_FROZEN_NAME> (+ .sha256). Returns the record."""
    dst = Path(n80_root) / K.PRE_APPROVAL_DIR / K.COUNTS_FROZEN_NAME
    digest = _replace_json(dst, raw)
    if AC.file_sha256(dst) != digest or AC.sidecar_digest(dst) != digest:
        raise FreezeRefused(f"{dst} did not read back as written")
    return {"path": NP.FROZEN_REL, "sha256": digest, "payload": json.loads(raw.decode("utf8"))}


# --------------------------------------------------------------------------------------------- check and write

def _load_approve_local(n80_root):
    path = Path(n80_root) / "approve_local_near80.py"
    name = f"near80_freeze_approve_local_{AC.sha256_bytes(str(path).encode())[:12]}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def problems(n80_root=K.N80, out_root=K.OUT, claims_fn=None):
    """Every problem steps 0-4 would refuse on, and whether the claims can be extracted."""
    out = (placeholder_problems(n80_root) + existing_problems(n80_root, out_root) + document_problems(n80_root)
           + code_folder_problems(n80_root))
    try:
        draft_state(n80_root)
    except FreezeRefused as error:
        out.append(str(error))
    if claims_fn is None and not (Path(n80_root) / "code" / "claims_text.py").is_file():
        out.append("code/claims_text.py is missing: the claims cannot be copied word for word")
    return out


def check(n80_root=K.N80, out_root=K.OUT, log=print):
    """'freeze.py check': what write would do, and every problem. Writes nothing."""
    n80_root, out_root = Path(n80_root), Path(out_root)
    found = problems(n80_root, out_root)
    try:
        state, path, sha = draft_state(n80_root)
        draft_line = (f"move {DRAFT_NAME} to {K.HISTORY_DIR}/{K.DRAFT_HISTORY_NAME.format(prefix=sha[:16])}"
                      if state == "move" else f"keep {K.HISTORY_DIR}/{path.name} (checked)")
    except FreezeRefused:
        draft_line = "(no draft to freeze)"
    prev, body = previous_prereg(n80_root)
    log("freeze.py write would:\n"
        f"  0. refuse while any DECISION.md placeholder remains ({len(K.placeholders_present())} now)\n"
        f"  1. {'move ' + K.PREREG_NAME + ' of ' + str(body.get('version')) + ' into ' + K.HISTORY_DIR + '/' if prev == 'previous' else 'check that nothing of this version is approved or run yet'}\n"
        f"  2-3. check that the documents are final and code/ holds exactly {len(K.CODE_FILES)} planned files\n"
        f"  4. {draft_line}\n"
        f"  5. write {K.MANIFEST_NAME} (+ .sha256): {len(K.TOP_FILES) + len(K.CODE_FILES)} near-80 files and the "
        f"repository files under {len(K.PIN_DIRS)} pinned folders\n"
        f"  6. run {', '.join(K.TEST_FILES)} from the work folder\n"
        f"  7. run facts_near80.py {' '.join(FACTS_ARGS)} --out {out_root / 'pre'}/{FACTS_NAME} (nothing trained; "
        f"the kill rule on the spent seeds must pass) and copy it to {NP.FROZEN_REL} (+ .sha256)\n"
        f"  8. write {K.PREREG_NAME} (+ .sha256) and print its fingerprint")
    if found:
        log("Problems (write would refuse):\n  - " + "\n  - ".join(found))
    else:
        log("No problems found.")
    return found


# --------------------------------------------------------------------------------------------- check --as-frozen

def _drop_draft_phrase_in_docstring(text):
    """The module docstring (the first triple-quoted string, with anything before it) without the draft phrase; the
    rest of the file byte for byte (the constants that name the draft state keep their text)."""
    i = text.find('"""')
    j = text.find('"""', i + 3) if i >= 0 else -1
    if j < 0:
        return text
    return DRAFT_PHRASE.sub("frozen", text[:j]) + text[j:]


def _drop_draft_phrase_in_header(text):
    """A shell script's comment lines without the draft phrase; every other line byte for byte."""
    out = []
    for line in text.split("\n"):
        out.append(DRAFT_PHRASE.sub("frozen", line) if line.startswith("#") else line)
    return "\n".join(out)


def finalize_copy(n80_root, dst, log=print):
    """'check --as-frozen': a copy of the near-80 folder at dst (which must not exist) in the state write's steps 2-4
    leave it (module docstring). Returns the list of what was changed in the copy."""
    n80_root, dst = Path(n80_root), Path(dst)
    if dst.exists():
        raise FreezeRefused(f"{dst} exists; the as-frozen copy needs a fresh folder")
    shutil.copytree(n80_root, dst, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    notes = []
    for name in ("PREREGISTRATION.md", "README.md"):
        p = dst / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf8")
        lines = text.split("\n")
        if len(lines) > 2 and lines[2] == f"**{K.STATUS_LINE_DRAFT}**":
            lines[2] = BOLD_FINAL
            notes.append(f"{name}: the FINAL status line")
        text = "\n".join(lines)
        n = len(TODO_MARK.findall(text))
        if n:
            text = TODO_MARK.sub(AS_FROZEN_FILL, text)
            notes.append(f"{name}: {n} TODO(...) mark{'s' if n != 1 else ''} replaced by the stand-in")
        p.write_text(text, encoding="utf8")
    changed = []
    for p in sorted((dst / "code").glob("*.py")) if (dst / "code").is_dir() else []:
        text = p.read_text(encoding="utf8")
        new = _drop_draft_phrase_in_docstring(text)
        if new != text:
            p.write_text(new, encoding="utf8")
            changed.append("code/" + p.name)
    for name in AS_FROZEN_SCRIPTS:
        p = dst / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf8")
        new = _drop_draft_phrase_in_docstring(text) if name.endswith(".py") else _drop_draft_phrase_in_header(text)
        if new != text:
            p.write_text(new, encoding="utf8")
            changed.append(name)
    if changed:
        notes.append(f"the draft phrase dropped from the docstring or header of {len(changed)} files")
    rel, _ = move_draft(dst, log=lambda *_a, **_k: None)
    notes.append(f"the draft at {rel}")
    for note in notes:
        log(f"as-frozen copy: {note}")
    return notes


def check_as_frozen(n80_root=K.N80, out_root=K.OUT, log=print, keep=False, tests_fn=None, claims_fn=None):
    """'freeze.py check --as-frozen' (module docstring): finalize_copy, then check's problems, the extraction of
    section 14 and the five test files, all on the copy. Returns every problem found (empty when write may run).
    tests_fn and claims_fn replace the test run and the extraction in tests only."""
    n80_root, out_root = Path(n80_root), Path(out_root)
    tmp = Path(tempfile.mkdtemp(prefix="near80_as_frozen_"))
    dst = tmp / "near80"
    try:
        finalize_copy(n80_root, dst, log)
        found = problems(dst, out_root)
        if found:
            log("Problems on the finalized copy (write would refuse):\n  - " + "\n  - ".join(found))
        else:
            log("No problems found.")
        try:
            fn = claims_fn or NP.load_claims_text(dst / "code").extract_section14
            claims = fn((dst / "PREREGISTRATION.md").read_text(encoding="utf8"))
            marks = claims.get("todo_marks") or []
            log(f"section 14 extracts: {len(claims.get('sentences') or {})} sentences, {len(marks)} TODO(...) marks")
            if marks:
                found.append(f"section 14 of the finalized copy carries TODO(...) marks: {marks}")
        except Exception as error:  # noqa: BLE001 - reported as a problem, never raised past the copy's removal
            found.append(f"section 14 of the finalized copy does not extract: {error}")
        log(f"running {', '.join(K.TEST_FILES)} on the finalized copy ({dst}) from the work folder")
        results = (tests_fn or run_unit_tests)(dst, log)
        bad = sorted(n for n in K.TEST_FILES if results.get(n) != 0)
        if bad:
            found.append(f"unit tests failed on the finalized copy: {bad}")
        else:
            log("the five test files pass on the finalized copy")
        if found:
            log("as-frozen: NOT ready for write:\n  - " + "\n  - ".join(found))
        else:
            log("as-frozen: ready for write (no problem on the finalized copy; every test file passed there)")
        return found
    finally:
        if keep:
            log(f"the finalized copy is kept at {dst}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def write(reason, facts_from=None, n80_root=K.N80, out_root=K.OUT, repo=K.ROOT, tests_fn=None, facts_fn=None,
          claims_fn=None, log=print):
    """'freeze.py write' (module docstring). tests_fn(n80_root, log) -> {name: exit status} and
    facts_fn(n80_root, out_path, log) (writes out_path) replace steps 6 and 7 in tests only. Returns prereg.json's
    payload."""
    n80_root, out_root = Path(n80_root), Path(out_root)
    if not str(reason or "").strip():
        raise FreezeRefused("a reason is required (--reason)")
    found = problems(n80_root, out_root, claims_fn)
    if found:
        raise FreezeRefused("refused before writing anything:\n  - " + "\n  - ".join(found))
    if claims_fn is None:
        claims_fn = NP.load_claims_text(n80_root / "code").extract_section14
    claims_fn((n80_root / "PREREGISTRATION.md").read_text(encoding="utf8"))      # refuses early if section 14 is off
    supersede_previous(n80_root, log)
    move_draft(n80_root, log)
    msha, manifest = write_manifest(n80_root, reason, repo, log)
    results = (tests_fn or run_unit_tests)(n80_root, log)
    bad = sorted(n for n in K.TEST_FILES if results.get(n) != 0)
    if bad:
        raise FreezeRefused(f"unit tests failed: {bad}")
    want = facts_path(out_root, msha)
    if facts_from is not None:
        src = Path(facts_from)
        if src.name != want.name:
            raise FreezeRefused(f"{src.name} does not carry this manifest's first 16 characters ({want.name})")
        if not src.is_file():
            raise FreezeRefused(f"{src} is missing")
    elif want.exists():
        src = want
        log(f"reusing {want} (it carries this manifest's first 16 characters)")
    else:
        want.parent.mkdir(parents=True, exist_ok=True)
        (facts_fn or run_facts)(n80_root, want, log)
        src = want
    raw = src.read_bytes()
    try:
        payload = json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise FreezeRefused(f"{src} is not JSON") from None
    found = facts_problems(payload)
    if found:
        raise FreezeRefused("the frozen code's pre-approval facts are not usable: " + "; ".join(found))
    if manifest_files(n80_root) != manifest["files"] or repository_files(repo) != manifest["repository_files"]:
        raise FreezeRefused("a pinned file changed while the tests or the counts ran; run write again")
    facts_record = copy_frozen(n80_root, raw)
    body = NP.build(n80_root, msha, manifest, facts_record, repo=repo, claims_fn=claims_fn)
    digest = AC.write_json_once(n80_root / K.PREREG_NAME, body)
    prefix = digest[:K.PREFIX_LEN]
    try:                                    # the owner's own script must accept what was written
        al = _load_approve_local(n80_root)
        al.check_folder(str(n80_root), str(n80_root / "approve_local_near80.py"), str(repo))
    except Exception as error:  # noqa: BLE001 - any failure moves the file aside and refuses
        hist = n80_root / K.HISTORY_DIR
        for src in (n80_root / K.PREREG_NAME, n80_root / (K.PREREG_NAME + ".sha256")):
            os.rename(src, hist / src.name.replace("prereg.json", f"prereg_rejected_{prefix}.json"))
        raise FreezeRefused(f"approve_local_near80.py would refuse the written prereg.json ({error}); it was moved "
                            f"to {K.HISTORY_DIR}/prereg_rejected_{prefix}.json") from None
    log(f"wrote {n80_root / K.PREREG_NAME} (final, awaiting approval)\n  SHA-256: {digest}\n"
        f"  first 16 characters: {prefix}\n"
        "Next: commit and push prereg.json, code_manifest.json (each with its .sha256), prereg_history/, "
        "code_manifest_history/, pre_approval/, the code and the documents. Then tell the owner the fingerprint and "
        "the command of section 11:\n"
        f"  python3 -I {K.N80_REL}/approve_local_near80.py --check\n  python3 -I {K.N80_REL}/approve_local_near80.py\n"
        f"Nothing trains before {K.IMPORT_NAME.format(prefix=prefix)} is pushed.")
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("check")
    c.add_argument("--as-frozen", action="store_true",
                   help="check and run the five test files on a temporary finalized copy (module docstring)")
    c.add_argument("--keep", action="store_true", help="with --as-frozen: keep the copy for inspection")
    w = sub.add_parser("write")
    w.add_argument("--reason", required=True)
    w.add_argument("--facts-from", default=None)
    a = ap.parse_args(argv)
    try:
        if a.mode == "check":
            if a.as_frozen:
                return 2 if check_as_frozen(keep=a.keep) else 0
            return 2 if check() else 0
        write(a.reason, facts_from=a.facts_from)
        return 0
    except (FreezeRefused, AC.ApprovalRefused, NP.PreregProblem) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
