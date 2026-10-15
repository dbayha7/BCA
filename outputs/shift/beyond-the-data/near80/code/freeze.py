"""Near-80 (version 8): freeze the final version (prereg.json, its code manifest and the frozen no-training checks).

FINAL, awaiting the owner's approval. Nothing here trains a learner, fits a BCA band or reads a result. The main
session runs it once, after the three parts of the code are integrated:

  python freeze.py check                                 prints what 'write' would do and every problem; writes nothing
  python freeze.py write --reason TEXT [--facts-from PATH]

'write', in order (any problem stops it before the next step):
  1. Refuses if prereg.json, any approval_near80_*.json or import_near80_*.json exists, or any official output under
     runs/shift_explore/near80/ other than pre/.
  2. The documents are final: PREREGISTRATION.md's title and status line, README.md's status line, no "**DRAFT" and no
     "this draft" in either, a section 14.6; no file under code/ says DRAFT in its first docstring line.
  3. The code folder holds exactly the planned files (near80_constants.CODE_FILES), no __pycache__, and the top-level
     files (near80_constants.TOP_FILES: the documents, run_near80.sh, push_runfiles_near80.sh and
     approve_local_near80.py) exist.
  4. prereg_draft.json is renamed into prereg_history/prereg_draft_<first 16 of its SHA-256>.json with a sidecar (a
     rerun checks it there instead).
  5. code_manifest.json (+ .sha256): every near-80 file and every repository file the code imports or executes; an
     earlier manifest is archived in code_manifest_history/ (a rerun with no change keeps it as it is).
  6. The four unit test files, each run alone from /tmp with the phases' environment; all must pass.
  7. The frozen code's no-training checks on seeds 0-4 (facts_near80.py --rollouts, about 5 CPU-minutes, nothing
     trained), written to runs/shift_explore/near80/pre/no_training_checks_<first 16 of the manifest>.json and copied
     to pre_approval/no_training_checks_seeds0_4.json (+ .sha256). A failure stops the freeze with section 14.5's
     sentence: no approval is asked and the design is redrafted.
  8. prereg.json (+ .sha256) from near80_prereg.build; prints the fingerprint and the next steps.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import approval_check as AC  # noqa: E402
import near80_constants as K  # noqa: E402
import near80_prereg as NP  # noqa: E402

TITLE = "# Hiding the commands near 80%: pre-registration (version 8, near-80)"
BOLD_FINAL = "**FINAL, awaiting the owner's approval. Nothing has been trained.**"
DRAFT_NAME = "prereg_draft.json"
FACTS_ARGS = ("--rollouts", "--seeds", "0", "1", "2", "3", "4", "--worlds", "stall", "no_stall")
FACTS_NAME = "no_training_checks_{prefix}.json"
TEST_TIMEOUT_S = 3 * 3600
FACTS_TIMEOUT_S = 4 * 3600


class FreezeRefused(Exception):
    """A freeze step refused; what was written before it stays (nothing official is written before step 8)."""


# --------------------------------------------------------------------------------------------- steps 1-3: checks

def existing_problems(n80_root, out_root):
    """Step 1: nothing approved, imported or run yet."""
    n80_root, out_root = Path(n80_root), Path(out_root)
    out = []
    for name in (K.PREREG_NAME, K.PREREG_NAME + ".sha256"):
        if (n80_root / name).exists():
            out.append(f"{name} exists already (the final version is written once)")
    for pattern in ("approval_near80_*", "import_near80_*"):
        for p in sorted(n80_root.glob(pattern)):
            out.append(f"{p.name} exists (an approval or import record freezes the code)")
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
    if not any(line.startswith("### 14.6") for line in dl):
        out.append("PREREGISTRATION.md has no section 14.6")
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
    """Step 3: exactly the planned code files, no __pycache__, both top-level scripts."""
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


# --------------------------------------------------------------------------------------------- step 4: the draft

def draft_state(n80_root):
    """('move', path, sha) | ('kept', path, sha); raises FreezeRefused when neither holds."""
    n80_root = Path(n80_root)
    top = n80_root / DRAFT_NAME
    hist = n80_root / K.HISTORY_DIR
    kept = sorted(p for p in hist.glob(K.DRAFT_HISTORY_NAME.format(prefix="*")) if p.suffix == ".json") \
        if hist.is_dir() else []
    if top.exists() and kept:
        raise FreezeRefused(f"both {DRAFT_NAME} and {[p.name for p in kept]} exist; exactly one draft is kept")
    if top.exists():
        raw = top.read_bytes()
        try:
            body = json.loads(raw.decode("utf8"))
        except (UnicodeDecodeError, ValueError):
            raise FreezeRefused(f"{DRAFT_NAME} is not JSON") from None
        if body.get("kind") != NP.DRAFT_KIND or body.get("status") != "draft":
            raise FreezeRefused(f"{DRAFT_NAME} is not the near-80 draft")
        return "move", top, AC.sha256_bytes(raw)
    if len(kept) != 1:
        raise FreezeRefused(f"no draft to freeze: {DRAFT_NAME} is missing and {hist} holds {len(kept)} drafts")
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
    """Every near-80 file the manifest lists: the four top-level files and code/<every planned file>."""
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
        if old.get("files") == files and old.get("repository_files") == repo_files:
            log(f"{K.MANIFEST_NAME} is unchanged ({AC.file_sha256(path)[:16]}); kept")
            return AC.file_sha256(path), old
        hist = n80_root / K.MANIFEST_HISTORY_DIR
        n = len(old.get("history") or []) + 1
        dst = hist / f"code_manifest_v{n}.json"
        if dst.exists():
            raise FreezeRefused(f"{dst} exists already")
        hist.mkdir(parents=True, exist_ok=True)
        old_sha = AC.file_sha256(path)
        shutil.copyfile(path, dst)
        AC.write_once(Path(str(dst) + ".sha256"), f"{old_sha}  {dst.name}\n".encode("utf8"))
        history = list(old.get("history") or []) + [dict(version=n, sha256=old_sha, reason=old.get("reason"),
                                                         archived=f"{K.MANIFEST_HISTORY_DIR}/{dst.name}")]
        log(f"archived the earlier manifest as {K.MANIFEST_HISTORY_DIR}/{dst.name}")
    body = dict(kind=K.MANIFEST_KIND, label=K.LABEL, files=files, repository_files=repo_files, reason=reason,
                history=history, written=AC.utc_now(),
                rule=("every near-80 phase, the import and the owner's approval script refuse unless each file here "
                      "has this SHA-256 and this manifest's SHA-256 is the one prereg.json pins"))
    digest = _replace_json(path, body)
    log(f"wrote {K.MANIFEST_NAME} ({digest[:16]}): {len(files)} near-80 files, {len(repo_files)} repository files")
    return digest, json.loads(AC.canonical(body))


# --------------------------------------------------------------------------------------------- steps 6-7

def run_unit_tests(n80_root, log=print):
    """Step 6: each test file alone, from /tmp, with the phases' environment. Returns {name: exit status}."""
    results = {}
    for name in K.TEST_FILES:
        cmd = [K.PYTHON, str(Path(n80_root) / "code" / name)]
        try:
            done = subprocess.run(cmd, cwd="/tmp", env=dict(K.PHASE_ENV), stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=TEST_TIMEOUT_S)
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
    """Step 7: facts_near80.py --rollouts on seeds 0-4, both worlds, with the frozen code (nothing trained)."""
    cmd = [K.PYTHON, str(Path(n80_root) / "code" / "facts_near80.py"), *FACTS_ARGS, "--out", str(out_path)]
    log("running: " + " ".join(cmd))
    done = subprocess.run(cmd, cwd="/tmp", env=dict(K.PHASE_ENV), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=FACTS_TIMEOUT_S)
    log(done.stdout.decode("utf8", "replace").strip())
    if done.returncode != 0:
        raise FreezeRefused(f"facts_near80.py exited with {done.returncode}")


def facts_path(out_root, manifest_sha256):
    return Path(out_root) / "pre" / FACTS_NAME.format(prefix=manifest_sha256[:K.PREFIX_LEN])


def facts_problems(payload):
    """The facts record is the default design's, on seeds 0-4 and both worlds, with every rule run and passed."""
    out = []
    if not isinstance(payload, dict):
        return ["the facts record is not a JSON object"]
    nt = payload.get("no_training_checks") or {}
    if payload.get("seeds") != list(K.EXPLORATORY):
        out.append(f"the facts ran on seeds {payload.get('seeds')}, not 0-4")
    if payload.get("worlds") != list(K.WORLDS):
        out.append(f"the facts ran on worlds {payload.get('worlds')}, not both")
    band = payload.get("band") or {}
    if (band.get("lo"), band.get("hi"), band.get("scope")) != (K.BAND_LO, K.BAND_HI, K.BAND_SCOPE):
        out.append(f"the facts ran on another band ({band})")
    if nt.get("not_run") != []:
        out.append(f"no-training checks not run: {nt.get('not_run')}")
    if nt.get("passed") is not True:
        failed = [r.get("rule") + f" ({r.get('scope')})" for r in nt.get("rules") or [] if not r.get("passed")]
        out.append(f"no-training checks failed: {failed}")
    return out


def default_sentence(rules):
    """Section 14.5's no-training stop sentence, through C's pooled.halt_sentence_no_training (a plain fallback
    names the first failed rule if the looks' code cannot be loaded)."""
    try:
        return K.load_local("pooled").halt_sentence_no_training(rules)
    except Exception:  # noqa: BLE001 - the fallback below still names the failed check
        first = next((r for r in rules if not r.get("passed")), None)
        if first is None:
            return "On seeds 0-4, before any training, a no-training check was not run. Nothing was trained."
        return (f"On seeds 0-4, before any training, the near-80 data failed {first.get('rule')} "
                f"({first.get('value')}; needed {first.get('threshold')}). The design was redrafted. Nothing was "
                f"trained.")


def copy_frozen(n80_root, raw):
    """The facts record's bytes as pre_approval/no_training_checks_seeds0_4.json (+ .sha256). Returns the record."""
    dst = Path(n80_root) / K.PRE_APPROVAL_DIR / K.NO_TRAINING_FROZEN_NAME
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
    """Every problem steps 1-4 would refuse on, and whether the claims can be extracted."""
    out = existing_problems(n80_root, out_root) + document_problems(n80_root) + code_folder_problems(n80_root)
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
    log("freeze.py write would:\n"
        f"  1-3. check that nothing is approved or run yet, the documents are final and code/ holds exactly "
        f"{len(K.CODE_FILES)} planned files\n"
        f"  4. {draft_line}\n"
        f"  5. write {K.MANIFEST_NAME} (+ .sha256): {len(K.TOP_FILES) + len(K.CODE_FILES)} near-80 files and the "
        f"repository files under {len(K.PIN_DIRS)} pinned folders\n"
        f"  6. run {', '.join(K.TEST_FILES)} from /tmp\n"
        f"  7. run facts_near80.py {' '.join(FACTS_ARGS)} --out {out_root / 'pre'}/{FACTS_NAME} and copy it to "
        f"{NP.FROZEN_REL} (+ .sha256)\n"
        f"  8. write {K.PREREG_NAME} (+ .sha256) and print its fingerprint")
    if found:
        log("Problems (write would refuse):\n  - " + "\n  - ".join(found))
    else:
        log("No problems found.")
    return found


def write(reason, facts_from=None, n80_root=K.N80, out_root=K.OUT, repo=K.ROOT, tests_fn=None, facts_fn=None,
          claims_fn=None, sentence_fn=None, log=print):
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
        rules = (payload.get("no_training_checks") or {}).get("rules") or [] if isinstance(payload, dict) else []
        sentence = (sentence_fn or default_sentence)(rules)
        log(sentence)
        raise FreezeRefused("the frozen code's no-training checks on seeds 0-4 did not pass, so no approval is asked: "
                            + "; ".join(found) + "\n" + sentence)
    if manifest_files(n80_root) != manifest["files"] or repository_files(repo) != manifest["repository_files"]:
        raise FreezeRefused("a pinned file changed while the tests or the no-training checks ran; run write again")
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
        "pre_approval/, the code and the documents. Then tell the owner the fingerprint and the command of "
        "section 11:\n"
        f"  python3 {K.N80_REL}/approve_local_near80.py\n"
        f"Nothing trains before {K.IMPORT_NAME.format(prefix=prefix)} is pushed.")
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("check")
    w = sub.add_parser("write")
    w.add_argument("--reason", required=True)
    w.add_argument("--facts-from", default=None)
    a = ap.parse_args(argv)
    try:
        if a.mode == "check":
            return 2 if check() else 0
        write(a.reason, facts_from=a.facts_from)
        return 0
    except (FreezeRefused, AC.ApprovalRefused, NP.PreregProblem) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
