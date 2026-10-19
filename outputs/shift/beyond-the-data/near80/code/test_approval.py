"""Unit tests of the near-80 approval chain and the freeze: the owner's approval script, the session's import, the
approval check every phase calls, the git identity refusal, the test fixture, near80_prereg and freeze.py, and the
values of near80_constants.

Everything runs on temporary copies outside the repository and on throwaway git repositories made with 'git init' in a
temporary folder (a bare "origin" and a clone; commits made with 'git -c user.name=... -c user.email=...'; git's global
and system configuration are replaced by an empty one for the tests). Never the real repository's git, never a push
anywhere but the throwaway origin. Nothing is trained; no recorded seed is generated, loaded or read. The freeze's unit
test runs and its no-training checks are injected as fakes (function arguments).

Run (plain unittest), from /tmp:
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \
    python /home/user/bca/outputs/shift/beyond-the-data/near80/code/test_approval.py
"""
import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import approval_check as AC  # noqa: E402
import import_approval as IA  # noqa: E402
import near80_prereg as NP  # noqa: E402
import freeze as FZ  # noqa: E402

APPROVER = f"{K.APPROVER_NAME} <{K.APPROVER_EMAIL}>"
AGENT = "Claude <noreply@anthropic.com>"
A_FILES = ("near80_constants.py", "near80lib.py", "facts_near80.py", "test_near80.py", "approval_check.py",
           "import_approval.py", "near80_prereg.py", "freeze.py", "test_approval.py")
CLAIMS_REAL = (K.CODE / "claims_text.py").is_file()


def quiet(*_a, **_k):
    return None


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AL = load_path("near80_test_approve_local", K.N80 / "approve_local_near80.py")


def write_json_sidecar(path, payload):
    """A test helper: canonical JSON and its sidecar, overwriting (only in temporary folders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = AC.canonical(payload).encode("utf8")
    digest = AC.sha256_bytes(text)
    path.write_bytes(text)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def tree(root):
    root = Path(root)
    return sorted((p.relative_to(root).as_posix(), AC.file_sha256(p) if p.is_file() else None)
                  for p in root.rglob("*"))


class Tty(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        raise OSError("no file descriptor (test)")


class NotTty(io.StringIO):
    def isatty(self):
        return False


def split_who(who):
    m = AC._AUTHOR_RE.fullmatch(who)
    return m.group(1), m.group(2)


def git(cwd, *args, committer=None, author=None, check=True):
    cmd = ["git", "-C", str(cwd)]
    if committer:
        name, email = split_who(committer)
        cmd += ["-c", f"user.name={name}", "-c", f"user.email={email}"]
    cmd += list(args)
    if author:
        cmd += ["--author", author]
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=dict(os.environ))
    if check and done.returncode != 0:
        raise AssertionError(f"git {args} failed: {done.stderr.decode()}")
    return done.stdout.decode("utf8", "replace").strip()


# ============================================================================================ a synthetic folder

DOC = (f"# Hiding the commands near 80%: pre-registration (version 8, near-80)\n\n**{K.STATUS_LINE_FINAL}**\n\n"
       "A synthetic document for a test.\n")


def build_n80(repo, status=K.STATUS_FINAL, frozen_passed=True):
    """A synthetic near-80 folder inside `repo` (the real repository's relative layout), with the repository files
    the pins name. Returns its prereg (AC.Prereg)."""
    repo = Path(repo)
    n80 = repo / K.N80_REL
    (n80 / "code").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(K.N80 / "approve_local_near80.py", n80 / "approve_local_near80.py")
    (n80 / "PREREGISTRATION.md").write_text(DOC, encoding="utf8")
    (n80 / "README.md").write_text(DOC, encoding="utf8")
    (n80 / "run_near80.sh").write_text("#!/bin/sh\necho synthetic\n", encoding="utf8")
    (n80 / "code" / "x.py").write_text('"""Synthetic code."""\nx = 1\n', encoding="utf8")
    files = {rel: AC.file_sha256(n80 / rel) for rel in ("PREREGISTRATION.md", "README.md", "approve_local_near80.py",
                                                         "run_near80.sh", "code/x.py")}
    rels = sorted(set(AC.PINS_REL.values()) | set(K.PIN_FILES) | {"experiments/shift/thing.py"})
    for rel in rels:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel == AC.V6_STAGE2_COMMON_REL:
            shutil.copyfile(K.ROOT / rel, p)
        else:
            p.write_text(json.dumps({"synthetic": rel}) + "\n", encoding="utf8")
    repo_files = {rel: AC.file_sha256(repo / rel) for rel in rels}
    msha = write_json_sidecar(n80 / K.MANIFEST_NAME, dict(kind=K.MANIFEST_KIND, label=K.LABEL, files=files,
                                                          repository_files=repo_files, history=[], reason="synthetic",
                                                          written="2026-10-15T20:00:00Z"))
    nt = dict(rules=[dict(rule="check 1: the band is really missing (leak check)", passed=frozen_passed)],
              not_run=[], passed=frozen_passed)
    fsha = write_json_sidecar(n80 / NP.FROZEN_REL, dict(seeds=list(K.EXPLORATORY), worlds=list(K.WORLDS),
                                                        no_training_checks=nt))
    draft = b'{"kind": "near80-preregistration-draft", "status": "draft"}\n'
    dsha = AC.sha256_bytes(draft)
    dname = K.DRAFT_HISTORY_NAME.format(prefix=dsha[:16])
    (n80 / K.HISTORY_DIR).mkdir(exist_ok=True)
    (n80 / K.HISTORY_DIR / dname).write_bytes(draft)
    (n80 / K.HISTORY_DIR / (dname + ".sha256")).write_text(f"{dsha}  {dname}\n", encoding="utf8")
    pins = AC.current_pins(repo)
    pins["repository_files_count"] = len(repo_files)
    prereg = dict(kind=K.PREREG_KIND, version=K.VERSION, label=K.LABEL, status=status,
                  status_line=K.STATUS_LINE_FINAL, approvable=True, missing=[], owner_answer=K.OWNER_ANSWER,
                  constants=AC.canonical_constants(), claims_word_for_word=None,
                  code=dict(code_manifest=K.MANIFEST_NAME, code_manifest_sha256=msha, files=files), pins=pins,
                  no_training_checks_frozen=dict(path=NP.FROZEN_REL, sha256=fsha, passed=frozen_passed, rules=1,
                                                 not_run=[], code_manifest_sha256=msha),
                  draft=dict(path=f"{K.HISTORY_DIR}/{dname}", sha256=dsha), approval=AC.approval_section(),
                  choices=[dict(number=1, choice="which thrusters", default="any thruster", taken="any thruster",
                                accepted=K.OWNER_ANSWER["utc"])],
                  seeds=dict(exploratory=list(K.EXPLORATORY), confirmatory=list(K.CONFIRMATORY),
                             refused_always=list(K.RESERVED)),
                  settings={"near80": {"role": "primary"}}, compute=dict(total="about 28-40", budget_cap=50),
                  summary=dict(claims=["claim one", "claim two", "claim three", "claim four"],
                               stop_rules=["harm", "rival"]))
    write_json_sidecar(n80 / K.PREREG_NAME, prereg)
    return AC.load_prereg(AC.ApprovalConfig(n80_root=n80, repo=repo))


class Isolated(unittest.TestCase):
    """A temporary folder, and git's global and system configuration replaced by an empty one (no identity, no
    signing): nothing of this machine's git setup leaks into the tests."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80_test_approval_"))
        gitconfig = self.tmp / "gitconfig"
        gitconfig.write_text("[commit]\n\tgpgsign = false\n[tag]\n\tgpgsign = false\n[init]\n\tdefaultBranch = main\n",
                             encoding="utf8")
        self._env = mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(gitconfig), "GIT_CONFIG_NOSYSTEM": "1",
                                                 "GIT_CONFIG_COUNT": "0", "GIT_TERMINAL_PROMPT": "0"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


# ============================================================================================ approve_local_near80.py

class TestApproveLocal(Isolated):
    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "repo"
        self.prereg = build_n80(self.repo)
        self.n80 = self.repo / K.N80_REL

    def run_al(self, args=(), stdin=None, typed=None, script=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = AL.main(list(args), folder=str(self.n80),
                           script_path=str(script or self.n80 / "approve_local_near80.py"),
                           stdin=Tty() if stdin is None else stdin,
                           input_fn=lambda _: self.prereg.prefix if typed is None else typed, repo=str(self.repo))
        return code, out.getvalue()

    def approvals(self):
        return sorted(p.name for p in self.n80.glob("approval_near80_*"))

    def refused(self, why=None, **kw):
        before = tree(self.n80)
        code, text = self.run_al(**kw)
        self.assertEqual(code, 2, text)
        self.assertIn("Refused.", text)
        self.assertIn("Nothing was written.", text)
        if why:
            self.assertIn(why, text)
        self.assertEqual(tree(self.n80), before)
        return text

    def test_writes_exactly_the_two_files_once(self):
        before = {p for p, _ in tree(self.n80)}
        code, text = self.run_al()
        self.assertEqual(code, 0, text)
        name = AC.approval_name(self.prereg.prefix)
        self.assertEqual({p for p, _ in tree(self.n80)} - before, {name, name + ".sha256"})
        raw = (self.n80 / name).read_bytes()
        self.assertEqual(AC.sidecar_digest(self.n80 / name), AC.sha256_bytes(raw))
        body = AC.check_local_approval(raw, self.prereg)
        self.assertEqual(body["statement"], AC.STATEMENT.format(version=K.VERSION, sha=self.prereg.sha256))
        self.assertIn(self.prereg.sha256, text)
        self.assertIn(f"git -c user.name=\"{K.APPROVER_NAME}\" -c user.email={K.APPROVER_EMAIL} commit -m "
                      f"\"Near-80: approve version 8 {self.prereg.prefix}\"", text)
        self.assertIn(f"git push origin {K.BRANCH}", text)
        self.assertIn(K.IMPORT_NAME.format(prefix=self.prereg.prefix), text)
        self.assertIn("accept the defaults", text)
        self.assertIn("all passed", text)
        self.refused("already exists")                                   # never overwrites

    def test_check_writes_nothing(self):
        before = tree(self.n80)
        code, text = self.run_al(args=["--check"], stdin=NotTty())
        self.assertEqual(code, 0, text)
        self.assertIn("Nothing was asked and nothing was written", text)
        self.assertEqual(tree(self.n80), before)

    def test_refuses_a_draft_status(self):
        p = json.loads((self.n80 / K.PREREG_NAME).read_text())
        p["status"] = "draft"
        write_json_sidecar(self.n80 / K.PREREG_NAME, p)
        self.refused("not final")

    def test_refuses_a_missing_entry(self):
        p = json.loads((self.n80 / K.PREREG_NAME).read_text())
        p["missing"] = ["something"]
        write_json_sidecar(self.n80 / K.PREREG_NAME, p)
        self.refused("not approvable")

    def test_refuses_a_changed_sidecar(self):
        (self.n80 / (K.PREREG_NAME + ".sha256")).write_text("0" * 64 + "  prereg.json\n")
        self.refused("fingerprint file")

    def test_refuses_a_changed_prereg(self):
        p = self.n80 / K.PREREG_NAME
        p.write_bytes(p.read_bytes().replace(b'"near80"', b'"near-80"'))
        self.refused("fingerprint file")

    def test_refuses_a_changed_manifest_file(self):
        (self.n80 / "code" / "x.py").write_text("x = 2\n")
        self.refused("code/x.py differs")

    def test_refuses_a_changed_script(self):
        other = self.tmp / "approve_local_near80.py"
        other.write_bytes((self.n80 / "approve_local_near80.py").read_bytes() + b"\n# changed\n")
        self.refused("is not the pinned approve_local_near80.py", script=other)

    def test_refuses_a_changed_repository_file(self):
        (self.repo / "experiments" / "shift" / "thing.py").write_text("y = 3\n")
        self.refused("experiments/shift/thing.py differs")

    def test_refuses_a_missing_frozen_no_training_file(self):
        (self.n80 / NP.FROZEN_REL).unlink()
        self.refused(f"{NP.FROZEN_REL} is missing")

    def test_refuses_failed_frozen_checks(self):
        shutil.rmtree(self.repo)
        self.prereg = build_n80(self.repo, frozen_passed=False)
        self.refused("did not all pass")

    def test_refuses_a_changed_history_draft(self):
        d = next(p for p in (self.n80 / K.HISTORY_DIR).glob("prereg_draft_*.json") if p.suffix == ".json")
        d.write_bytes(d.read_bytes() + b" ")
        self.refused("missing or differs")

    def test_refuses_an_existing_approval(self):
        (self.n80 / AC.approval_name(self.prereg.prefix)).write_text("{}")
        self.refused("already exists")

    def test_refuses_standard_input_that_is_not_a_terminal(self):
        self.refused("not a terminal", stdin=NotTty())

    def test_refuses_files_the_manifest_does_not_list(self):
        """A stray module next to the script or in code/, a package folder, a __pycache__, or another fingerprint's
        approval file is refused, even by --check (the documented command also runs Python with -I). This
        fingerprint's own approval and import records are allowed."""
        cases = (("json.py", "json.py is not a file of this pre-registration"),
                 ("code/evil.py", "code/evil.py is not in the code manifest"),
                 ("approval_near80_0000000000000000.json",
                  "approval_near80_0000000000000000.json is not a file of this pre-registration"),
                 ("hashlib/__init__.py", "hashlib/ holds Python files"),
                 ("__pycache__/x.pyc", "__pycache__ exists"),
                 ("code/__pycache__/x.pyc", "code/__pycache__ exists"))
        for rel, why in cases:
            p = self.n80 / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x = 1\n")
            self.refused(why, args=["--check"], stdin=NotTty())
            top = self.n80 / rel.split("/")[0] if "/" in rel and not rel.startswith("code/") else p
            if rel.startswith("code/__pycache__"):
                top = self.n80 / "code" / "__pycache__"
            shutil.rmtree(top) if top.is_dir() else top.unlink()
        for name in (AC.import_name(self.prereg.prefix), AC.import_name(self.prereg.prefix) + ".sha256"):
            (self.n80 / name).write_text("{}")
        code, text = self.run_al(args=["--check"], stdin=NotTty())
        self.assertEqual(code, 0, text)

    def test_refuses_wrong_or_empty_input(self):
        self.refused("do not match", typed="0" * 16)
        self.refused("do not match", typed=self.prereg.prefix[:15])
        self.refused("nothing was typed", typed="")
        self.refused("nothing was typed", typed="   ")

    def test_constants_equal_the_codes(self):
        self.assertEqual(AL.FOLDER_POSIX, K.N80_REL)
        self.assertEqual(AL.REPO, str(K.ROOT))
        self.assertEqual((AL.PREREG_NAME, AL.MANIFEST_NAME), (K.PREREG_NAME, K.MANIFEST_NAME))
        self.assertEqual(AL.APPROVAL_NAME.format("p"), K.APPROVAL_NAME.format(prefix="p"))
        self.assertEqual(AL.IMPORT_NAME.format("p"), K.IMPORT_NAME.format(prefix="p"))
        self.assertEqual((AL.KIND, AL.MODE, AL.PREREG_KIND, AL.STATUS_FINAL),
                         (K.LOCAL_KIND, K.LOCAL_MODE, K.PREREG_KIND, K.STATUS_FINAL))
        self.assertEqual((AL.APPROVER_EMAIL, AL.APPROVER_NAME, AL.BRANCH, AL.PREFIX_LEN),
                         (K.APPROVER_EMAIL, K.APPROVER_NAME, K.BRANCH, K.PREFIX_LEN))
        self.assertEqual(AL.FROZEN_NAME, NP.FROZEN_REL)
        self.assertEqual(AL.STATEMENT, AC.STATEMENT)

    def test_standard_library_only(self):
        """It imports nothing from code/ and only standard-library modules, and the system Python can parse it."""
        src = (K.N80 / "approve_local_near80.py").read_text()
        names = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertTrue(names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names))
        system = shutil.which("python3", path="/usr/bin:/bin")
        if system:
            done = subprocess.run([system, "-I", "-c", "import ast, sys; ast.parse(open(sys.argv[1]).read())",
                                   str(K.N80 / "approve_local_near80.py")], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            self.assertEqual(done.returncode, 0, done.stderr)

    def test_real_folder_check_writes_nothing(self):
        """Run with the system Python on the real folder: --check either refuses (before the freeze) or passes;
        it never writes."""
        system = shutil.which("python3", path="/usr/bin:/bin") or sys.executable
        before = sorted(p.name for p in K.N80.iterdir())
        done = subprocess.run([system, "-I", str(K.N80 / "approve_local_near80.py"), "--check"], cwd="/tmp",
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        text = (done.stdout + done.stderr).decode()
        self.assertIn(done.returncode, (0, 2), text)
        self.assertTrue("Refused." in text or "--check: every check passed" in text, text)
        self.assertEqual(sorted(p.name for p in K.N80.iterdir()), before)


# ============================================================================================ the import, with git

class ChainBase(Isolated):
    """A throwaway bare origin and a clone on the near-80 branch, holding a synthetic final near-80 folder committed
    by the session and pushed."""

    def setUp(self):
        super().setUp()
        self.origin = self.tmp / "origin.git"
        git(self.tmp, "init", "-q", "--bare", str(self.origin))
        self.repo = self.tmp / "repo"
        git(self.tmp, "clone", "-q", str(self.origin), str(self.repo))
        git(self.repo, "checkout", "-q", "-b", K.BRANCH)
        self.prereg = build_n80(self.repo)
        self.n80 = self.repo / K.N80_REL
        self.commit(AGENT, "freeze (synthetic)", ".")
        self.push()
        self.cfg = AC.ApprovalConfig(n80_root=self.n80, repo=self.repo, branch=K.BRANCH)
        self.aname = AC.approval_name(self.prereg.prefix)
        self.iname = AC.import_name(self.prereg.prefix)

    def commit(self, committer, msg, *paths, author=None):
        git(self.repo, "add", "-A", *paths)
        git(self.repo, "commit", "-q", "-m", msg, committer=committer, author=author)
        return git(self.repo, "rev-parse", "HEAD")

    def push(self):
        git(self.repo, "push", "-q", "origin", K.BRANCH)

    def owner_approves(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = AL.main([], folder=str(self.n80), script_path=str(self.n80 / "approve_local_near80.py"),
                           stdin=Tty(), input_fn=lambda _: self.prereg.prefix, repo=str(self.repo))
        self.assertEqual(code, 0, out.getvalue())

    def owner_commits(self, author=APPROVER, committer=APPROVER, push=True, approve=True):
        if approve:
            self.owner_approves()
        rels = [f"{K.N80_REL}/{self.aname}", f"{K.N80_REL}/{self.aname}.sha256"]
        c = self.commit(committer, f"Near-80: approve version 8 {self.prereg.prefix}", *rels, author=author)
        if push:
            self.push()
        return c

    def do_import(self):
        return IA.import_record(self.cfg, log=quiet)

    def push_import(self):
        rels = [f"{K.N80_REL}/{self.iname}", f"{K.N80_REL}/{self.iname}.sha256"]
        self.commit(AGENT, "Near-80: import record", *rels)
        self.push()

    def import_refused(self, why=None):
        before = tree(self.n80)
        with self.assertRaises(AC.ApprovalRefused) as cm:
            self.do_import()
        msg = str(cm.exception)
        self.assertTrue(msg.startswith("refused: "), msg)
        if why:
            self.assertIn(why, msg)
        self.assertEqual(tree(self.n80), before)
        self.assertFalse((self.n80 / self.iname).exists())
        return msg

    def set_session_email(self, email):
        git(self.repo, "config", "user.email", email)


class TestImport(ChainBase):
    def test_good_chain_imports_once(self):
        commit = self.owner_commits()
        body = self.do_import()
        self.assertEqual((body["commit"], body["author"], body["committer"]), (commit, APPROVER, APPROVER))
        self.assertEqual(body["kind"], K.IMPORT_KIND)
        self.assertEqual(body["approval_file"], AC.approval_rel(self.prereg.prefix))
        self.assertEqual(body["approval_file_sha256"], AC.file_sha256(self.n80 / self.aname))
        self.assertEqual(body["branch"], K.BRANCH)
        self.assertEqual(body["approver_email"], K.APPROVER_EMAIL)
        self.assertIn("Check on GitHub that this commit is yours", body["note"])
        self.assertEqual(AC.sidecar_digest(self.n80 / self.iname), AC.file_sha256(self.n80 / self.iname))
        before = tree(self.n80)
        with self.assertRaises(AC.ApprovalRefused) as cm:                 # written once
            self.do_import()
        self.assertIn("exists already", str(cm.exception))
        self.assertEqual(tree(self.n80), before)

    def test_refuses_the_agent_as_author_or_committer(self):
        self.owner_commits(author=AGENT)
        self.import_refused("own git identity")

    def test_refuses_the_agent_as_committer(self):
        self.owner_commits(committer=AGENT)
        self.import_refused("own git identity")

    def test_refuses_the_agent_in_another_case(self):
        self.owner_commits(author="Someone <NoReply@Anthropic.com>")
        self.import_refused("own git identity")

    def test_refuses_the_sessions_configured_email_as_author(self):
        self.set_session_email("agent@session.example")
        self.owner_commits(author="Agent <agent@session.example>")
        self.import_refused("own git identity")

    def test_refuses_the_sessions_configured_email_as_committer(self):
        self.set_session_email("agent@session.example")
        self.owner_commits(committer="Agent <agent@session.example>")
        self.import_refused("own git identity")

    def test_refuses_when_the_session_is_configured_as_the_owner(self):
        """A session whose git identity is the owner's address cannot import a commit made under that address."""
        self.set_session_email(K.APPROVER_EMAIL)
        self.owner_commits()
        self.import_refused("own git identity")

    def test_refuses_an_author_or_committer_who_is_not_the_owner(self):
        self.owner_commits(author="Synthetic User <user@example.org>")
        self.import_refused("not the owner")

    def test_refuses_a_committer_who_is_not_the_owner(self):
        self.owner_commits(committer="Synthetic User <user@example.org>")
        self.import_refused("not the owner")

    def test_refuses_zero_commits(self):
        self.owner_approves()                                              # written, never committed
        self.import_refused("exactly one commit")

    def test_refuses_two_adding_commits(self):
        self.owner_commits()
        rels = [f"{K.N80_REL}/{self.aname}", f"{K.N80_REL}/{self.aname}.sha256"]
        git(self.repo, "rm", "-q", *rels)
        git(self.repo, "commit", "-q", "-m", "remove", committer=APPROVER)
        git(self.repo, "checkout", "HEAD~1", "--", *rels)
        git(self.repo, "commit", "-q", "-m", "add again", committer=APPROVER)
        self.push()
        self.import_refused("exactly one commit")

    def test_refuses_a_commit_not_on_origins_branch(self):
        self.owner_commits(push=False)
        self.import_refused("exactly one commit")

    def test_refuses_an_approval_of_another_fingerprint(self):
        self.owner_approves()
        p = self.n80 / self.aname
        body = json.loads(p.read_text())
        body["prereg_sha256"] = "f" * 64
        write_json_sidecar(p, body)
        self.owner_commits(approve=False)
        self.import_refused("approves another version")

    def test_refuses_a_commit_whose_tree_holds_a_changed_pinned_file(self):
        self.owner_approves()
        x = self.n80 / "code" / "x.py"
        good = x.read_bytes()
        x.write_text("x = 99\n")
        rels = [f"{K.N80_REL}/{self.aname}", f"{K.N80_REL}/{self.aname}.sha256", f"{K.N80_REL}/code/x.py"]
        self.commit(APPROVER, "approve with a change", *rels)
        self.push()
        x.write_bytes(good)                                                # the working tree is back as pinned
        self.import_refused("does not hold the pinned near-80 files")

    def test_refuses_when_not_approvable_now(self):
        self.owner_commits()
        (self.repo / "experiments" / "shift" / "thing.py").write_text("y = 4\n")
        self.import_refused("cannot be approved now")

    def test_refuses_a_fixture(self):
        cfg = AC.write_test_fixture(self.tmp / "fx")
        with self.assertRaises(AC.ApprovalRefused):
            IA.import_record(cfg, log=quiet)

    def test_show(self):
        lines = []
        prereg, ok = IA.show(self.cfg, log=lines.append)
        self.assertTrue(ok, lines)
        self.assertIn(prereg.sha256, lines[0])
        self.assertIn("Approvable now: yes", lines[0])
        self.assertIn("Import record present: False", lines[0])


class TestRequireApproval(ChainBase):
    def setUp(self):
        super().setUp()
        self.commit_sha = self.owner_commits()
        self.do_import()

    def refuses(self, why=None, cfg=None):
        with self.assertRaises(AC.ApprovalRefused) as cm:
            AC.require_approval(cfg or self.cfg)
        msg = str(cm.exception)
        self.assertTrue(msg.startswith("refused: "), msg)
        if why:
            self.assertIn(why, msg)
        return msg

    def test_passes_on_a_good_chain(self):
        self.refuses("not on origin's branch")                             # the import record is not pushed yet
        self.push_import()
        got = AC.require_approval(self.cfg)
        self.assertEqual(set(got), {"prereg_sha256", "prefix", "code_manifest_sha256", "approval_file",
                                    "approval_sha256", "import_file", "import_sha256", "commit", "author", "fixture"})
        self.assertEqual(got["commit"], self.commit_sha)
        self.assertEqual(got["author"], APPROVER)
        self.assertIs(got["fixture"], False)
        self.assertEqual(got["prereg_sha256"], self.prereg.sha256)
        self.assertEqual(got["approval_file"], AC.approval_rel(self.prereg.prefix))

    def test_refuses_each_tamper(self):
        self.push_import()
        AC.require_approval(self.cfg)
        cases = [
            ("code/x.py", lambda: (self.n80 / "code" / "x.py").write_text("x = 5\n"), "code/x.py differs"),
            ("an unlisted code file", lambda: (self.n80 / "code" / "y.py").write_text("y = 1\n"),
             "code/y.py is not in the manifest"),
            ("a bytecode cache in code/", lambda: (self.n80 / "code" / "__pycache__").mkdir(),
             "code/__pycache__ exists; remove it"),
            ("a compiled file in code/", lambda: (self.n80 / "code" / "x.cpython-312.pyc").write_bytes(b"\0"),
             "code/x.cpython-312.pyc is compiled bytecode"),
            ("a repository file", lambda: (self.repo / "experiments/shift/thing.py").write_text("y = 6\n"),
             "experiments/shift/thing.py differs"),
            ("a pinned result", lambda: (self.repo / AC.PINS_REL["s1_results_sha256"]).write_text("{}\n"),
             "differs"),
            ("the prereg sidecar", lambda: (self.n80 / "prereg.json.sha256").write_text("0" * 64 + "  prereg.json\n"),
             "sidecar"),
            ("the manifest", lambda: (self.n80 / K.MANIFEST_NAME).write_text("{}"), "sidecar"),
            ("the approval file", lambda: (self.n80 / self.aname).write_bytes((self.n80 / self.aname).read_bytes()
                                                                              + b" "), "sidecar"),
            ("the import record's sidecar", lambda: (self.n80 / (self.iname + ".sha256")).write_text("0" * 64),
             "sidecar"),
            ("a missing import record", lambda: (self.n80 / self.iname).unlink(), "no import record"),
            ("a missing approval", lambda: (self.n80 / self.aname).unlink(), "no approval of this fingerprint"),
        ]
        snapshot = self.tmp / "snapshot"
        shutil.copytree(self.repo, snapshot, symlinks=True)
        for name, tamper, why in cases:
            with self.subTest(tamper=name):
                tamper()
                self.refuses(why)
                shutil.rmtree(self.repo)
                shutil.copytree(snapshot, self.repo, symlinks=True)
                AC.require_approval(self.cfg)

    def test_refuses_changed_constants(self):
        self.push_import()
        with mock.patch.object(K, "ALPHA", 4.0):
            self.refuses("near80_constants.py differs")

    def test_refuses_a_relaxed_configuration_without_a_fixture(self):
        self.push_import()
        for kw in (dict(check_git=False), dict(check_pins=False)):
            with self.subTest(**kw):
                self.refuses("only for a test fixture", AC.ApprovalConfig(n80_root=self.n80, repo=self.repo, **kw))

    def test_refuses_another_branch(self):
        self.push_import()
        self.refuses("names the branch", AC.ApprovalConfig(n80_root=self.n80, repo=self.repo, branch="other"))

    def test_refuses_the_agent_identity_at_check_time(self):
        """verify_chain asks git for the configured user.email at every check: a commit by it is refused."""
        self.push_import()
        self.set_session_email(K.APPROVER_EMAIL)
        self.refuses("own git identity")


class TestRealFolder(unittest.TestCase):
    def test_the_real_folder_refuses_before_the_freeze(self):
        """Before the freeze there is no prereg.json; after it, nothing is approved and imported until the owner and
        the session have done so. Either way no phase may start now."""
        if (K.N80 / AC.import_name("x")).exists():
            self.skipTest("unexpected file")
        imported = list(K.N80.glob("import_near80_*.json"))
        if imported:
            self.skipTest("an import record exists (after approval this test has nothing to show)")
        with self.assertRaises(AC.ApprovalRefused):
            AC.require_approval()


# ============================================================================================ the git identity

class TestGitIdentity(Isolated):
    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "r"
        git(self.tmp, "init", "-q", str(self.repo))

    def test_refuses_the_owner_and_the_agent_and_passes_unset(self):
        AC.refuse_git_identity(self.repo)                                   # unset: passes
        for email, refused in ((K.APPROVER_EMAIL, True), (K.APPROVER_EMAIL.upper(), True),
                               ("noreply@anthropic.com", True), ("NoReply@Anthropic.com", True),
                               ("someone@example.org", False)):
            with self.subTest(email=email):
                git(self.repo, "config", "user.email", email)
                if refused:
                    with self.assertRaises(AC.ApprovalRefused) as cm:
                        AC.refuse_git_identity(self.repo)
                    self.assertIn("git config --global --unset user.email", str(cm.exception))
                else:
                    AC.refuse_git_identity(self.repo)
        git(self.repo, "config", "--unset", "user.email")
        AC.refuse_git_identity(self.repo)

    def test_reads_the_global_configuration_too(self):
        with open(os.environ["GIT_CONFIG_GLOBAL"], "a", encoding="utf8") as handle:
            handle.write(f"[user]\n\temail = {K.APPROVER_EMAIL}\n")
        with self.assertRaises(AC.ApprovalRefused):
            AC.refuse_git_identity(self.repo)
        git(self.repo, "config", "user.email", "someone@example.org")      # a local value does not hide it
        with self.assertRaises(AC.ApprovalRefused):
            AC.refuse_git_identity(self.repo)

    def test_refuses_when_git_cannot_answer(self):
        with self.assertRaises(AC.ApprovalRefused):
            AC.refuse_git_identity(self.tmp / "no such folder")


# ============================================================================================ the test fixture

class TestFixture(Isolated):
    def a_code_root(self):
        """A copy of A's files only (other builders' files may be half-written while this runs in parallel)."""
        root = self.tmp / "src"
        if (root / "code").is_dir():
            return root / "code"
        (root / "code").mkdir(parents=True)
        for name in ("PREREGISTRATION.md", "README.md", "approve_local_near80.py"):
            shutil.copyfile(K.N80 / name, root / name)
        for name in A_FILES:
            shutil.copyfile(K.CODE / name, root / "code" / name)
        return root / "code"

    def test_good_fixture_passes(self):
        cfg = AC.write_test_fixture(self.tmp / "fx", code_root=self.a_code_root())
        self.assertEqual((cfg.check_git, cfg.check_pins), (False, False))
        got = AC.require_approval(cfg)
        self.assertIs(got["fixture"], True)
        self.assertEqual(got["commit"], "0" * 40)
        prereg = AC.load_prereg(cfg)
        for key in ("kind", "status", "status_line", "approvable", "missing", "constants", "code", "approval",
                    "claims_word_for_word"):
            self.assertIn(key, prereg.payload)
        self.assertEqual(prereg.payload["constants"], AC.canonical_constants())
        self.assertEqual(set(prereg.payload["code"]["files"]),
                         {"PREREGISTRATION.md", "README.md", "approve_local_near80.py"}
                         | {"code/" + n for n in A_FILES})
        self.assertEqual((self.tmp / "fx" / K.TEST_MARKER).read_text(), K.TEST_MARKER_TEXT)
        AC.check_local_approval((self.tmp / "fx" / AC.approval_name(prereg.prefix)).read_bytes(), prereg)

    def test_fixture_from_the_real_code_folder(self):
        cfg = AC.write_test_fixture(self.tmp / "fx")
        AC.require_approval(cfg)
        claims = AC.load_prereg(cfg).payload["claims_word_for_word"]
        if CLAIMS_REAL:
            self.assertIsInstance(claims, dict)
        else:
            self.assertIsNone(claims)

    def test_fixture_refusals(self):
        with self.assertRaises(AC.ApprovalRefused):
            AC.write_test_fixture(K.ROOT / "runs" / "near80_fixture_never_written")
        self.assertFalse((K.ROOT / "runs" / "near80_fixture_never_written").exists())
        with self.assertRaises(AC.ApprovalRefused):
            AC.fixture_config(self.tmp)                                     # no marker
        (self.tmp / K.TEST_MARKER).write_text("something else\n")
        with self.assertRaises(AC.ApprovalRefused):
            AC.fixture_config(self.tmp)                                     # a wrong marker
        cfg = AC.write_test_fixture(self.tmp / "fx", code_root=self.a_code_root())
        with self.assertRaises(AC.ApprovalRefused):
            AC.write_test_fixture(self.tmp / "fx", code_root=self.a_code_root())   # written once
        (self.tmp / "fx" / "code" / "near80lib.py").write_text("changed = True\n")
        with self.assertRaises(AC.ApprovalRefused) as cm:
            AC.require_approval(cfg)
        self.assertIn("code/near80lib.py differs", str(cm.exception))
        with self.assertRaises(AC.ApprovalRefused) as cm:
            AC.require_approval(AC.ApprovalConfig(n80_root=self.tmp / "fx", check_git=False, check_pins=False))
        self.assertIn("only for a test fixture", str(cm.exception))

    def test_fixture_import_record_is_checked(self):
        cfg = AC.write_test_fixture(self.tmp / "fx", code_root=self.a_code_root())
        prereg = AC.load_prereg(cfg)
        imp = self.tmp / "fx" / AC.import_name(prereg.prefix)
        body = json.loads(imp.read_text())
        body["author"] = AGENT
        write_json_sidecar(imp, body)
        with self.assertRaises(AC.ApprovalRefused) as cm:
            AC.require_approval(cfg)
        self.assertIn("own git identity", str(cm.exception))
        body["author"] = "Synthetic User <user@example.org>"
        write_json_sidecar(imp, body)
        with self.assertRaises(AC.ApprovalRefused) as cm:
            AC.require_approval(cfg)
        self.assertIn("not the owner", str(cm.exception))


# ============================================================================================ the JSON helpers

class TestJson(Isolated):
    def test_write_once_and_read_verified(self):
        p = self.tmp / "a" / "x.json"
        digest = AC.write_json_once(p, {"b": 1, "a": float("nan"), "c": (1, 2)})
        self.assertEqual(p.read_text(), '{\n "a": null,\n "b": 1,\n "c": [\n  1,\n  2\n ]\n}')
        self.assertEqual(Path(str(p) + ".sha256").read_text(), f"{digest}  x.json\n")
        self.assertEqual(AC.read_json_verified(p), {"a": None, "b": 1, "c": [1, 2]})
        with self.assertRaises(AC.ApprovalRefused):
            AC.write_json_once(p, {})
        p.write_text("{}")
        with self.assertRaises(AC.ApprovalRefused):
            AC.read_json_verified(p)
        Path(str(p) + ".sha256").unlink()
        with self.assertRaises(AC.ApprovalRefused):
            AC.read_json_verified(p)
        with self.assertRaises(AC.ApprovalRefused):
            AC.read_json_verified(self.tmp / "missing.json")
        self.assertIsNone(AC.sidecar_digest(p))

    def test_refusal_messages_start_with_refused(self):
        self.assertEqual(str(AC.ApprovalRefused("x")), "refused: x")
        self.assertEqual(str(AC.ApprovalRefused("refused: y")), "refused: y")


# ============================================================================================ amendment 1

TEST_CHECK_SHA = dict(py="1" * 64, sh="2" * 64)    # a test amendment's 'desktop_check' (the check code it names)


def write_test_amendment(fx, edits, typed=None, approve=True, imported=True, delta=None, approval=None):
    """A test fixture's amendment 1 (test helper; nothing here is a real approval). edits {path in the fixture: bytes}:
    a .md path gets the bytes appended, any other path is rewritten with them. Writes amendment_1.json (+ .sha256)
    with the full file list and the delta (keys of `delta` replace the delta's), then (approve) the owner's approval of
    the amendment (keys of `approval` replace its fields) and (imported) its import record, commit '0' * 40. Returns
    the amendment's SHA-256."""
    fx = Path(fx)
    pre = AC.read_json_verified(fx / K.PREREG_NAME)
    psha = AC.file_sha256(fx / K.PREREG_NAME)
    parent_files = dict(AC.read_json_verified(fx / K.MANIFEST_NAME)["files"])
    appended = {}
    for rel, data in edits.items():
        p = fx / rel
        if rel.endswith(".md"):
            appended[rel] = dict(parent_bytes=p.stat().st_size, parent_sha256=parent_files[rel])
            p.write_bytes(p.read_bytes() + data)
        else:
            p.write_bytes(data)
    files = {rel: AC.file_sha256(fx / rel) for rel in parent_files}
    changed = {rel: [parent_files[rel], files[rel]] for rel in parent_files if files[rel] != parent_files[rel]}
    prefix = psha[:K.PREFIX_LEN]
    body = dict(kind=AC.AMENDMENT_KIND, number=AC.AMENDMENT_NUMBER, label=K.LABEL,
                parent=dict(prereg_sha256=psha, code_manifest_sha256=pre["code"]["code_manifest_sha256"],
                            approval_sha256=AC.file_sha256(fx / AC.approval_name(prefix)),
                            import_sha256=AC.file_sha256(fx / AC.import_name(prefix))),
                files=files,
                delta=dict(dict(changed=changed, added={}, removed=[], appended=appended), **(delta or {})),
                desktop_check=TEST_CHECK_SHA, fixture=True)
    asha = AC.write_json_once(fx / AC.AMENDMENT_NAME, body)
    a16 = asha[:K.PREFIX_LEN]
    if approve:
        appr = dict(kind=AC.AMEND_LOCAL_KIND, label=K.LABEL, mode=K.LOCAL_MODE,
                    amendment_path=K.N80_REL + "/" + AC.AMENDMENT_NAME, amendment_number=AC.AMENDMENT_NUMBER,
                    amendment_sha256=asha, parent_prereg_sha256=psha, typed=typed or a16, stdin_isatty=True,
                    approved_at_utc=AC.utc_now(), approve_local_sha256=files.get("approve_local_near80.py"),
                    statement=AC.AMEND_STATEMENT.format(sha=asha, parent=psha),
                    desktop_checks={AC.AMEND_CHECK_PREFIX + "t.json": "0" * 64}, fixture=True)
        appr.update(approval or {})
        apsha = AC.write_json_once(fx / AC.AMEND_APPROVAL_NAME.format(prefix=a16), appr)
        if imported:
            AC.write_json_once(fx / AC.AMEND_IMPORT_NAME.format(prefix=a16), dict(
                kind=AC.AMEND_IMPORT_KIND, label=K.LABEL, prereg_sha256=psha, amendment_sha256=asha,
                amendment_number=AC.AMENDMENT_NUMBER, approval_file=AC.amend_approval_rel(a16),
                approval_file_sha256=apsha, commit="0" * 40, author=APPROVER, committer=APPROVER, branch=K.BRANCH,
                fixture=True))
    return asha


def write_desktop_check(n80, asha, name="desktop_check_20261019T000000Z.json", **fields):
    """A test helper: the desktop check's result file (near80_amendment1/, next to the near-80 folder), as
    check_amendment1.py writes it when it passed in official mode, made by the check code TEST_CHECK_SHA, with the four
    unit test files run and passed; keys of `fields` replace its fields (None removes one). Returns its path relative
    to the folder's parent."""
    body = dict(kind=AL.CHECK_KIND, mode="official", passed=True, amendment_sha256=asha, check_sha256=TEST_CHECK_SHA,
                unit_tests={n: 0 for n in K.TEST_FILES}, tests_run=True, fixture=True)
    body.update(fields)
    body = {k: v for k, v in body.items() if v is not None}
    write_json_sidecar(Path(n80).parent / "near80_amendment1" / name, body)
    return "near80_amendment1/" + name


def rewrite_amendment(fx, change):
    """A test helper: amendment_1.json rewritten in place (+ .sha256) after change(body)."""
    p = Path(fx) / AC.AMENDMENT_NAME
    body = json.loads(p.read_text(encoding="utf8"))
    change(body)
    return write_json_sidecar(p, body)


class TestAmendmentFixture(Isolated):
    """Amendment 1's gate on test fixtures (no git): the amended files pass only with an approved, imported
    amendment_1.json that names every change, and the facts keep the approved prereg.json's fingerprint."""

    def fixture(self, name="fx"):
        cfg = AC.write_test_fixture(self.tmp / name)
        return Path(cfg.n80_root), cfg

    @staticmethod
    def good_edits(fx):
        return {"code/stubs.py": (fx / "code" / "stubs.py").read_bytes() + b"\n# amendment test\n",
                "PREREGISTRATION.md": b"\n---\n\n## 19. Amendment 1 (test)\n\nAppended.\n"}

    def test_no_amendment_passes_as_today(self):
        _, cfg = self.fixture()
        got = AC.require_approval(cfg)
        self.assertNotIn("amendment", got)
        self.assertIsNone(AC.load_amendment(AC.load_prereg(cfg), cfg))

    def test_a_good_amendment_passes_and_keeps_the_fingerprint(self):
        fx, cfg = self.fixture()
        parent = AC.require_approval(cfg)
        asha = write_test_amendment(fx, self.good_edits(fx))
        got = AC.require_approval(cfg)
        self.assertEqual(got["prereg_sha256"], parent["prereg_sha256"])
        self.assertEqual({k: v for k, v in got.items() if k != "amendment"}, parent)
        a = got["amendment"]
        self.assertEqual((a["number"], a["sha256"], a["prefix"]), (1, asha, asha[:K.PREFIX_LEN]))
        self.assertEqual((a["approval_file"], a["import_file"]),
                         (AC.amend_approval_rel(asha[:K.PREFIX_LEN]), AC.amend_import_rel(asha[:K.PREFIX_LEN])))
        manifest = AC.check_code(AC.load_prereg(cfg), cfg)
        self.assertEqual(manifest["amendment"], dict(file=AC.AMENDMENT_NAME, sha256=asha))
        self.assertEqual(manifest["files"]["code/stubs.py"], AC.file_sha256(fx / "code" / "stubs.py"))

    def test_each_refusal(self):
        good = self.good_edits

        def change_without_amendment(fx):
            p = fx / "code" / "stubs.py"
            p.write_bytes(p.read_bytes() + b"# x\n")

        def change_after_the_amendment(fx):
            write_test_amendment(fx, good(fx))
            p = fx / "code" / "phases.py"
            p.write_bytes(p.read_bytes() + b"# y\n")

        def document_edited_in_the_middle(fx):
            p = fx / "PREREGISTRATION.md"
            data = p.read_bytes()
            p.write_bytes(data[:100] + b"X" + data[100:] + b"\n## 19. Amendment 1 (test)\n")
            parent = AC.read_json_verified(fx / K.MANIFEST_NAME)["files"]["PREREGISTRATION.md"]
            write_test_amendment(fx, {"code/stubs.py": good(fx)["code/stubs.py"]},
                                 delta=dict(appended={"PREREGISTRATION.md": dict(parent_bytes=len(data),
                                                                                 parent_sha256=parent)}))

        def constants_changed(fx):
            e = good(fx)
            e["code/near80_constants.py"] = (fx / "code" / "near80_constants.py").read_bytes() + b"\n# a comment\n"
            write_test_amendment(fx, e)

        def another_amendment_file(fx):
            write_test_amendment(fx, good(fx))
            (fx / "amendment_2.json").write_text("{}", encoding="utf8")

        def malformed(key, value, in_delta=False):
            def tamper(fx):
                write_test_amendment(fx, good(fx))
                rewrite_amendment(fx, lambda b: (b["delta"] if in_delta else b).__setitem__(key, value))
            return tamper

        def a_removed_file(fx):
            rel = sorted(AC.read_json_verified(fx / K.MANIFEST_NAME)["files"])[0]
            write_test_amendment(fx, good(fx), delta=dict(removed=[rel]))

        cases = [
            ("a changed file, no amendment", change_without_amendment, "code/stubs.py differs"),
            ("not approved", lambda fx: write_test_amendment(fx, good(fx), approve=False),
             "no approval of amendment 1"),
            ("not imported", lambda fx: write_test_amendment(fx, good(fx), imported=False), "no import record"),
            ("a change after the amendment", change_after_the_amendment, "code/phases.py differs"),
            ("a delta that hides a change", lambda fx: write_test_amendment(fx, good(fx), delta=dict(changed={})),
             "are not the ones its delta names"),
            ("a document edited in the middle", document_edited_in_the_middle, "may only be appended to"),
            ("near80_constants.py changed", constants_changed,
             "code/near80_constants.py may not change in an amendment"),
            ("a wrong typed prefix", lambda fx: write_test_amendment(fx, good(fx), typed="0" * 16),
             "the typed characters are not exactly the first 16"),
            ("another approve_local_near80.py",
             lambda fx: write_test_amendment(fx, good(fx), approval=dict(approve_local_sha256="f" * 64)),
             "made with another approve_local_near80.py"),
            ("a statement without both fingerprints",
             lambda fx: write_test_amendment(fx, good(fx), approval=dict(statement="I approve amendment 1.")),
             "its statement does not name both fingerprints"),
            ("an amendment_2.json", another_amendment_file, "only amendment 1 is known to this code"),
            ("a delta that is a list", malformed("delta", [1]), "its parent, files or delta are not of the expected"),
            ("a parent that is a list", malformed("parent", [1]), "its parent, files or delta are not of the expected"),
            ("a changed map that is a list", malformed("changed", [1], in_delta=True),
             "its parent, files or delta are not of the expected"),
            ("removed paths that are not strings", malformed("removed", [[1]], in_delta=True),
             "its parent, files or delta are not of the expected"),
            ("an added file",
             lambda fx: write_test_amendment(fx, good(fx), delta=dict(added={"code/extra.py": "0" * 64})),
             "amendment 1 adds and removes no file"),
            ("a removed file", a_removed_file, "amendment 1 adds and removes no file"),
            ("an approval that names no desktop check",
             lambda fx: write_test_amendment(fx, good(fx), approval=dict(desktop_checks={})), "names no desktop check"),
            ("an approval whose desktop check lies elsewhere",
             lambda fx: write_test_amendment(fx, good(fx), approval=dict(
                 desktop_checks={K.N80_REL + "/desktop_check_t.json": "0" * 64})), "names no desktop check"),
        ]
        for n, (name, tamper, why) in enumerate(cases):
            with self.subTest(case=name):
                fx, cfg = self.fixture(f"fx{n}")
                tamper(fx)
                with self.assertRaises(AC.ApprovalRefused) as cm:
                    AC.require_approval(cfg)
                self.assertTrue(str(cm.exception).startswith("refused: "), str(cm.exception))
                self.assertIn(why, str(cm.exception))

    def test_show_names_the_amendment(self):
        fx, cfg = self.fixture()
        lines = []
        IA.show(cfg, log=lines.append)
        self.assertNotIn("Amendment 1", "\n".join(lines))
        asha = write_test_amendment(fx, self.good_edits(fx), imported=False)
        lines = []
        IA.show(cfg, log=lines.append)
        text = "\n".join(lines)
        self.assertIn(f"SHA-256: {asha}\nFirst 16 characters: {asha[:K.PREFIX_LEN]}", text)
        self.assertIn("import_approval.py import-amendment", text)
        self.assertIn(AC.amend_import_rel(asha[:K.PREFIX_LEN]), text)
        self.assertIn("Owner's approval of amendment 1 present: True", text)
        self.assertIn("Amendment import record present: False", text)
        rewrite_amendment(fx, lambda b: b.__setitem__("delta", [1]))
        lines = []
        IA.show(cfg, log=lines.append)                                       # a malformed one is named, not raised
        self.assertIn("Amendment 1: refused: amendment_1.json does not amend", "\n".join(lines))

    def test_desktop_check_problems(self):
        """The import's rules for a result file of the desktop check (the same as approve_local_near80's)."""
        fx, _ = self.fixture()
        asha = write_test_amendment(fx, self.good_edits(fx), approve=False)
        amendment = AC.Amendment(fx / AC.AMENDMENT_NAME, asha, AC.read_json_verified(fx / AC.AMENDMENT_NAME))

        def result(**fields):
            body = dict(kind=AC.AMEND_CHECK_KIND, mode="official", passed=True, amendment_sha256=asha,
                        check_sha256=TEST_CHECK_SHA, unit_tests={n: 0 for n in K.TEST_FILES})
            body.update(fields)
            return json.dumps({k: v for k, v in body.items() if v is not None}).encode("utf8")

        self.assertEqual(AC.desktop_check_problems(result(), amendment), [])
        cases = [(dict(kind="x"), "it is not a result of the desktop check"),
                 (dict(mode="stub"), "it did not run in official mode"),
                 (dict(passed=False), "it did not pass"),
                 (dict(amendment_sha256="f" * 64), "it checked another version of amendment_1.json"),
                 (dict(check_sha256=dict(TEST_CHECK_SHA, py="3" * 64)), "made by another check"),
                 (dict(check_sha256=None), "made by another check"),
                 (dict(unit_tests=None), "it did not run and pass the four unit test files"),
                 (dict(unit_tests={}), "it did not run and pass the four unit test files"),
                 (dict(unit_tests={n: 0 for n in K.TEST_FILES[:3]}), "the four unit test files"),
                 (dict(unit_tests=dict({n: 0 for n in K.TEST_FILES}, **{K.TEST_FILES[-1]: 1})),
                  "the four unit test files"),
                 (dict(unit_tests={n: False for n in K.TEST_FILES}), "the four unit test files")]
        for fields, why in cases:
            with self.subTest(fields=sorted(fields)):
                self.assertIn(why, "; ".join(AC.desktop_check_problems(result(**fields), amendment)))
        self.assertEqual(AC.desktop_check_problems(b"\xff", amendment), ["it is not UTF-8 JSON"])
        no_code = AC.Amendment(amendment.path, asha, {k: v for k, v in amendment.payload.items()
                                                      if k != "desktop_check"})
        self.assertIn("made by another check", "; ".join(AC.desktop_check_problems(result(), no_code)))


class TestAmendmentChain(ChainBase):
    """Amendment 1 with git: the session commits the amended files and amendment_1.json; the owner approves it at
    their own terminal and commits the approval under their own name; the session imports it. Nothing runs until the
    import is pushed, and the readers' fingerprint (prereg_sha256) never changes."""

    def setUp(self):
        super().setUp()
        self.owner_commits()
        self.do_import()
        self.push_import()
        self.parent_facts = AC.require_approval(self.cfg)
        parent_files = json.loads((self.n80 / K.MANIFEST_NAME).read_text())["files"]
        (self.n80 / "code" / "x.py").write_text('"""Synthetic code."""\nx = 2\n', encoding="utf8")
        doc = self.n80 / "PREREGISTRATION.md"
        n = doc.stat().st_size
        doc.write_bytes(doc.read_bytes() + b"\n---\n\n## 19. Amendment 1 (synthetic)\n")
        files = {r: AC.file_sha256(self.n80 / r) for r in parent_files}
        body = dict(kind=AC.AMENDMENT_KIND, number=AC.AMENDMENT_NUMBER, label=K.LABEL, title="synthetic",
                    parent=dict(prereg_sha256=self.prereg.sha256,
                                code_manifest_sha256=self.prereg.payload["code"]["code_manifest_sha256"],
                                approval_sha256=self.parent_facts["approval_sha256"],
                                import_sha256=self.parent_facts["import_sha256"]),
                    files=files, delta=dict(changed={r: [parent_files[r], files[r]] for r in parent_files
                                                     if files[r] != parent_files[r]}, added={}, removed=[],
                                            appended={"PREREGISTRATION.md": dict(
                                                parent_bytes=n, parent_sha256=parent_files["PREREGISTRATION.md"])}),
                    desktop_check=TEST_CHECK_SHA)
        self.asha = AC.write_json_once(self.n80 / AC.AMENDMENT_NAME, body)
        self.a16 = self.asha[:K.PREFIX_LEN]
        self.a_name = AC.AMEND_APPROVAL_NAME.format(prefix=self.a16)
        self.i_name = AC.AMEND_IMPORT_NAME.format(prefix=self.a16)
        self.commit(AGENT, "Near-80: amendment 1 (synthetic)", ".")
        self.push()

    def owner_approves_amendment(self, author=APPROVER, committer=APPROVER, commit=True):
        check = K.N80_REL.rsplit("/", 1)[0] + "/" + write_desktop_check(self.n80, self.asha)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = AL.main([], folder=str(self.n80), script_path=str(self.n80 / "approve_local_near80.py"),
                           stdin=Tty(), input_fn=lambda _: self.a16, repo=str(self.repo))
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn(f"Near-80: approve amendment 1 {self.a16}", out.getvalue())
        if not commit:
            return None
        rels = [f"{K.N80_REL}/{self.a_name}", f"{K.N80_REL}/{self.a_name}.sha256", check, check + ".sha256"]
        c = self.commit(committer, f"Near-80: approve amendment 1 {self.a16}", *rels, author=author)
        self.push()
        return c

    def import_amendment(self):
        return IA.import_amendment(self.cfg, log=quiet)

    def push_amendment_import(self):
        rels = [f"{K.N80_REL}/{self.i_name}", f"{K.N80_REL}/{self.i_name}.sha256"]
        self.commit(AGENT, "Near-80: import record of amendment 1", *rels)
        self.push()

    def amendment_import_refused(self, why):
        before = tree(self.n80)
        with self.assertRaises(AC.ApprovalRefused) as cm:
            self.import_amendment()
        msg = str(cm.exception)
        self.assertTrue(msg.startswith("refused: "), msg)
        self.assertIn(why, msg)
        self.assertEqual(tree(self.n80), before)
        return msg

    def require_refused(self, why):
        with self.assertRaises(AC.ApprovalRefused) as cm:
            AC.require_approval(self.cfg)
        self.assertIn(why, str(cm.exception))

    def test_good_chain(self):
        self.require_refused(f"no approval of amendment 1 ({self.a16})")     # amended code: nothing runs before it
        self.amendment_import_refused("no approval of amendment 1")
        commit = self.owner_approves_amendment()
        body = self.import_amendment()
        self.assertEqual((body["commit"], body["author"], body["committer"]), (commit, APPROVER, APPROVER))
        self.assertEqual((body["kind"], body["prereg_sha256"], body["amendment_sha256"], body["approval_file"]),
                         (AC.AMEND_IMPORT_KIND, self.prereg.sha256, self.asha, AC.amend_approval_rel(self.a16)))
        self.assertEqual(body["approval_file_sha256"], AC.file_sha256(self.n80 / self.a_name))
        self.assertIn("Check on GitHub that this commit is yours", body["note"])
        self.require_refused("not on origin's branch")                        # the import is not pushed yet
        self.push_amendment_import()
        got = AC.require_approval(self.cfg)
        self.assertEqual(got["prereg_sha256"], self.prereg.sha256)           # the readers' fingerprint is unchanged
        self.assertEqual({k: v for k, v in got.items() if k != "amendment"}, self.parent_facts)
        a = got["amendment"]
        self.assertEqual((a["sha256"], a["prefix"], a["commit"], a["author"]), (self.asha, self.a16, commit, APPROVER))
        self.assertEqual(a["import_sha256"], AC.file_sha256(self.n80 / self.i_name))
        self.amendment_import_refused("exists already")                        # written once

    def test_refuses_the_agent_as_author(self):
        self.owner_approves_amendment(author=AGENT)
        self.amendment_import_refused("own git identity")

    def test_refuses_the_agent_as_committer(self):
        self.owner_approves_amendment(committer=AGENT)
        self.amendment_import_refused("own git identity")

    def test_refuses_a_committer_who_is_not_the_owner(self):
        self.owner_approves_amendment(committer="Synthetic User <user@example.org>")
        self.amendment_import_refused("not the owner")

    def test_refuses_two_adding_commits(self):
        self.owner_approves_amendment()
        rels = [f"{K.N80_REL}/{self.a_name}", f"{K.N80_REL}/{self.a_name}.sha256"]
        git(self.repo, "rm", "-q", *rels)
        git(self.repo, "commit", "-q", "-m", "remove", committer=APPROVER)
        git(self.repo, "checkout", "HEAD~1", "--", *rels)
        git(self.repo, "commit", "-q", "-m", "add again", committer=APPROVER)
        self.push()
        self.amendment_import_refused("exactly one commit")

    def test_refuses_an_approval_of_another_amendment(self):
        self.owner_approves_amendment(commit=False)
        p = self.n80 / self.a_name
        body = json.loads(p.read_text())
        body["amendment_sha256"] = "f" * 64
        write_json_sidecar(p, body)
        rels = [f"{K.N80_REL}/{self.a_name}", f"{K.N80_REL}/{self.a_name}.sha256"]
        self.commit(APPROVER, f"Near-80: approve amendment 1 {self.a16}", *rels)
        self.push()
        self.amendment_import_refused("approves another version of amendment_1.json")

    def test_refuses_a_commit_whose_tree_holds_another_amended_file(self):
        self.owner_approves_amendment(commit=False)
        x = self.n80 / "code" / "x.py"
        good = x.read_bytes()
        x.write_text("x = 99\n", encoding="utf8")
        rels = [f"{K.N80_REL}/{self.a_name}", f"{K.N80_REL}/{self.a_name}.sha256", f"{K.N80_REL}/code/x.py"]
        self.commit(APPROVER, "approve with a change", *rels)
        self.push()
        x.write_bytes(good)                                                    # the working tree is as amended
        self.amendment_import_refused("does not hold the pinned near-80 files")

    def test_refuses_an_owner_commit_without_the_desktop_check(self):
        self.owner_approves_amendment(commit=False)
        rels = [f"{K.N80_REL}/{self.a_name}", f"{K.N80_REL}/{self.a_name}.sha256"]   # the result file is left out
        self.commit(APPROVER, f"Near-80: approve amendment 1 {self.a16}", *rels)
        self.push()
        self.amendment_import_refused("does not hold the passing desktop check")
        self.require_refused("no import record")

    def test_refuses_an_approval_that_names_a_result_of_another_check(self):
        self.owner_approves_amendment(commit=False)
        other = write_desktop_check(self.n80, self.asha, name="desktop_check_20261019T000009Z.json",
                                    check_sha256=dict(TEST_CHECK_SHA, sh="3" * 64))
        crel = AL.CHECK_POSIX + "/" + Path(other).name
        p = self.n80 / self.a_name
        body = json.loads(p.read_text())
        body["desktop_checks"] = {crel: AC.file_sha256(self.n80.parent / other)}  # as if the script had named it
        write_json_sidecar(p, body)
        rels = [f"{K.N80_REL}/{self.a_name}", f"{K.N80_REL}/{self.a_name}.sha256", crel, crel + ".sha256"]
        self.commit(APPROVER, f"Near-80: approve amendment 1 {self.a16}", *rels)
        self.push()
        self.amendment_import_refused("made by another check than the one amendment_1.json names")

    def test_refuses_a_change_after_approval(self):
        self.owner_approves_amendment()
        self.import_amendment()
        self.push_amendment_import()
        AC.require_approval(self.cfg)
        (self.n80 / "README.md").write_text("changed\n", encoding="utf8")
        self.require_refused("README.md differs")

    def test_refuses_an_import_that_names_another_amendment(self):
        self.owner_approves_amendment()
        self.import_amendment()
        p = self.n80 / self.i_name
        body = json.loads(p.read_text())
        body["amendment_sha256"] = "f" * 64
        write_json_sidecar(p, body)
        self.push_amendment_import()
        self.require_refused("does not match its approval (amendment_sha256)")


class TestApproveLocalAmendment(Isolated):
    """approve_local_near80.py's amendment mode on a synthetic folder whose prereg.json is approved and imported."""

    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "repo"
        self.prereg = build_n80(self.repo)
        self.n80 = self.repo / K.N80_REL
        p16 = self.prereg.prefix
        self.p_approval = AC.approval_name(p16)
        self.p_import = AC.import_name(p16)
        asha_p = write_json_sidecar(self.n80 / self.p_approval, {"kind": K.LOCAL_KIND,
                                                                 "prereg_sha256": self.prereg.sha256})
        isha_p = write_json_sidecar(self.n80 / self.p_import, {"kind": K.IMPORT_KIND,
                                                               "prereg_sha256": self.prereg.sha256})
        self.parent_files = json.loads((self.n80 / K.MANIFEST_NAME).read_text())["files"]
        (self.n80 / "code" / "x.py").write_text('"""Synthetic code."""\nx = 2\n', encoding="utf8")
        doc = self.n80 / "PREREGISTRATION.md"
        self.n = doc.stat().st_size
        doc.write_bytes(doc.read_bytes() + b"\n---\n\n## 19. Amendment 1 (test)\n")
        files = {r: AC.file_sha256(self.n80 / r) for r in self.parent_files}
        self.body = dict(kind=AC.AMENDMENT_KIND, number=AC.AMENDMENT_NUMBER, label=K.LABEL, title="test amendment",
                         parent=dict(prereg_sha256=self.prereg.sha256,
                                     code_manifest_sha256=self.prereg.payload["code"]["code_manifest_sha256"],
                                     approval_sha256=asha_p, import_sha256=isha_p),
                         files=files,
                         delta=dict(changed={r: [self.parent_files[r], files[r]] for r in self.parent_files
                                             if files[r] != self.parent_files[r]}, added={}, removed=[],
                                    appended={"PREREGISTRATION.md": dict(
                                        parent_bytes=self.n, parent_sha256=self.parent_files["PREREGISTRATION.md"])}),
                         unchanged=["prereg.json"], runs=["the failed phases"],
                         cleared_failures=["critic.main.stall.s10"], why_no_result_changes="the hash chain",
                         desktop_check=TEST_CHECK_SHA)
        self.write_amendment()
        self.check_rel = write_desktop_check(self.n80, self.asha)

    def write_amendment(self):
        self.asha = write_json_sidecar(self.n80 / AC.AMENDMENT_NAME, self.body)
        self.a16 = self.asha[:K.PREFIX_LEN]
        self.a_name = AC.AMEND_APPROVAL_NAME.format(prefix=self.a16)

    def run_al(self, args=(), stdin=None, typed=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = AL.main(list(args), folder=str(self.n80), script_path=str(self.n80 / "approve_local_near80.py"),
                           stdin=Tty() if stdin is None else stdin,
                           input_fn=lambda _: self.a16 if typed is None else typed, repo=str(self.repo))
        return code, out.getvalue()

    def refused(self, why, **kw):
        before = tree(self.n80)
        code, text = self.run_al(**kw)
        self.assertEqual(code, 2, text)
        self.assertIn("Refused.", text)
        self.assertIn("Nothing was written.", text)
        self.assertIn(why, " ".join(text.split()))
        self.assertEqual(tree(self.n80), before)
        return text

    def test_writes_exactly_the_two_files_once(self):
        before = {p for p, _ in tree(self.n80)}
        code, text = self.run_al()
        self.assertEqual(code, 0, text)
        self.assertEqual({p for p, _ in tree(self.n80)} - before, {self.a_name, self.a_name + ".sha256"})
        raw = (self.n80 / self.a_name).read_bytes()
        self.assertEqual(AC.sidecar_digest(self.n80 / self.a_name), AC.sha256_bytes(raw))
        amendment = AC.Amendment(self.n80 / AC.AMENDMENT_NAME, self.asha, self.body)
        body = AC.check_local_amendment_approval(raw, self.prereg, amendment)
        self.assertEqual(body["statement"], AC.AMEND_STATEMENT.format(sha=self.asha, parent=self.prereg.sha256))
        self.assertEqual((body["typed"], body["parent_prereg_sha256"]), (self.a16, self.prereg.sha256))
        self.assertNotIn("prereg_sha256", body)
        check = self.n80.parent / self.check_rel
        self.assertEqual(body["desktop_checks"], {AL.CHECK_POSIX + "/" + check.name: AC.file_sha256(check)})
        self.assertIn(self.asha, text)
        rel = f"{K.N80_REL}/{self.a_name}"
        self.assertIn(f"git add {rel} {rel}.sha256 {AL.CHECK_POSIX}/desktop_check_*", text)
        self.assertIn(f"git -c user.name=\"{K.APPROVER_NAME}\" -c user.email={K.APPROVER_EMAIL} commit -m "
                      f"\"Near-80: approve amendment 1 {self.a16}\"", text)
        self.assertIn(AC.AMEND_IMPORT_NAME.format(prefix=self.a16), text)
        self.refused("already exists")                                     # never overwrites

    def test_check_writes_nothing_and_shows_each_change(self):
        before = tree(self.n80)
        code, text = self.run_al(args=["--check"], stdin=NotTty())
        self.assertEqual(code, 0, text)
        flat = " ".join(text.split())
        self.assertIn("Nothing was asked and nothing was written", flat)
        self.assertIn("Near-80 (version 8): amendment 1", flat)
        for rel, (old, new) in self.body["delta"]["changed"].items():
            self.assertIn(f"{rel}: {old[:16]} -> {new[:16]}", flat)
        self.assertIn(f"appended to; the first {self.n} bytes are the approved text", flat)
        self.assertIn(self.asha, flat)
        self.assertIn("PREREGISTRATION.md, section 19", flat)
        self.assertIn("The 1 phases that failed and run again as attempt 2 (their failed attempt 1 stays in the ledger "
                      "and counts toward the budget):", flat)
        self.assertNotIn("clears", flat)
        self.assertIn(f"passed: {Path(self.check_rel).name} (official mode, this amendment_1.json, that code; the four "
                      f"unit test files ran and passed)", flat)
        self.assertIn(f"its code, as amendment_1.json names it: check_amendment1.py {'1' * 16}, check_amendment1.sh "
                      f"{'2' * 16}", flat)
        self.assertIn("Approving also confirms the item of section 19.6 that is marked for you: to your knowledge "
                      "nobody, you included, opened a record of the exploratory stage that holds a score or a learner "
                      "number", flat)
        self.assertIn("looks/exploratory.json and exploratory.md, and the records of seeds 0-4 under learners/, "
                      "policy/, eval/, readouts/ and stops/", flat)
        self.assertIn("If anyone did, press Enter to stop and tell the session; it states it in section 19.6 and "
                      "gives you a new fingerprint.", flat)
        self.assertLess(flat.index("Before you type:"), flat.index("Fingerprint (SHA-256) of amendment_1.json:"))
        self.assertEqual(tree(self.n80), before)

    def test_refuses_without_a_passing_desktop_check(self):
        check = self.n80.parent / self.check_rel
        variants = [("none", None), ("failed", dict(passed=False)), ("a stub run", dict(mode="stub")),
                    ("another amendment", dict(amendment_sha256="f" * 64)), ("another kind", dict(kind="x")),
                    ("a changed result file", "edit"),
                    ("no unit tests (--no-tests)", dict(unit_tests={}, tests_run=False)),
                    ("no unit_tests field", dict(unit_tests=None)),
                    ("three of the four unit test files", dict(unit_tests={n: 0 for n in K.TEST_FILES[:3]})),
                    ("a failing unit test file", dict(unit_tests=dict({n: 0 for n in K.TEST_FILES},
                                                                      **{K.TEST_FILES[0]: 1}))),
                    ("exit codes that are booleans", dict(unit_tests={n: False for n in K.TEST_FILES})),
                    ("another check code", dict(check_sha256=dict(TEST_CHECK_SHA, py="3" * 64))),
                    ("no check code", dict(check_sha256=None))]
        for name, change in variants:
            with self.subTest(case=name):
                for p in check.parent.iterdir():
                    p.unlink()
                if isinstance(change, dict):
                    write_desktop_check(self.n80, self.asha, **change)
                elif change == "edit":
                    write_desktop_check(self.n80, self.asha)
                    check.write_bytes(check.read_bytes() + b" ")
                self.refused("No desktop check of this amendment has passed")
                code, text = self.run_al(args=["--check"], stdin=NotTty())  # --check reports it and passes
                self.assertEqual(code, 0, text)
                self.assertIn("none has passed yet", " ".join(text.split()))
        for p in check.parent.iterdir():
            p.unlink()
        write_desktop_check(self.n80, self.asha, name="desktop_check_20261019T000001Z.json", passed=False)
        write_desktop_check(self.n80, self.asha, name="desktop_check_20261019T000002Z.json")
        code, text = self.run_al()                                         # one passing result file is enough
        self.assertEqual(code, 0, text)

    def test_refuses_a_malformed_amendment(self):
        good = json.loads(json.dumps(self.body))
        for key, value, in_delta in (("delta", [1], False), ("parent", [1], False), ("changed", [1], True),
                                     ("appended", {"PREREGISTRATION.md": 5}, True), ("cleared_failures", 5, False),
                                     ("unchanged", 3, False), ("runs", [1], False), ("title", ["x"], False),
                                     ("why_no_result_changes", 7, False), ("desktop_check", {"py": 1}, False)):
            with self.subTest(case=key):
                self.body = json.loads(json.dumps(good))
                (self.body["delta"] if in_delta else self.body)[key] = value
                self.write_amendment()
                self.refused("amendment_1.json is malformed", args=["--check"], stdin=NotTty())

    def test_refuses_an_amendment_of_another_study(self):
        self.body["label"] = "another"
        self.write_amendment()
        self.refused("amendment_1.json names another study", args=["--check"], stdin=NotTty())

    def test_refuses_an_amendment_that_names_no_check_code(self):
        del self.body["desktop_check"]
        self.write_amendment()
        for p in (self.n80.parent / self.check_rel).parent.iterdir():
            p.unlink()
        self.check_rel = write_desktop_check(self.n80, self.asha)
        self.refused("No desktop check of this amendment has passed")

    def test_refuses_an_added_or_removed_file(self):
        good = json.loads(json.dumps(self.body))
        extra = self.n80 / "code" / "extra.py"
        extra.write_text("x = 1\n", encoding="utf8")
        for change in (dict(added={"code/extra.py": AC.file_sha256(extra)}), dict(removed=["code/x.py"])):
            with self.subTest(change=sorted(change)):
                self.body = json.loads(json.dumps(good))
                self.body["delta"].update(change)
                self.write_amendment()
                self.refused("amendment 1 adds and removes no file", args=["--check"], stdin=NotTty())

    def test_refuses_a_wrong_typed_prefix(self):
        self.refused("do not match", typed="0" * 16)
        self.refused("do not match", typed=self.prereg.prefix)            # the parent's fingerprint is not it

    def test_refuses_an_existing_approval(self):
        (self.n80 / self.a_name).write_text("{}", encoding="utf8")
        self.refused("already exists")

    def test_refuses_a_stray_file(self):
        (self.n80 / "stray.txt").write_text("x", encoding="utf8")
        self.refused("stray.txt is not a file of this pre-registration", args=["--check"], stdin=NotTty())

    def test_refuses_a_missing_parent_approval(self):
        for name in (self.p_approval, self.p_approval + ".sha256"):
            (self.n80 / name).unlink()
        self.refused("is missing: an amendment is approved only on top of your approval", args=["--check"],
                     stdin=NotTty())

    def test_refuses_another_parent_import_record(self):
        write_json_sidecar(self.n80 / self.p_import, {"kind": K.IMPORT_KIND, "prereg_sha256": "f" * 64})
        self.refused(f"{self.p_import} is not the one amendment_1.json names", args=["--check"], stdin=NotTty())

    def test_refuses_a_changed_amendment_sidecar(self):
        (self.n80 / (AC.AMENDMENT_NAME + ".sha256")).write_text("0" * 64 + "  amendment_1.json\n", encoding="utf8")
        self.refused("amendment_1.json does not match its fingerprint file", args=["--check"], stdin=NotTty())

    def test_refuses_an_unnamed_change(self):
        (self.n80 / "README.md").write_text("changed\n", encoding="utf8")
        self.refused("README.md differs", args=["--check"], stdin=NotTty())

    def test_refuses_a_delta_that_hides_a_change(self):
        self.body["delta"]["changed"] = {}
        self.write_amendment()
        self.refused("are not the ones its delta names", args=["--check"], stdin=NotTty())

    def test_refuses_a_document_edited_in_the_middle(self):
        doc = self.n80 / "PREREGISTRATION.md"
        data = doc.read_bytes()
        doc.write_bytes(data[:10] + b"X" + data[10:])
        new = AC.file_sha256(doc)
        self.body["files"]["PREREGISTRATION.md"] = new
        self.body["delta"]["changed"]["PREREGISTRATION.md"] = [self.parent_files["PREREGISTRATION.md"], new]
        self.write_amendment()
        self.refused("PREREGISTRATION.md may only be appended to", args=["--check"], stdin=NotTty())

    def test_constants_equal_approval_checks(self):
        self.assertEqual((AL.AMENDMENT_NAME, AL.AMENDMENT_KIND, AL.AMEND_KIND, AL.AMEND_STATEMENT, AL.NEVER_AMENDED),
                         (AC.AMENDMENT_NAME, AC.AMENDMENT_KIND, AC.AMEND_LOCAL_KIND, AC.AMEND_STATEMENT,
                          AC.NEVER_AMENDED))
        self.assertEqual(AL.AMEND_APPROVAL_NAME.format("p"), AC.AMEND_APPROVAL_NAME.format(prefix="p"))
        self.assertEqual(AL.AMEND_IMPORT_NAME.format("p"), AC.AMEND_IMPORT_NAME.format(prefix="p"))
        self.assertEqual(AC.AMENDMENT_NUMBER, 1)
        self.assertEqual((AL.CHECK_KIND, AL.CHECK_POSIX + "/desktop_check_"),
                         (AC.AMEND_CHECK_KIND, AC.AMEND_CHECK_PREFIX))
        self.assertEqual(AL.CHECK_TESTS, K.TEST_FILES)


# ============================================================================================ freeze.py

def real_draft_bytes():
    top = K.N80 / "prereg_draft.json"
    if top.is_file():
        return top.read_bytes()
    kept = [p for p in (K.N80 / K.HISTORY_DIR).glob("prereg_draft_*.json") if p.suffix == ".json"]
    return kept[0].read_bytes()


def make_n80_copy(dst, claims_real=False):
    """A temporary copy of the near-80 folder for the freeze: the real documents, the real draft and A's code; the
    other builders' files as stubs (claims_text.py real when asked)."""
    dst = Path(dst)
    (dst / "code").mkdir(parents=True)
    for name in ("PREREGISTRATION.md", "README.md", "approve_local_near80.py"):
        shutil.copyfile(K.N80 / name, dst / name)
    (dst / "run_near80.sh").write_text("#!/bin/sh\n# a stub for a freeze test\n", encoding="utf8")
    (dst / "push_runfiles_near80.sh").write_text("#!/bin/sh\n# a stub for a freeze test\n", encoding="utf8")
    (dst / "prereg_draft.json").write_bytes(real_draft_bytes())
    for name in K.CODE_FILES:
        if name in A_FILES or (claims_real and name == "claims_text.py"):
            shutil.copyfile(K.CODE / name, dst / "code" / name)
        else:
            (dst / "code" / name).write_text(f'"""A stub of {name} for a freeze test."""\n', encoding="utf8")
    return dst


def fake_tests(fail=()):
    def fn(n80_root, log):
        fn.calls += 1
        return {n: (1 if n in fail else 0) for n in K.TEST_FILES}
    fn.calls = 0
    return fn


def fake_facts(passed=True, seeds=None):
    def fn(n80_root, out_path, log):
        rules = [dict(rule="check 1: the band is really missing (leak check)", scope="stall/seed0", value=True,
                      threshold="all checks pass", passed=True, seen_on_0_4="pass"),
                 dict(rule="check 2: the evaluation split keeps band rows", scope="stall/seed0",
                      value=1051 if passed else 120, threshold=">= 300", passed=passed, seen_on_0_4="775-1,072")]
        payload = dict(status=K.STATUS_LINE_FINAL, label="EXPLORATORY no-training facts, seeds 0-4",
                       band=dict(lo=K.BAND_LO, hi=K.BAND_HI, scope=K.BAND_SCOPE),
                       seeds=list(seeds or K.EXPLORATORY), worlds=list(K.WORLDS), results=[], fixed_banks={},
                       anchors={}, no_training_checks=dict(rules=rules, not_run=[], passed=passed), seconds=0.0)
        Path(out_path).write_text(json.dumps(payload, indent=1), encoding="utf8")
        fn.calls.append(Path(out_path))
    fn.calls = []
    return fn


def fake_claims(text):
    return {"source": "PREREGISTRATION.md section 14", "synthetic": True, "length": len(text)}


class FreezeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80_test_freeze_"))
        self.n80 = make_n80_copy(self.tmp / "near80")
        self.out = self.tmp / "out"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, **kw):
        args = dict(n80_root=self.n80, out_root=self.out, repo=K.ROOT, tests_fn=fake_tests(), facts_fn=fake_facts(),
                    claims_fn=fake_claims, sentence_fn=lambda rules: "STOP SENTENCE", log=quiet)
        args.update(kw)
        return FZ.write("a freeze test", **args)

    def refused(self, why=None, **kw):
        with self.assertRaises(FZ.FreezeRefused) as cm:
            self.write(**kw)
        if why:
            self.assertIn(why, str(cm.exception))
        self.assertFalse((self.n80 / K.PREREG_NAME).exists())
        return str(cm.exception)


class TestFreeze(FreezeBase):
    def test_write(self):
        draft_sha = AC.sha256_bytes(real_draft_bytes())
        facts = fake_facts()
        body = self.write(facts_fn=facts)
        # step 4: the draft moved into history with a sidecar
        self.assertFalse((self.n80 / "prereg_draft.json").exists())
        kept = self.n80 / K.HISTORY_DIR / K.DRAFT_HISTORY_NAME.format(prefix=draft_sha[:16])
        self.assertEqual(AC.file_sha256(kept), draft_sha)
        self.assertEqual(AC.sidecar_digest(kept), draft_sha)
        # step 5: the manifest lists exactly the planned files
        man = AC.read_json_verified(self.n80 / K.MANIFEST_NAME)
        self.assertEqual(set(man["files"]), set(K.TOP_FILES) | {"code/" + n for n in K.CODE_FILES})
        self.assertEqual(man["kind"], K.MANIFEST_KIND)
        self.assertTrue(set(K.PIN_FILES) <= set(man["repository_files"]))
        self.assertFalse(any("__pycache__" in r for r in man["repository_files"]))
        self.assertTrue(all(any(r.startswith(d + "/") for d in K.PIN_DIRS) or r in K.PIN_FILES
                            for r in man["repository_files"]))
        self.assertTrue(all(Path(r).suffix in K.PIN_SUFFIXES for r in man["repository_files"]))
        self.assertIn("outputs/shift/beyond-the-data/stage2/code/stage2_common.py", man["repository_files"])
        # step 7: the facts ran once, under this manifest's name, and were copied
        msha = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
        self.assertEqual(facts.calls, [self.out / "pre" / f"no_training_checks_{msha[:16]}.json"])
        frozen = self.n80 / NP.FROZEN_REL
        self.assertEqual(frozen.read_bytes(), facts.calls[0].read_bytes())
        self.assertEqual(AC.sidecar_digest(frozen), AC.file_sha256(frozen))
        # step 8: prereg.json
        on_disk = AC.read_json_verified(self.n80 / K.PREREG_NAME)
        self.assertEqual(on_disk, body)
        for key in ("kind", "version", "label", "status", "status_line", "approvable", "missing", "owner_answer",
                    "draft", "choices", "constants", "claims_word_for_word", "code", "pins",
                    "no_training_checks_frozen", "approval", "runner") + NP.CONTENT_KEYS:
            self.assertIn(key, body)
        self.assertEqual(body["kind"], K.PREREG_KIND)
        self.assertEqual(body["status"], K.STATUS_FINAL)
        self.assertEqual(body["status_line"], K.STATUS_LINE_FINAL)
        self.assertEqual((body["approvable"], body["missing"]), (True, []))
        self.assertEqual(body["owner_answer"], K.OWNER_ANSWER)
        self.assertEqual(body["draft"]["path"], f"{K.HISTORY_DIR}/{kept.name}")
        self.assertEqual(body["draft"]["sha256"], draft_sha)
        self.assertEqual(body["constants"], AC.canonical_constants())
        self.assertEqual(body["claims_word_for_word"], fake_claims((self.n80 / "PREREGISTRATION.md").read_text()))
        self.assertEqual(body["code"], {"code_manifest": K.MANIFEST_NAME, "code_manifest_sha256": msha,
                                        "files": man["files"]})
        self.assertEqual(set(body["pins"]), set(AC.PINS_REL) | {"repository_files_count"})
        self.assertEqual(body["pins"]["v6_stage2_common_sha256"], AC.V6_STAGE2_COMMON_SHA256)
        self.assertEqual(body["pins"]["repository_files_count"], len(man["repository_files"]))
        self.assertEqual(body["no_training_checks_frozen"],
                         {"path": NP.FROZEN_REL, "sha256": AC.file_sha256(frozen), "passed": True, "rules": 2,
                          "not_run": [], "code_manifest_sha256": msha})
        self.assertEqual(body["approval"]["approver_email"], K.APPROVER_EMAIL)
        self.assertEqual(body["approval"]["branch"], K.BRANCH)
        self.assertIs(body["approval"]["local_terminal_allowed"], True)
        self.assertIs(body["approval"]["accepts_defaults"], True)
        self.assertEqual(body["runner"]["phases"], list(K.PHASE_KINDS))
        self.assertEqual(body["runner"]["lanes_default"], 10)
        draft = json.loads(real_draft_bytes())
        self.assertEqual(len(body["choices"]), len(draft["choices"]))
        for c, d in zip(body["choices"], draft["choices"]):
            self.assertEqual(c["taken"], d["default"])
            self.assertEqual(c["accepted"], K.OWNER_ANSWER["utc"])
        # the arm names of the code, and no draft wording
        self.assertEqual(body["arms"]["logical_ids"], {k: list(v) for k, v in K.ARMS.items()})
        text = json.dumps({k: v for k, v in body.items() if k != "draft"})
        for old in NP.ARM_RENAMES:
            self.assertNotIn(old, text)
        rivals = body["arms"]["near80_both_worlds"]["world_blind_rivals"]
        self.assertTrue({"random_1", "random_2", "standin", "even_pull", "obca_forced"} <= set(rivals))
        self.assertIn("range_hinge", body["arms"]["near80_both_worlds"]["reported"])
        self.assertEqual(NP.draft_words({k: v for k, v in body.items() if k not in ("draft", "claims_word_for_word")}),
                         [])
        # the approval chain and the owner's script accept it
        cfg = AC.ApprovalConfig(n80_root=self.n80, repo=K.ROOT)
        prereg = AC.load_prereg(cfg)
        AC.check_constants(prereg)
        manifest = AC.check_code(prereg, cfg)
        AC.check_pins(manifest, cfg)
        AC.check_prereg_pins(prereg, manifest, cfg)
        AL.check_folder(str(self.n80), str(self.n80 / "approve_local_near80.py"), str(K.ROOT))
        # a second write refuses and changes nothing
        before = tree(self.n80)
        with self.assertRaises(FZ.FreezeRefused) as cm:
            self.write()
        self.assertIn("exists already", str(cm.exception))
        self.assertEqual(tree(self.n80), before)

    def test_refuses_draft_documents(self):
        doc = self.n80 / "PREREGISTRATION.md"
        lines = doc.read_text().split("\n")
        lines[2] = "**DRAFT for the owner's approval. Nothing has been trained.**"
        doc.write_text("\n".join(lines))
        before = tree(self.n80)
        self.refused("**DRAFT")
        self.assertEqual(tree(self.n80), before)                           # nothing moved, nothing written
        cases = {
            "title": lambda t: t.replace("# Hiding the commands near 80%: pre-registration", "# A title", 1),
            "this draft": lambda t: t + "\nSee this draft.\n",
            "14.6": lambda t: t.replace("### 14.6 How the code fills in the sentences", "### How", 1),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                shutil.rmtree(self.n80)
                self.n80 = make_n80_copy(self.tmp / f"near80_{name.replace(' ', '_')}")
                p = self.n80 / "PREREGISTRATION.md"
                p.write_text(change(p.read_text()))
                self.refused()
        shutil.rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_readme")
        p = self.n80 / "README.md"
        p.write_text(p.read_text().replace(f"**{K.STATUS_LINE_FINAL}**", "**DRAFT.**"))
        self.refused("README.md")
        shutil.rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_code")
        (self.n80 / "code" / "pooled.py").write_text('"""DRAFT looks."""\n')
        self.refused("DRAFT in its first docstring line")

    def test_refuses_an_existing_prereg_or_approval_or_output(self):
        for name, make in (("prereg", lambda: (self.n80 / K.PREREG_NAME).write_text("{}")),
                           ("approval", lambda: (self.n80 / "approval_near80_0123456789abcdef.json").write_text("{}")),
                           ("import", lambda: (self.n80 / "import_near80_0123456789abcdef.json").write_text("{}")),
                           ("output", lambda: (self.out / "data").mkdir(parents=True))):
            with self.subTest(case=name):
                shutil.rmtree(self.n80)
                shutil.rmtree(self.out, ignore_errors=True)
                self.n80 = make_n80_copy(self.tmp / f"near80_{name}")
                make()
                with self.assertRaises(FZ.FreezeRefused):
                    self.write()
                self.assertTrue((self.n80 / "prereg_draft.json").exists())
        shutil.rmtree(self.out, ignore_errors=True)
        (self.out / "pre").mkdir(parents=True)                              # pre/ alone is allowed
        self.assertEqual(FZ.existing_problems(make_n80_copy(self.tmp / "near80_pre"), self.out), [])

    def test_refuses_an_unplanned_code_folder(self):
        (self.n80 / "code" / "extra.py").write_text("x = 1\n")
        self.refused("extra.py is not one of the planned code files")
        (self.n80 / "code" / "extra.py").unlink()
        (self.n80 / "code" / "__pycache__").mkdir()
        self.refused("__pycache__")
        (self.n80 / "code" / "__pycache__").rmdir()
        (self.n80 / "code" / "stubs.py").unlink()
        self.refused("code/stubs.py is missing")

    def test_refuses_failing_unit_tests_then_a_rerun_passes(self):
        self.refused("unit tests failed", tests_fn=fake_tests(fail=("test_runner.py",)))
        self.assertFalse((self.n80 / "prereg_draft.json").exists())        # the draft was moved before the tests
        man1 = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
        self.write()                                                        # the rerun checks the kept draft
        self.assertEqual(AC.file_sha256(self.n80 / K.MANIFEST_NAME), man1)  # an unchanged manifest is kept
        self.assertFalse((self.n80 / K.MANIFEST_HISTORY_DIR).exists())

    def test_a_changed_file_rewrites_the_manifest_with_history(self):
        self.refused("unit tests failed", tests_fn=fake_tests(fail=("test_pooled.py",)))
        man1 = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
        (self.n80 / "code" / "stubs.py").write_text('"""A changed stub."""\n')
        self.write()
        man = AC.read_json_verified(self.n80 / K.MANIFEST_NAME)
        self.assertEqual(man["history"][0]["sha256"], man1)
        archived = self.n80 / K.MANIFEST_HISTORY_DIR / "code_manifest_v1.json"
        self.assertEqual(AC.file_sha256(archived), man1)
        self.assertEqual(AC.sidecar_digest(archived), man1)

    def test_refuses_failing_no_training_checks(self):
        sentences = []

        def sentence(rules):
            sentences.append(rules)
            return "STOP SENTENCE"
        msg = self.refused("STOP SENTENCE", facts_fn=fake_facts(passed=False), sentence_fn=sentence)
        self.assertIn("check 2: the evaluation split keeps band rows", msg)
        self.assertEqual(sentences[0][1]["passed"], False)
        self.assertFalse((self.n80 / NP.FROZEN_REL).exists())

    def test_refuses_facts_on_other_seeds(self):
        self.refused("not 0-4", facts_fn=fake_facts(seeds=[0, 1]))

    def test_default_sentence_names_the_first_failed_rule(self):
        rules = [dict(rule="check 1: x", passed=True, value=1, threshold="t"),
                 dict(rule="check 3: the band is the planned size", passed=False, value=0.071, threshold="0.01-0.06")]
        text = FZ.default_sentence(rules)
        self.assertTrue(text.startswith("On seeds 0-4, before any training, the near-80 data failed "), text)
        self.assertIn("Nothing was trained.", text)

    def test_facts_from_needs_this_manifests_name(self):
        wrong = self.tmp / "no_training_checks_0000000000000000.json"
        wrong.write_text("{}")
        self.refused("does not carry this manifest's first 16 characters", facts_from=wrong)

    def test_reuses_facts_already_written_under_this_manifest(self):
        self.refused("unit tests failed", tests_fn=fake_tests(fail=("test_near80.py",)))
        msha = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
        path = self.out / "pre" / f"no_training_checks_{msha[:16]}.json"
        path.parent.mkdir(parents=True)
        fake_facts()(self.n80, path, quiet)
        facts = fake_facts()
        self.write(facts_fn=facts)
        self.assertEqual(facts.calls, [])

    def test_check_writes_nothing(self):
        before = tree(self.n80)
        lines = []
        found = FZ.check(self.n80, self.out, log=lines.append)
        self.assertEqual(tree(self.n80), before)
        self.assertFalse(self.out.exists())
        self.assertIn("freeze.py write would:", lines[0])
        self.assertEqual([f for f in found if "claims_text.py" not in f], [])

    @unittest.skipUnless(CLAIMS_REAL, "code/claims_text.py is not written yet (builder C)")
    def test_claims_word_for_word(self):
        shutil.rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_claims", claims_real=True)
        body = self.write(claims_fn=None)
        mod = NP.load_claims_text(self.n80 / "code")
        doc = (self.n80 / "PREREGISTRATION.md").read_text(encoding="utf8")
        self.assertEqual(body["claims_word_for_word"], json.loads(AC.canonical(mod.extract_section14(doc))))
        self.assertEqual(body["claims_word_for_word"]["source"], "PREREGISTRATION.md section 14")


class TestFreezeOnTheRealFolder(unittest.TestCase):
    def test_the_real_documents_and_code_pass_steps_2_and_3_checks(self):
        """The real documents are final; the real code folder holds only planned files (some may still be missing
        while the three parts are built)."""
        self.assertEqual(FZ.document_problems(K.N80), [])
        problems = [p for p in FZ.code_folder_problems(K.N80) if not p.endswith(" is missing")]
        self.assertEqual(problems, [])


# ============================================================================================ near80_constants

class TestConstants(unittest.TestCase):
    def test_values_equal_the_draft_and_the_gate(self):
        import numpy
        from experiments.shift import thruster_chain as TC
        import near80lib as N
        self.assertEqual(K.KNEE32, float(numpy.float32(0.8)))
        self.assertEqual(K.KNEE32, TC.KNEE)
        self.assertEqual(K.CAP, 0.8)
        self.assertEqual(K.HOLD, N.HOLD)
        self.assertEqual((N.BAND_LO, N.BAND_HI, N.SCOPE), (K.BAND_LO, K.BAND_HI, K.BAND_SCOPE))
        self.assertEqual((N.EXPLORATORY_SEEDS, N.CONFIRMATORY_SEEDS, N.RESERVED_SEEDS),
                         (K.EXPLORATORY, K.CONFIRMATORY, K.RESERVED))
        self.assertEqual(tuple(TC.SITUATIONS), K.SITUATIONS)
        self.assertEqual(N.NO_TRAINING, dict(
            eval_band_rows_min=300, extra_share_min=0.01, extra_share_max=0.06, full_throttle_removed_min=0.99,
            kept_past80_share_max=0.002, identical_rule_min=0.998, region_kept_min=0.35, region_band_share_min=0.10,
            gap_lo_min=0.695, gap_lo_max=0.70, bank_min=500, stall_aware_band_share_min=0.50, hold_loss_min=15.0,
            data_edge_loss_min=10.0))
        draft = json.loads(real_draft_bytes())
        self.assertEqual(draft["kill_rules"]["no_training_checks"]["thresholds"], N.NO_TRAINING)
        self.assertEqual((draft["band"]["lo"], draft["band"]["hi"], draft["band"]["scope"]),
                         (K.BAND_LO, K.BAND_HI, K.BAND_SCOPE))
        self.assertEqual(draft["seeds"]["exploratory"], list(K.EXPLORATORY))
        self.assertEqual(draft["seeds"]["confirmatory"], list(K.CONFIRMATORY))
        self.assertEqual(draft["seeds"]["refused_always"], list(K.RESERVED))
        self.assertEqual(draft["measures"]["hold_level"], K.HOLD)
        self.assertEqual(draft["situations"]["worlds"], list(K.WORLDS))
        self.assertEqual(draft["situations"]["scored"], K.SCORED)
        self.assertEqual(set(draft["situations"]["test_start_sets"]), set(K.SITUATIONS))
        self.assertEqual(draft["compute"]["budget_cap"], K.CAP_CPU_HOURS)
        self.assertEqual(draft["compute"]["lanes_default"], K.DEFAULT_LANES)
        self.assertEqual(draft["label"], K.LABEL)
        choices = {c["number"]: c["default"] for c in draft["choices"]}
        self.assertEqual(choices[2], list(K.DEFAULTS["band"]))
        self.assertEqual(choices[5], K.DEFAULTS["region_floor"])
        self.assertEqual(choices[13], K.DEFAULTS["lanes"])
        self.assertEqual(choices[14], K.DEFAULTS["name"])
        self.assertEqual(choices[14], K.VERSION)
        self.assertEqual(choices[15], K.DEFAULTS["claim2_marks"])
        self.assertEqual(choices[9], K.DEFAULTS["after_poor_bca"])
        self.assertEqual(N.NO_TRAINING["region_kept_min"], K.DEFAULTS["region_floor"])

    def test_values_equal_step_1_and_step_2(self):
        s1 = json.loads((K.S1_DIR / "prespec.json").read_text())
        s2 = json.loads((K.S2_DIR / "prespec.json").read_text())
        self.assertEqual({k: s1["certificate"][k] for k in K.CERT}, K.CERT)
        self.assertEqual({k: s1["groups"]["gap_rule"][k] for k in K.GAP_RULE}, K.GAP_RULE)
        self.assertIn(f"[{K.CERT_TAG_HEAD[0]}, {K.CERT_TAG_HEAD[1]}, seed, w", s1["certificate"]["rng_tags"])
        self.assertEqual({k: s2["arms"]["rule"][k] for k in K.ARM_RULE}, K.ARM_RULE)
        self.assertEqual(set(s2["arms"]["rule"]), set(K.ARM_RULE))
        self.assertEqual(tuple(s2["arms"]["definitions"]), K.STEP2_RAMP_ARMS)
        self.assertEqual(list(s2["seeds"]), list(K.EXPLORATORY))
        p = s2["protocol"]
        self.assertEqual((p["alpha"], p["actor_steps"], p["chunk"]), (K.ALPHA, K.ACTOR_STEPS, K.CHUNK))
        v7 = load_path("near80_test_v7_common", K.V7_DIR / "code" / "v7_common.py")
        self.assertEqual(v7.PHASE_ENV, {k: v for k, v in K.PHASE_ENV.items() if k != "PYTHONPYCACHEPREFIX"})
        self.assertEqual(K.PHASE_ENV["PYTHONPYCACHEPREFIX"], K.PYCACHE_PREFIX)
        self.assertFalse(Path(K.PYCACHE_PREFIX).exists())
        self.assertEqual(v7.PYTHON, K.PYTHON)
        self.assertEqual(v7.APPROVER_EMAIL, K.APPROVER_EMAIL)
        self.assertEqual(v7.BRANCH, K.BRANCH)
        self.assertEqual(v7.W0_TOL, K.W0_TOL)
        self.assertEqual(v7.SCRATCH, K.SCRATCH)
        self.assertEqual(v7.R7, K.V7_R7)

    def test_layout_and_helpers(self):
        self.assertEqual(K.path("/r", "values", setting="near80", world="stall", seed=3),
                         Path("/r/eval/near80/stall/seed3/values.json"))
        self.assertEqual(K.phase_id("policy", "near80", "stall", seed=3), "policy.near80.stall.s3")
        self.assertEqual(K.phase_id("lock", group="exploratory"), "lock.x.x.exploratory")
        self.assertEqual(K.group_of(2), "exploratory")
        self.assertEqual(K.group_of(12), "confirmatory")
        for s in (5, 9, 15, -1):
            with self.assertRaises(ValueError):
                K.group_of(s)
        d = AC.canonical_constants()
        self.assertEqual(d["ARMS"]["random_rows"], list(K.ARMS["random_rows"]))
        self.assertEqual(d["ROOT"], str(K.ROOT))
        self.assertTrue(set(K.TEST_FILES) <= set(K.CODE_FILES))
        self.assertEqual(len(set(K.CODE_FILES)), len(K.CODE_FILES))
        self.assertTrue(set(K.STEP2_RAMP_ARMS) | set(K.NEW_RAMP_ARMS) <= set(K.ARMS["near80"]))
        self.assertEqual(set(K.ARM_PLAIN), set(K.ARMS["near80"]))
        self.assertEqual(set(K.PHASE_PLAIN), set(K.PHASE_KINDS))
        self.assertEqual(K.APPROVAL_NAME.format(prefix="p"), "approval_near80_p.json")
        self.assertEqual(K.N80, K.ROOT / K.N80_REL)

    def test_load_local_avoids_step_2s_modules(self):
        mod = K.load_local("near80_prereg")
        self.assertEqual(Path(mod.__file__), K.CODE / "near80_prereg.py")
        self.assertIs(K.load_local("near80_prereg"), mod)


if __name__ == "__main__":
    unittest.main(verbosity=2)
