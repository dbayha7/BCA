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
# The values DECISION.md fixes, as the freeze tests fix them (every test of a written prereg.json patches them in):
OWNER_ANSWER = dict(text="accept the defaults", utc="2026-10-21T00:00:00Z", where="a test", choices="every default")
MARKS_C2_FIXED = dict(net_margin=5.0, positive_seeds_min=4, multiplicity="all_three")
WORK_DIR = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()


def fixed_decision():
    """A context that fixes every placeholder of DECISION.md in near80_constants (the freeze refuses otherwise)."""
    return contextlib.ExitStack()


def enter_fixed(stack):
    stack.enter_context(mock.patch.object(K, "PLACEHOLDERS", {}))
    stack.enter_context(mock.patch.object(K, "MARKS_C2", dict(MARKS_C2_FIXED)))
    stack.enter_context(mock.patch.object(K, "OWNER_ANSWER", dict(OWNER_ANSWER)))
    return stack


def quiet(*_a, **_k):
    return None


def rmtree(path, ignore_errors=False):
    """shutil.rmtree that also removes read-only files (git's objects on Windows)."""
    import stat

    def onexc(fn, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            fn(p)
        except OSError:
            if not ignore_errors:
                raise
    if not Path(path).exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=onexc)
    else:
        shutil.rmtree(path, onerror=lambda fn, p, info: onexc(fn, p, info[1]))


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

DOC = (f"# The main setting on fresh seeds: pre-registration (version 9, near-80 main)\n\n**{K.STATUS_LINE_FINAL}**\n\n"
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
    counts = dict(kind="near80-v9-pooled-group-counts", seeds=[0, 1, 2, 3, 4], worlds=list(K.WORLDS),
                  summary={"stall": {"n_full_throttle_min": 40}}, results=[])
    fsha = write_json_sidecar(n80 / NP.FROZEN_REL, counts)
    if not frozen_passed:                                   # a frozen file that is not the pinned one
        fsha = "0" * 64
    draft = b'{"kind": "near80-preregistration-draft", "status": "draft", "version": "%s"}\n' % K.VERSION.encode()
    dsha = AC.sha256_bytes(draft)
    dname = K.DRAFT_HISTORY_NAME.format(prefix=dsha[:16])
    (n80 / K.HISTORY_DIR).mkdir(exist_ok=True)
    (n80 / K.HISTORY_DIR / dname).write_bytes(draft)
    (n80 / K.HISTORY_DIR / (dname + ".sha256")).write_text(f"{dsha}  {dname}\n", encoding="utf8")
    pins = AC.current_pins(repo)
    pins["repository_files_count"] = len(repo_files)
    prereg = dict(kind=K.PREREG_KIND, version=K.VERSION, label=K.LABEL, status=status,
                  status_line=K.STATUS_LINE_FINAL, approvable=True, missing=[], owner_answer=OWNER_ANSWER,
                  constants=AC.canonical_constants(), claims_word_for_word=None,
                  code=dict(code_manifest=K.MANIFEST_NAME, code_manifest_sha256=msha, files=files), pins=pins,
                  pooled_group_counts_frozen=dict(path=NP.FROZEN_REL, sha256=fsha, seeds=[0, 1, 2, 3, 4],
                                                  worlds=list(K.WORLDS), summary=counts["summary"],
                                                  code_manifest_sha256=msha),
                  draft=dict(path=f"{K.HISTORY_DIR}/{dname}", sha256=dsha), approval=AC.approval_section(),
                  choices=[dict(number=1, choice="the setting", default="the main setting only",
                                taken="the main setting only", accepted=OWNER_ANSWER["utc"])],
                  seeds=dict(fresh=list(K.FRESH), fresh_count=len(K.FRESH), plumbing=list(K.SMOKE_SEEDS)),
                  setting={"main": {"role": "the only setting"}},
                  compute=dict(per_fresh_seed="about 2.2", plumbing="about 2.5", budget_cap=K.CAP_CPU_HOURS),
                  hosts=dict(desktop=dict(entry="run_near80.sh")),
                  summary=dict(claims=["C1", "C2"], rules=["the refusal rule", "the noise floor"]))
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
        rmtree(self.tmp, ignore_errors=True)


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
                      f"\"Near-80: approve version 9 {self.prereg.prefix}\"", text)
        self.assertIn(f"git push origin {K.BRANCH}", text)
        self.assertIn(K.IMPORT_NAME.format(prefix=self.prereg.prefix), text)
        self.assertIn("accept the defaults", text)
        self.assertIn("n_full_throttle_min", text)
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
        changed = p.read_bytes().replace(b'"version 9,', b'"version  9,')
        self.assertNotEqual(changed, p.read_bytes())
        p.write_bytes(changed)
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

    def test_refuses_a_missing_frozen_counts_file(self):
        (self.n80 / NP.FROZEN_REL).unlink()
        self.refused(f"{NP.FROZEN_REL} is missing")

    def test_refuses_a_frozen_counts_file_that_is_not_the_pinned_one(self):
        rmtree(self.repo)
        self.prereg = build_n80(self.repo, frozen_passed=False)
        self.refused("is not the one prereg.json pins")

    def test_version8_history_is_allowed_and_a_stranger_is_not(self):
        """Version 8's approval chain, amendment and frozen file stay in the folder; the script allows exactly them."""
        for name in K.V8_HISTORY_FILES:
            (self.n80 / name).write_text("{}")
            (self.n80 / (name + ".sha256")).write_text("0" * 64 + f"  {name}\n")
        (self.n80 / K.PRE_APPROVAL_DIR / K.V8_FROZEN_NAME).write_text("{}")
        (self.n80 / K.V9_NOTES).write_text("# notes\n")
        for d in K.V8_HISTORY_DIRS:
            (self.n80 / d / "files").mkdir(parents=True)
            (self.n80 / d / "RESULTS.json").write_text("{}")
        code, text = self.run_al(args=["--check"], stdin=NotTty())
        self.assertEqual(code, 0, text)
        (self.n80 / "approval_near80_amendment1_ffffffffffffffff.json").write_text("{}")
        self.refused("is not a file of this pre-registration", args=["--check"], stdin=NotTty())

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
            rmtree(top) if top.is_dir() else top.unlink()
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
        self.assertEqual(AL.REPO, K.DESKTOP_ROOT)                        # the owner's own machine
        if K.HOST == "desktop":
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
        self.assertEqual(AL.VERSION, K.VERSION)
        self.assertEqual(AL.V8_HISTORY_FILES, K.V8_HISTORY_FILES)
        self.assertEqual(AL.V8_HISTORY_DIRS, K.V8_HISTORY_DIRS)
        self.assertEqual(AL.V8_FROZEN, K.PRE_APPROVAL_DIR + "/" + K.V8_FROZEN_NAME)
        self.assertEqual(AL.V9_NOTES, K.V9_NOTES)
        self.assertEqual(AL.CHECK_TESTS, K.TEST_FILES)

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
        done = subprocess.run([system, "-I", str(K.N80 / "approve_local_near80.py"), "--check"], cwd=WORK_DIR,
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
        c = self.commit(committer, f"Near-80: approve version 9 {self.prereg.prefix}", *rels, author=author)
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
                rmtree(self.repo)
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
            ("an amendment_2.json", another_amendment_file, f"only {AC.AMENDMENT_NAME} is known to this code"),
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
        self.assertIn(f"Amendment (version 9): refused: {AC.AMENDMENT_NAME} does not amend", "\n".join(lines))

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
                 (dict(amendment_sha256="f" * 64), f"it checked another version of {AC.AMENDMENT_NAME}"),
                 (dict(check_sha256=dict(TEST_CHECK_SHA, py="3" * 64)), "made by another check"),
                 (dict(check_sha256=None), "made by another check"),
                 (dict(unit_tests=None), "it did not run and pass the five unit test files"),
                 (dict(unit_tests={}), "it did not run and pass the five unit test files"),
                 (dict(unit_tests={n: 0 for n in K.TEST_FILES[:3]}), "the five unit test files"),
                 (dict(unit_tests=dict({n: 0 for n in K.TEST_FILES}, **{K.TEST_FILES[-1]: 1})),
                  "the five unit test files"),
                 (dict(unit_tests={n: False for n in K.TEST_FILES}), "the five unit test files")]
        for fields, why in cases:
            with self.subTest(fields=sorted(fields)):
                self.assertIn(why, "; ".join(AC.desktop_check_problems(result(**fields), amendment)))
        self.assertEqual(AC.desktop_check_problems(b"\xff", amendment), ["it is not UTF-8 JSON"])
        no_code = AC.Amendment(amendment.path, asha, {k: v for k, v in amendment.payload.items()
                                                      if k != "desktop_check"})
        self.assertIn("made by another check", "; ".join(AC.desktop_check_problems(result(), no_code)))

    def test_deeply_nested_json_is_a_refusal(self):
        """JSON nested deeper than the parser allows raises RecursionError, which is not a ValueError: the gate's
        readers refuse it (a desktop check's result, the owner's approval of the amendment, amendment_1.json) instead of
        crashing."""
        fx, cfg = self.fixture()
        asha = write_test_amendment(fx, self.good_edits(fx), approve=False)
        amendment = AC.Amendment(fx / AC.AMENDMENT_NAME, asha, AC.read_json_verified(fx / AC.AMENDMENT_NAME))
        deep = b"[" * 200000 + b"]" * 200000
        self.assertEqual(AC.desktop_check_problems(deep, amendment), ["it is not UTF-8 JSON"])
        with self.assertRaisesRegex(AC.ApprovalRefused, "is not UTF-8 JSON"):
            AC.check_local_amendment_approval(deep, AC.load_prereg(cfg), amendment)
        p = fx / AC.AMENDMENT_NAME
        p.write_bytes(deep)
        Path(str(p) + ".sha256").write_text(f"{AC.sha256_bytes(deep)}  {p.name}\n", encoding="utf8")
        with self.assertRaisesRegex(AC.ApprovalRefused, "is not JSON"):
            AC.read_json_verified(p)
        with self.assertRaisesRegex(AC.ApprovalRefused, "is not JSON"):
            AC.require_approval(cfg)


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
        self.assertIn(f"Near-80: approve version 9 amendment 1 {self.a16}", out.getvalue())
        if not commit:
            return None
        rels = [f"{K.N80_REL}/{self.a_name}", f"{K.N80_REL}/{self.a_name}.sha256", check, check + ".sha256"]
        c = self.commit(committer, f"Near-80: approve version 9 amendment 1 {self.a16}", *rels, author=author)
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
        self.commit(APPROVER, f"Near-80: approve version 9 amendment 1 {self.a16}", *rels)
        self.push()
        self.amendment_import_refused(f"approves another version of {AC.AMENDMENT_NAME}")

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
        self.commit(APPROVER, f"Near-80: approve version 9 amendment 1 {self.a16}", *rels)
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
        self.commit(APPROVER, f"Near-80: approve version 9 amendment 1 {self.a16}", *rels)
        self.push()
        self.amendment_import_refused(f"made by another check than the one {AC.AMENDMENT_NAME} names")

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
                      f"\"Near-80: approve version 9 amendment 1 {self.a16}\"", text)
        self.assertIn(AC.AMEND_IMPORT_NAME.format(prefix=self.a16), text)
        self.refused("already exists")                                     # never overwrites

    def test_check_writes_nothing_and_shows_each_change(self):
        before = tree(self.n80)
        code, text = self.run_al(args=["--check"], stdin=NotTty())
        self.assertEqual(code, 0, text)
        flat = " ".join(text.split())
        self.assertIn("Nothing was asked and nothing was written", flat)
        self.assertIn("Near-80 (version 9): amendment 1", flat)
        for rel, (old, new) in self.body["delta"]["changed"].items():
            self.assertIn(f"{rel}: {old[:16]} -> {new[:16]}", flat)
        self.assertIn(f"appended to; the first {self.n} bytes are the approved text", flat)
        self.assertIn(self.asha, flat)
        self.assertIn("PREREGISTRATION.md, its amendment section", flat)
        self.assertIn("What does not change: - prereg.json", flat)
        self.assertIn("Kept, and what runs after your approval is imported: - the failed phases", flat)
        self.assertIn("Why it cannot change a result: the hash chain", flat)
        self.assertIn(f"passed: {Path(self.check_rel).name}", flat)
        self.assertIn(f"its code, as the amendment names it: check_amendment1.py {'1' * 16}, check_amendment1.sh "
                      f"{'2' * 16}", flat)
        self.assertLess(flat.index("What changes"), flat.index(f"Fingerprint (SHA-256) of {AC.AMENDMENT_NAME}:"))
        self.assertEqual(tree(self.n80), before)

    def test_refuses_without_a_passing_desktop_check(self):
        check = self.n80.parent / self.check_rel
        variants = [("none", None), ("failed", dict(passed=False)), ("a stub run", dict(mode="stub")),
                    ("another amendment", dict(amendment_sha256="f" * 64)), ("another kind", dict(kind="x")),
                    ("a changed result file", "edit"),
                    ("no unit tests (--no-tests)", dict(unit_tests={}, tests_run=False)),
                    ("no unit_tests field", dict(unit_tests=None)),
                    ("three of the five unit test files", dict(unit_tests={n: 0 for n in K.TEST_FILES[:3]})),
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

    def test_refuses_deeply_nested_json(self):
        """A deeply nested desktop check result, or amendment_1.json, whose .sha256 matches: a refusal, not a
        traceback (the JSON parser raises RecursionError, which is not a ValueError)."""
        deep = b"[" * 200000 + b"]" * 200000
        check = self.n80.parent / self.check_rel
        check.write_bytes(deep)
        Path(str(check) + ".sha256").write_text(f"{AC.sha256_bytes(deep)}  {check.name}\n", encoding="utf8")
        self.refused("No desktop check of this amendment has passed")
        code, text = self.run_al(args=["--check"], stdin=NotTty())
        self.assertEqual(code, 0, text)
        self.assertIn("none has passed yet", " ".join(text.split()))
        p = self.n80 / AC.AMENDMENT_NAME
        p.write_bytes(deep)
        Path(str(p) + ".sha256").write_text(f"{AC.sha256_bytes(deep)}  {p.name}\n", encoding="utf8")
        self.refused(f"{AC.AMENDMENT_NAME} is not readable JSON", args=["--check"], stdin=NotTty())

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
                self.refused(f"{AC.AMENDMENT_NAME} is malformed", args=["--check"], stdin=NotTty())

    def test_refuses_an_amendment_of_another_study(self):
        self.body["label"] = "another"
        self.write_amendment()
        self.refused(f"{AC.AMENDMENT_NAME} names another study", args=["--check"], stdin=NotTty())

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
                self.refused("adds and removes no file", args=["--check"], stdin=NotTty())

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
        self.refused(f"{self.p_import} is not the one {AC.AMENDMENT_NAME} names", args=["--check"], stdin=NotTty())

    def test_refuses_a_changed_amendment_sidecar(self):
        (self.n80 / (AC.AMENDMENT_NAME + ".sha256")).write_text(f"{'0' * 64}  {AC.AMENDMENT_NAME}\n", encoding="utf8")
        self.refused(f"{AC.AMENDMENT_NAME} does not match its fingerprint file", args=["--check"], stdin=NotTty())

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


TODO_RE = __import__("re").compile(r"TODO\(DECISION\.md[^)]*\)")


def finalized(text, name):
    """A document or draft as DECISION.md and the freeze-time edit would leave it: the final status line, every
    TODO(DECISION.md) mark replaced by a value, no 'this draft'."""
    text = text.replace(f"**{K.STATUS_LINE_DRAFT}**", f"**{K.STATUS_LINE_FINAL}**", 1)
    text = TODO_RE.sub("5", text)
    text = text.replace("This draft", "This version").replace("this draft", "this version")
    return text


def final_code(src):
    """A copied code file with the DRAFT mark of its first docstring line removed (the freeze-time edit of the
    documents' and the code's marks, done on the copy only)."""
    lines = src.split("\n")
    head = __import__("re").sub(r"DRAFT,\s+not\s+frozen", "frozen (a test copy)", "\n".join(lines[:3]))
    return "\n".join(head.split("\n") + lines[3:])


def make_n80_copy(dst, claims_real=False, final=True, previous=False):
    """A temporary copy of the near-80 folder for the freeze: the real documents and draft (finalized, unless
    final=False), A's code and the top-level scripts; the other builders' files as stubs (claims_text.py real when
    asked). previous=True also puts a synthetic version 8 prereg.json at the top (the freeze supersedes it)."""
    dst = Path(dst)
    (dst / "code").mkdir(parents=True)
    for name in ("PREREGISTRATION.md", "README.md"):
        text = (K.N80 / name).read_text(encoding="utf8")
        (dst / name).write_text(finalized(text, name) if final else text, encoding="utf8")
    shutil.copyfile(K.N80 / "approve_local_near80.py", dst / "approve_local_near80.py")
    for name in K.TOP_FILES:
        if not (dst / name).exists():
            (dst / name).write_text("#!/bin/sh\n# a stub for a freeze test\n", encoding="utf8")
    draft = real_draft_bytes().decode("utf8")
    (dst / "prereg_draft.json").write_text(finalized(draft, "draft") if final else draft, encoding="utf8")
    for name in K.CODE_FILES:
        if name in A_FILES or (claims_real and name == "claims_text.py"):
            src = (K.CODE / name).read_text(encoding="utf8")
            (dst / "code" / name).write_text(final_code(src) if final else src, encoding="utf8")
        else:
            (dst / "code" / name).write_text(f'"""A stub of {name} for a freeze test."""\n', encoding="utf8")
    if previous:
        body = dict(kind=K.PREREG_KIND, version=K.PREVIOUS_VERSION, label="version 8 (synthetic)", status="final",
                    synthetic=True)
        write_json_sidecar(dst / K.PREREG_NAME, body)
        write_json_sidecar(dst / K.MANIFEST_NAME, dict(kind=K.MANIFEST_KIND, label="version 8 (synthetic)",
                                                       files={"a": "b"}, repository_files={}, history=[],
                                                       reason="version 8 (synthetic)"))
    return dst


def fake_tests(fail=()):
    def fn(n80_root, log):
        fn.calls += 1
        return {n: (1 if n in fail else 0) for n in K.TEST_FILES}
    fn.calls = 0
    return fn


def fake_facts(passed=True, seeds=None, worlds=None):
    def fn(n80_root, out_path, log):
        seeds_ = list(seeds if seeds is not None else [0, 1, 2, 3, 4])
        worlds_ = list(worlds if worlds is not None else K.WORLDS)
        results = [dict(rep=s, world=w, defined=passed, groups=dict(n_full_throttle=40 + s, n_rest=4000) if passed
                        else None, certification=dict(reaches_min_rows=True) if passed else None,
                        version8_groups=dict(n_full_throttle=30 + s, n_rest=800) if passed else None)
                   for w in worlds_ for s in seeds_]
        payload = dict(kind="near80-v9-pooled-group-counts", status=K.STATUS_LINE_FINAL, label=K.LABEL, seeds=seeds_,
                       worlds=worlds_, results=results, summary={w: dict(seed_worlds=len(seeds_)) for w in worlds_},
                       min_rows=K.CERT_MIN_ROWS, seconds=0.0)
        Path(out_path).write_text(json.dumps(payload, indent=1), encoding="utf8")
        fn.calls.append(Path(out_path))
    fn.calls = []
    return fn


def fake_claims(text):
    return {"source": "PREREGISTRATION.md section 14", "synthetic": True, "length": len(text), "todo_marks": []}


class FreezeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="near80_test_freeze_"))
        self.n80 = make_n80_copy(self.tmp / "near80")
        self.out = self.tmp / "out"
        self.stack = enter_fixed(contextlib.ExitStack())

    def tearDown(self):
        self.stack.close()
        rmtree(self.tmp, ignore_errors=True)

    def write(self, **kw):
        args = dict(n80_root=self.n80, out_root=self.out, repo=K.ROOT, tests_fn=fake_tests(), facts_fn=fake_facts(),
                    claims_fn=fake_claims, log=quiet)
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
        draft_sha = AC.file_sha256(self.n80 / "prereg_draft.json")
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
        self.assertEqual((man["kind"], man["version"], man["history"]), (K.MANIFEST_KIND, K.VERSION, []))
        self.assertTrue(set(K.PIN_FILES) <= set(man["repository_files"]))
        self.assertFalse(any("__pycache__" in r for r in man["repository_files"]))
        self.assertTrue(all(any(r.startswith(d + "/") for d in K.PIN_DIRS) or r in K.PIN_FILES
                            for r in man["repository_files"]))
        self.assertTrue(all(Path(r).suffix in K.PIN_SUFFIXES for r in man["repository_files"]))
        self.assertIn("outputs/shift/beyond-the-data/stage2/code/stage2_common.py", man["repository_files"])
        # step 7: the counts ran once, under this manifest's name, and were copied
        msha = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
        self.assertEqual(facts.calls, [self.out / "pre" / FZ.FACTS_NAME.format(prefix=msha[:16])])
        frozen = self.n80 / NP.FROZEN_REL
        self.assertEqual(frozen.read_bytes(), facts.calls[0].read_bytes())
        self.assertEqual(AC.sidecar_digest(frozen), AC.file_sha256(frozen))
        # step 8: prereg.json
        on_disk = AC.read_json_verified(self.n80 / K.PREREG_NAME)
        self.assertEqual(on_disk, body)
        for key in ("kind", "version", "label", "status", "status_line", "approvable", "missing", "owner_answer",
                    "supersedes", "draft", "choices", "constants", "claims_word_for_word", "code", "pins",
                    "pooled_group_counts_frozen", "approval", "hosts", "runner", "summary", "version8_history",
                    "files") + NP.CONTENT_KEYS:
            self.assertIn(key, body)
        self.assertEqual((body["kind"], body["version"], body["supersedes"]),
                         (K.PREREG_KIND, K.VERSION, K.PREVIOUS_VERSION))
        self.assertEqual(body["status"], K.STATUS_FINAL)
        self.assertEqual(body["status_line"], K.STATUS_LINE_FINAL)
        self.assertEqual((body["approvable"], body["missing"]), (True, []))
        self.assertEqual(body["owner_answer"], OWNER_ANSWER)
        self.assertEqual(body["draft"]["path"], f"{K.HISTORY_DIR}/{kept.name}")
        self.assertEqual(body["draft"]["sha256"], draft_sha)
        self.assertEqual(body["constants"], AC.canonical_constants())
        self.assertEqual(body["constants"]["MARKS_C2"], MARKS_C2_FIXED)
        self.assertEqual(body["constants"]["PLACEHOLDERS"], {})
        self.assertEqual(body["claims_word_for_word"],
                         fake_claims((self.n80 / "PREREGISTRATION.md").read_text(encoding="utf8")))
        self.assertEqual(body["code"], {"code_manifest": K.MANIFEST_NAME, "code_manifest_sha256": msha,
                                        "files": man["files"]})
        self.assertEqual(set(body["pins"]), set(AC.PINS_REL) | {"repository_files_count"})
        self.assertEqual(body["pins"]["v6_stage2_common_sha256"], AC.V6_STAGE2_COMMON_SHA256)
        self.assertEqual(body["pins"]["repository_files_count"], len(man["repository_files"]))
        self.assertEqual(body["pooled_group_counts_frozen"],
                         {"path": NP.FROZEN_REL, "sha256": AC.file_sha256(frozen), "seeds": [0, 1, 2, 3, 4],
                          "worlds": list(K.WORLDS), "summary": {w: {"seed_worlds": 5} for w in K.WORLDS},
                          "code_manifest_sha256": msha, "role": "reported; outcome-free; no mark reads it"})
        self.assertEqual(body["approval"]["approver_email"], K.APPROVER_EMAIL)
        self.assertEqual(body["approval"]["branch"], K.BRANCH)
        self.assertIs(body["approval"]["local_terminal_allowed"], True)
        self.assertIs(body["approval"]["accepts_defaults"], True)
        self.assertIn("cluster", body["approval"]["hosts"])
        self.assertEqual(set(body["hosts"]), {"desktop", "cluster"})
        self.assertEqual(body["runner"]["phases"], list(K.PHASE_KINDS))
        self.assertEqual(body["runner"]["lanes_default"], K.DEFAULT_LANES)
        self.assertEqual((body["runner"]["desktop_entry"], body["runner"]["cluster_entry"]),
                         ("run_near80.sh", "run_near80_cluster.sh"))
        draft = json.loads(kept.read_text(encoding="utf8"))
        self.assertEqual(len(body["choices"]), len(draft["choices"]))
        for c, d in zip(body["choices"], draft["choices"]):
            self.assertEqual(c["taken"], d["default"])
            self.assertEqual(c["accepted"], OWNER_ANSWER["utc"])
        # the marks DECISION.md fixed, the seeds, the arm names of the code, the summary, and no draft wording
        for k, v in MARKS_C2_FIXED.items():
            self.assertEqual(body["pass_marks"]["c2_twins"][k], v)
        self.assertEqual(body["pass_marks"]["c2_twins"]["twins"], list(K.TWIN_ARMS))
        self.assertEqual((body["seeds"]["fresh"], body["seeds"]["fresh_count"], body["seeds"]["plumbing"]),
                         (list(K.FRESH), len(K.FRESH), list(K.SMOKE_SEEDS)))
        self.assertEqual(body["arms"]["logical_ids"], {k: list(v) for k, v in K.ARMS.items()})
        self.assertEqual(len(body["summary"]["claims"]), 2)
        self.assertTrue(body["summary"]["claims"][0].startswith("C1"))
        self.assertIn(f"margin {MARKS_C2_FIXED['net_margin']}", body["summary"]["claims"][1])
        self.assertTrue(body["summary"]["rules"][0].startswith("The refusal rule:"))
        self.assertEqual(body["version8_history"]["files"], list(K.V8_HISTORY_FILES))
        check = {k: v for k, v in body.items() if k not in ("draft", "claims_word_for_word", "files",
                                                             "version8_history")}
        check["constants"] = {k: v for k, v in check["constants"].items() if k not in NP.DRAFT_STATE_CONSTANTS}
        self.assertEqual(NP.draft_words(check), [])
        self.assertEqual(NP.todo_marks({k: v for k, v in body.items() if k != "constants"}), [])
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
            "title": lambda t: t.replace(FZ.TITLE, "# A title", 1),
            "this draft": lambda t: t + "\nSee this draft.\n",
            "14.5": lambda t: t.replace("### 14.5 How the code fills in the sentences", "### How", 1),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                rmtree(self.n80)
                self.n80 = make_n80_copy(self.tmp / f"near80_{name.replace(' ', '_')}")
                p = self.n80 / "PREREGISTRATION.md"
                p.write_text(change(p.read_text()))
                self.refused()
        rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_readme")
        p = self.n80 / "README.md"
        p.write_text(p.read_text().replace(f"**{K.STATUS_LINE_FINAL}**", "**DRAFT.**"))
        self.refused("README.md")
        rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_code")
        (self.n80 / "code" / "pooled.py").write_text('"""DRAFT looks."""\n')
        self.refused("DRAFT in its first docstring line")

    def test_refuses_an_existing_prereg_or_approval_or_output(self):
        for name, make in (("prereg", lambda: (self.n80 / K.PREREG_NAME).write_text("{}")),
                           ("approval", lambda: (self.n80 / "approval_near80_0123456789abcdef.json").write_text("{}")),
                           ("import", lambda: (self.n80 / "import_near80_0123456789abcdef.json").write_text("{}")),
                           ("output", lambda: (self.out / "data").mkdir(parents=True))):
            with self.subTest(case=name):
                rmtree(self.n80)
                rmtree(self.out, ignore_errors=True)
                self.n80 = make_n80_copy(self.tmp / f"near80_{name}")
                make()
                with self.assertRaises(FZ.FreezeRefused):
                    self.write()
                self.assertTrue((self.n80 / "prereg_draft.json").exists())
        rmtree(self.out, ignore_errors=True)
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

    def test_refuses_counts_that_are_not_usable(self):
        """The counts are reported, not a pass mark; but a record whose gap rule is undefined on a seed-world, or that
        ran on other seeds or one world, cannot be frozen."""
        msg = self.refused("not usable", facts_fn=fake_facts(passed=False))
        self.assertIn("the gap rule is not defined or the groups are missing", msg)
        self.assertFalse((self.n80 / NP.FROZEN_REL).exists())
        for why, facts in (("not 0-4", fake_facts(seeds=[0, 1])), ("not both", fake_facts(worlds=["stall"]))):
            msha = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
            FZ.facts_path(self.out, msha).unlink()                          # else the kept record is reused
            self.refused(why, facts_fn=facts)
        self.assertFalse((self.n80 / K.PREREG_NAME).exists())

    def test_refuses_while_a_placeholder_of_decision_md_remains(self):
        """Step 0: the constants' placeholders (the seed count, C2's marks, the owner's answer), a TODO(DECISION.md)
        mark in a document or the draft, or an unknown multiplicity rule; nothing is moved or written."""
        self.stack.close()                                                  # the constants as they are
        self.stack = contextlib.ExitStack()
        before = tree(self.n80)
        msg = self.refused("placeholder not fixed (DECISION.md): seed_count")
        self.assertIn("c2_net_margin", msg)
        self.assertIn("owner_answer", msg)
        self.assertEqual(tree(self.n80), before)
        self.stack = enter_fixed(contextlib.ExitStack())                    # the constants fixed, the documents not
        rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_marks", final=False)
        before = tree(self.n80)
        msg = self.refused("TODO(DECISION.md) mark")
        self.assertIn("PREREGISTRATION.md carries", msg)
        self.assertIn("prereg_draft.json carries", msg)
        self.assertIn("still says **DRAFT", msg)
        self.assertEqual(tree(self.n80), before)
        rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_rule")
        with mock.patch.object(K, "MARKS_C2", dict(MARKS_C2_FIXED, multiplicity="each_tenth")):
            self.refused("c2_multiplicity (not one of the known rules)")
        with mock.patch.object(K, "FRESH", (40, 41, 60)):
            self.refused("seeds (outside the block)")
        lines = []
        FZ.check(self.n80, self.out, log=lines.append)
        self.assertIn("0. refuse while any DECISION.md placeholder remains (0 now)", lines[0])

    def test_supersedes_a_previous_versions_prereg(self):
        """Step 1: version 8's prereg.json (approved and run) is moved into prereg_history/ as
        prereg_superseded_<first 16>.json with its sidecar, its manifest is archived, and the new prereg.json names
        what it supersedes; a prereg.json of this version is never moved."""
        rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_prev", previous=True)
        old = self.n80 / K.PREREG_NAME
        old_sha = AC.file_sha256(old)
        man_old = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
        lines = []
        FZ.check(self.n80, self.out, log=lines.append)
        self.assertIn(f"1. move {K.PREREG_NAME} of {K.PREVIOUS_VERSION} into {K.HISTORY_DIR}/", lines[0])
        body = self.write()
        kept = self.n80 / K.HISTORY_DIR / K.SUPERSEDED_NAME.format(prefix=old_sha[:16])
        self.assertEqual(AC.file_sha256(kept), old_sha)
        self.assertEqual(AC.sidecar_digest(kept), old_sha)
        self.assertEqual(json.loads(kept.read_text(encoding="utf8"))["version"], K.PREVIOUS_VERSION)
        self.assertEqual(AC.read_json_verified(old)["version"], K.VERSION)
        self.assertEqual(body["supersedes"], K.PREVIOUS_VERSION)
        man = AC.read_json_verified(self.n80 / K.MANIFEST_NAME)
        self.assertEqual(man["history"][0]["sha256"], man_old)
        self.assertEqual(AC.file_sha256(self.n80 / K.MANIFEST_HISTORY_DIR / "code_manifest_v1.json"), man_old)
        with self.assertRaises(FZ.FreezeRefused) as cm:
            self.write()
        self.assertIn("exists already", str(cm.exception))
        # a previous prereg.json whose sidecar does not match is not moved
        rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_prev2", previous=True)
        (self.n80 / (K.PREREG_NAME + ".sha256")).write_text("0" * 64 + f"  {K.PREREG_NAME}\n", encoding="utf8")
        before = tree(self.n80)
        with self.assertRaises(FZ.FreezeRefused) as cm:
            self.write()
        self.assertIn("differs from its sidecar", str(cm.exception))
        self.assertEqual(tree(self.n80), before)

    def test_facts_from_needs_this_manifests_name(self):
        wrong = self.tmp / "pooled_group_counts_0000000000000000.json"
        wrong.write_text("{}")
        self.refused("does not carry this manifest's first 16 characters", facts_from=wrong)

    def test_reuses_facts_already_written_under_this_manifest(self):
        self.refused("unit tests failed", tests_fn=fake_tests(fail=("test_near80.py",)))
        msha = AC.file_sha256(self.n80 / K.MANIFEST_NAME)
        path = self.out / "pre" / FZ.FACTS_NAME.format(prefix=msha[:16])
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
        rmtree(self.n80)
        self.n80 = make_n80_copy(self.tmp / "near80_claims", claims_real=True)
        body = self.write(claims_fn=None)
        mod = NP.load_claims_text(self.n80 / "code")
        doc = (self.n80 / "PREREGISTRATION.md").read_text(encoding="utf8")
        self.assertEqual(body["claims_word_for_word"], json.loads(AC.canonical(mod.extract_section14(doc))))
        self.assertEqual(body["claims_word_for_word"]["source"], "PREREGISTRATION.md section 14")


class TestFreezeOnTheRealFolder(unittest.TestCase):
    def test_the_real_folder_is_a_draft_the_freeze_refuses(self):
        """Version 9 is a draft awaiting the owner: the constants' placeholders are open and the documents say DRAFT,
        so freeze.py refuses at step 0 and step 2 and writes nothing; the title is already final and the code folder
        holds only planned files."""
        self.assertIn("seed_count", K.placeholders_present())
        found = FZ.placeholder_problems(K.N80)
        self.assertTrue(any(p.startswith("placeholder not fixed (DECISION.md): seed_count") for p in found), found)
        self.assertTrue(any(p.startswith("PREREGISTRATION.md carries") for p in found), found)
        docs = FZ.document_problems(K.N80)
        self.assertEqual([p for p in docs if "line 1" in p], [])
        self.assertTrue(any("line 3 is not" in p for p in docs), docs)
        self.assertTrue(any("still says **DRAFT" in p for p in docs), docs)
        code_problems = [p for p in FZ.code_folder_problems(K.N80) if not p.endswith(" is missing")]
        self.assertEqual(code_problems, [])
        before = sorted(p.name for p in K.N80.iterdir())
        lines = []
        self.assertTrue(FZ.check(K.N80, K.OUT, log=lines.append))
        self.assertIn("Problems (write would refuse)", lines[1])
        with self.assertRaises(FZ.FreezeRefused) as cm:
            FZ.write("a test", n80_root=K.N80, out_root=K.OUT, tests_fn=fake_tests(), facts_fn=fake_facts(),
                     claims_fn=fake_claims, log=quiet)
        self.assertIn("placeholder not fixed", str(cm.exception))
        self.assertEqual(sorted(p.name for p in K.N80.iterdir()), before)
        self.assertFalse(K.OUT.exists())


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
        self.assertEqual((N.FRESH_SEEDS, N.SMOKE_SEEDS, N.POOLED_SPLITS), (K.FRESH, K.SMOKE_SEEDS, K.POOLED_SPLITS))
        self.assertEqual(tuple(TC.SITUATIONS), K.SITUATIONS)
        draft = json.loads(real_draft_bytes())
        self.assertEqual((draft["kind"], draft["status"], draft["version"], draft["label"]),
                         (NP.DRAFT_KIND, "draft", K.VERSION, K.LABEL))
        self.assertEqual(draft["seeds"]["fresh"], list(K.FRESH))
        self.assertEqual(draft["seeds"]["fresh_block"], list(K.FRESH_SEED_BLOCK))
        self.assertEqual(draft["seeds"]["plumbing"], list(K.SMOKE_SEEDS))
        for block in draft["seeds"]["used_by_other_studies"]:
            lo, hi = (int(x) for x in block.split("-"))
            self.assertTrue(set(range(lo, hi + 1)) <= K.USED_SEEDS, block)
        self.assertEqual(draft["situations"]["worlds"], list(K.WORLDS))
        self.assertEqual(draft["situations"]["scored"], K.SCORED)
        self.assertEqual(set(draft["situations"]["test_start_sets"]), set(K.SITUATIONS))
        self.assertEqual(draft["compute"]["lanes_default"], K.DEFAULT_LANES)
        self.assertEqual(draft["pooled_groups"]["splits"], list(K.POOLED_SPLITS))
        c1 = draft["pass_marks"]["c1_version8_claim4"]
        self.assertIn(f"+ {K.MARKS['net_lead_margin']:g}", c1["net_lead"])
        self.assertIn(f">= {K.MARKS['keeps_floor']:g}", c1["keeps_the_gain"])
        self.assertIn(f">= {K.MARKS['even_pull_margin']:g}", c1["beats_even_pull"])
        c2 = draft["pass_marks"]["c2_twins"]
        self.assertEqual(c2["twins"], list(K.TWIN_ARMS))
        for k, v in K.MARKS_C2.items():      # a placeholder in the code is a TODO mark in the draft, and the other way
            self.assertEqual(v is None, str(c2[k]).startswith("TODO(DECISION.md"), k)
        self.assertIn(f">= {K.MARKS['harm_min']:g}", draft["validity"]["claims"])
        self.assertIn(f">= {K.MARKS['past80_min'] * 100:g}%", draft["validity"]["claims"])
        self.assertIn(f"<= {K.MARKS['noise_floor_max']:g}", draft["kill_rules"]["noise_floor"]["rule"])
        arms = draft["arms"]["main_both_worlds"]
        self.assertEqual(set(arms["primary"]), {K.PRIMARY})
        self.assertEqual(set(arms["twins_of_c2"]), set(K.TWIN_ARMS))
        self.assertEqual(set(arms["noise_floor_copy"]), {K.NOISE_ARM})
        self.assertEqual(set(arms["world_blind_rivals"]), set(K.WORLD_BLIND))
        self.assertTrue(set(K.REPORTED_ONLY) <= set(arms["reported_only"]))
        choices = {c["number"]: c["default"] for c in draft["choices"]}
        self.assertEqual(sorted(choices), list(range(1, len(choices) + 1)))
        self.assertEqual(choices[8], K.DEFAULTS["name"])
        self.assertEqual(choices[8], K.VERSION)
        self.assertEqual(int(choices[7]), K.DEFAULTS["lanes"])
        self.assertEqual(K.DEFAULTS["lanes"], K.DEFAULT_LANES)
        self.assertIn(K.DEFAULTS["seeds"], choices[2])
        self.assertEqual(draft["files"]["unit_tests"], list(K.TEST_FILES))
        self.assertEqual(set(draft["files"]["entries"]), set(K.TOP_FILES) - {"PREREGISTRATION.md", "README.md",
                                                                               "approve_local_near80.py"})

    def test_values_equal_step_1_and_step_2(self):
        s1 = json.loads((K.S1_DIR / "prespec.json").read_text())
        s2 = json.loads((K.S2_DIR / "prespec.json").read_text())
        self.assertEqual({k: s1["certificate"][k] for k in K.CERT}, K.CERT)
        self.assertEqual({k: s1["groups"]["gap_rule"][k] for k in K.GAP_RULE}, K.GAP_RULE)
        self.assertIn(f"[{K.CERT_TAG_HEAD[0]}, {K.CERT_TAG_HEAD[1]}, seed, w", s1["certificate"]["rng_tags"])
        self.assertEqual({k: s2["arms"]["rule"][k] for k in K.ARM_RULE}, K.ARM_RULE)
        self.assertEqual(set(s2["arms"]["rule"]), set(K.ARM_RULE))
        self.assertEqual(tuple(s2["arms"]["definitions"]), K.STEP2_RAMP_ARMS)
        self.assertEqual(list(s2["seeds"]), [0, 1, 2, 3, 4])
        self.assertTrue(set(s2["seeds"]) <= K.USED_SEEDS)
        p = s2["protocol"]
        self.assertEqual((p["alpha"], p["actor_steps"], p["chunk"]), (K.ALPHA, K.ACTOR_STEPS, K.CHUNK))
        self.assertEqual(K.MEANDIFF_MARGIN, 0.10)
        self.assertEqual(K.MEANDIFF_MARGIN, K.ARM_RULE["conformal_margin"])
        v7 = load_path("near80_test_v7_common", K.V7_DIR / "code" / "v7_common.py")
        for k, v in v7.PHASE_ENV.items():
            if k not in ("PATH", "HOME", "PYTHONPATH"):
                self.assertEqual(K.PHASE_ENV.get(k), v, k)
        if K.HOST == "desktop":
            self.assertEqual(v7.PHASE_ENV, {k: v for k, v in K.PHASE_ENV.items() if k != "PYTHONPYCACHEPREFIX"})
            self.assertEqual(v7.PYTHON, K.PYTHON)
        self.assertEqual(v7.PYTHON, K.HOST_PROFILES["desktop"]["python"])
        self.assertEqual(K.PHASE_ENV["PYTHONPYCACHEPREFIX"], K.PYCACHE_PREFIX)
        self.assertFalse(Path(K.PYCACHE_PREFIX).exists())
        self.assertEqual(v7.APPROVER_EMAIL, K.APPROVER_EMAIL)
        self.assertEqual(v7.BRANCH, K.BRANCH)
        self.assertEqual(v7.W0_TOL, K.W0_TOL)

    def test_layout_and_helpers(self):
        self.assertEqual(K.path("/r", "values", setting="main", world="stall", seed=3),
                         Path("/r/eval/main/stall/seed3/values.json"))
        self.assertEqual(K.phase_id("policy", "main", "stall", seed=3), "policy.main.stall.s3")
        self.assertEqual(K.phase_id("lock", group="fresh"), "lock.x.x.fresh")
        self.assertEqual(K.group_of(0), "smoke")
        for s in K.FRESH:
            self.assertEqual(K.group_of(s), "fresh")
        for s in (1, 5, 10, 12, 20, 34, 60, 100, 900, -1):
            with self.assertRaises(ValueError):
                K.group_of(s)
        self.assertFalse(set(K.FRESH) & K.USED_SEEDS)
        self.assertTrue(all(K.FRESH_SEED_BLOCK[0] <= s <= K.FRESH_SEED_BLOCK[1] for s in K.FRESH))
        self.assertFalse(set(range(K.FRESH_SEED_BLOCK[0], K.FRESH_SEED_BLOCK[1] + 1)) & K.USED_SEEDS)
        self.assertTrue(set(K.SMOKE_SEEDS) <= K.USED_SEEDS)
        d = AC.canonical_constants()
        self.assertEqual(d["ARMS"]["main"], list(K.ARMS["main"]))
        self.assertEqual(d["FRESH"], list(K.FRESH))
        for name in K.HOST_DEPENDENT:
            self.assertNotIn(name, d)
        self.assertTrue(set(K.TEST_FILES) <= set(K.CODE_FILES))
        self.assertIn("test_v9.py", K.TEST_FILES)
        self.assertEqual(len(set(K.CODE_FILES)), len(K.CODE_FILES))
        self.assertTrue(set(K.STEP2_RAMP_ARMS) | set(K.NEW_ARMS) | set(K.TWIN_ARMS) | set(K.BCA_DEPENDENT)
                        <= set(K.ARMS["main"]))
        self.assertIn(K.NOISE_ARM, K.BCA_DEPENDENT)
        self.assertIn(K.PRIMARY, K.BCA_DEPENDENT)
        self.assertFalse(set(K.TWIN_ARMS) & set(K.BCA_DEPENDENT))
        self.assertEqual(set(K.ARM_PLAIN), set(K.ARMS["main"]))
        self.assertEqual(set(K.PHASE_PLAIN), set(K.PHASE_KINDS))
        self.assertEqual(K.SETTINGS, ("main",))
        self.assertEqual(K.APPROVAL_NAME.format(prefix="p"), "approval_near80_p.json")
        self.assertEqual(K.N80, K.ROOT / K.N80_REL)
        self.assertFalse(set(K.V8_HISTORY_FILES) & set(K.TOP_FILES))
        self.assertNotIn(K.V9_NOTES, K.TOP_FILES)
        self.assertEqual(K.CAP_CPU_HOURS, round(2.5 * K.FORECAST_CPU_HOURS_PER_SEED * len(K.FRESH), 1))
        self.assertEqual(K.FORECAST_CPU_HOURS["fresh"], K.FORECAST_CPU_HOURS_PER_SEED * len(K.FRESH))
        self.assertEqual(K.BUDGET_STAGES, ("fresh",))

    def test_placeholders_of_decision_md(self):
        left = K.placeholders_present()
        for name in ("seed_count", "c2_net_margin", "c2_positive_seeds_min", "c2_multiplicity", "owner_answer"):
            self.assertIn(name, left)
        self.assertTrue(all(v.startswith(K.TODO) for v in K.PLACEHOLDERS.values()))
        with enter_fixed(contextlib.ExitStack()):
            self.assertEqual(K.placeholders_present(), [])
            with mock.patch.object(K, "MARKS_C2", dict(MARKS_C2_FIXED, multiplicity="x")):
                self.assertEqual(K.placeholders_present(), ["c2_multiplicity (not one of the known rules)"])
            with mock.patch.object(K, "MARKS_C2", dict(MARKS_C2_FIXED, net_margin=None)):
                self.assertEqual(K.placeholders_present(), ["c2_net_margin"])

    def test_load_local_avoids_step_2s_modules(self):
        mod = K.load_local("near80_prereg")
        self.assertEqual(Path(mod.__file__), K.CODE / "near80_prereg.py")
        self.assertIs(K.load_local("near80_prereg"), mod)


if __name__ == "__main__":
    unittest.main(verbosity=2)
