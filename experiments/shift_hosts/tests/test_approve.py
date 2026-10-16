"""approve_hosts.py and the gate's positive path, on a throwaway git repository (never the real one).

The user's address never appears on a commit here: the positive path patches gate.APPROVAL_EMAIL to a test address, and
every test commit is authored by that test address. Run in WSL (either environment; pure Python and git):
    PYTHONDONTWRITEBYTECODE=1 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_approve
"""

import datetime
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.shift_hosts import approve_hosts, gate

TEST_EMAIL = "approval-test@example.invalid"
NOW = datetime.datetime(2026, 10, 15, 20, 0, tzinfo=datetime.timezone.utc)


def git(repo, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=Approval test", "-c", f"user.email={TEST_EMAIL}",
                           *args], check=True, capture_output=True, env=env).stdout


class Terminal(io.StringIO):
    """A stream that says it is a terminal; `on_read` runs when the question is answered."""

    def __init__(self, text="", on_read=None):
        super().__init__(text)
        self.on_read = on_read

    def isatty(self):
        return True

    def readline(self, *args):
        if self.on_read:
            self.on_read()
        return super().readline(*args)


class ApproveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.pkg = self.repo / "experiments" / "shift_hosts"
        self.pkg.mkdir(parents=True)
        (self.repo / "experiments" / "__init__.py").write_text("")
        (self.pkg / "module.py").write_text("x = 1\n")
        (self.pkg / "PRESPEC.md").write_text("# frozen\n")
        (self.pkg / "prespec.json").write_text(json.dumps({"title": "test", "approval_summary": ["first", "second"]}))
        (self.repo / "algorithms").mkdir()
        (self.repo / "algorithms" / "host.py").write_text("y = 2\n")  # matched by EXTERNAL_CODE's algorithms/*.py
        git(self.repo, "init", "-q", "-b", "claude/shift-hosts")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")

    def tearDown(self):
        self.tmp.cleanup()

    def files(self):
        return sorted(p.relative_to(self.repo).as_posix() for p in self.repo.rglob("*")
                      if p.is_file() and ".git" not in p.relative_to(self.repo).parts)

    def fp(self):
        return gate.fingerprint(self.pkg, self.repo)[0]

    def approve(self, typed=None):
        return approve_hosts.approve(self.fp()[:16] if typed is None else typed, self.pkg, self.repo, now=NOW)

    def test_manifest_covers_package_and_repository_code(self):
        manifest = gate.code_manifest(self.pkg, self.repo)
        self.assertEqual(sorted(manifest), ["algorithms/host.py", "experiments/__init__.py",
                                            "experiments/shift_hosts/PRESPEC.md", "experiments/shift_hosts/module.py"])

    def test_refuses_without_prespec(self):
        git(self.repo, "rm", "-q", "experiments/shift_hosts/prespec.json")
        git(self.repo, "commit", "-q", "-m", "no prespec")
        with self.assertRaisesRegex(approve_hosts.Refused, "no frozen pre-specification"):
            approve_hosts.prepare(self.pkg, self.repo)

    def test_refuses_uncommitted_code_prespec_or_new_file(self):
        before = self.files()
        for change in ("module", "prespec", "new", "external"):
            with self.subTest(change=change):
                git(self.repo, "reset", "-q", "--hard")
                git(self.repo, "clean", "-qfd")
                if change == "module":
                    (self.pkg / "module.py").write_text("x = 2\n")
                elif change == "prespec":
                    (self.pkg / "prespec.json").write_text("{}")
                elif change == "new":
                    (self.pkg / "extra.py").write_text("z = 3\n")
                else:
                    (self.repo / "algorithms" / "host.py").write_text("y = 3\n")
                with self.assertRaisesRegex(approve_hosts.Refused, "differ from git HEAD"):
                    self.approve()
                self.assertFalse(list(self.pkg.glob("approval_*")))
        git(self.repo, "reset", "-q", "--hard")
        git(self.repo, "clean", "-qfd")
        self.assertEqual(self.files(), before)

    def test_refuses_wrong_characters(self):
        for typed in ("0" * 16, self.fp()[:15], self.fp()[1:17], self.fp()):
            with self.subTest(typed=typed):
                with self.assertRaisesRegex(approve_hosts.Refused, "not the fingerprint's first 16"):
                    self.approve(typed)
        self.assertFalse(list(self.pkg.glob("approval_*")))

    def test_writes_the_gate_format_and_nothing_else(self):
        before = self.files()
        fp = self.fp()
        spaced = " ".join(fp[i:i + 4] for i in range(0, 16, 4)).upper()
        path, side = self.approve(spaced)
        self.assertEqual(path.name, f"approval_{fp[:16]}.json")
        self.assertEqual(side.name, f"approval_{fp[:16]}.json.sha256")
        self.assertEqual(self.files(), sorted(before + [path.relative_to(self.repo).as_posix(),
                                                        side.relative_to(self.repo).as_posix()]))
        body = json.loads(path.read_text(encoding="utf8"))
        _, prespec_sha, manifest = gate.fingerprint(self.pkg, self.repo)
        self.assertEqual(body["kind"], gate.APPROVAL_KIND)
        self.assertEqual(body["fingerprint"], fp)
        self.assertEqual(body["typed"], fp[:16])
        self.assertEqual(body["prespec_sha256"], prespec_sha)
        self.assertEqual(body["code_manifest"], manifest)
        self.assertEqual(body["head_commit"], git(self.repo, "rev-parse", "HEAD").decode().strip())
        self.assertEqual(body["approved_at_utc"], "2026-10-15T20:00:00Z")
        self.assertEqual(side.read_text(), f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
        subprocess.run(["sha256sum", "--check", "--quiet", side.name], cwd=self.pkg, check=True)

    def test_gate_passes_only_after_the_approvers_commit(self):
        path, side = self.approve()
        with self.assertRaisesRegex(gate.ApprovalError, "not committed in git HEAD"):
            gate.require_run_approval("test", package=self.pkg, repo=self.repo)
        git(self.repo, "add", path.relative_to(self.repo).as_posix(), side.relative_to(self.repo).as_posix())
        git(self.repo, "commit", "-q", "-m", "approve")
        with self.assertRaisesRegex(gate.ApprovalError, "not by"):  # authored by the test address, not the user's
            gate.require_run_approval("test", package=self.pkg, repo=self.repo)
        with mock.patch.object(gate, "APPROVAL_EMAIL", TEST_EMAIL):
            approval = gate.require_run_approval("test", package=self.pkg, repo=self.repo)
            self.assertEqual(approval.fingerprint, self.fp())
            self.assertEqual(approval.author_email, TEST_EMAIL)
            self.assertIs(gate.verify(approval, package=self.pkg, repo=self.repo), approval)
            (self.pkg / "module.py").write_text("x = 2\n")  # any code change: a new fingerprint, so refused
            with self.assertRaisesRegex(gate.ApprovalError, "no approval file"):
                gate.require_run_approval("test", package=self.pkg, repo=self.repo)
            with self.assertRaises(gate.ApprovalError):
                gate.verify(approval, package=self.pkg, repo=self.repo)

    def test_never_overwrites(self):
        path, side = self.approve()
        data = path.read_bytes(), side.read_bytes()
        with self.assertRaisesRegex(approve_hosts.Refused, "exists already"):
            self.approve()
        self.assertEqual((path.read_bytes(), side.read_bytes()), data)

    def test_main_refuses_without_a_terminal(self):
        out = io.StringIO()
        self.assertEqual(approve_hosts.main(io.StringIO(self.fp()[:16] + "\n"), out, self.pkg, self.repo), 1)
        self.assertIn("terminal", out.getvalue())
        self.assertFalse(list(self.pkg.glob("approval_*")))

    def test_main_at_a_terminal(self):
        fp = self.fp()
        out = Terminal()
        self.assertEqual(approve_hosts.main(Terminal(fp[:16] + "\n"), out, self.pkg, self.repo), 0)
        text = out.getvalue()
        for shown in (fp, "first", "second", "claude/shift-hosts", f"approval_{fp[:16]}.json",
                      f"user.email={gate.APPROVAL_EMAIL}", "GIT_CONFIG_GLOBAL=/root/.bca-gitconfig git push origin",
                      "Stop rule S0       NOT YET: no stage-0 summary"):
            self.assertIn(shown, text)
        self.assertTrue((self.pkg / f"approval_{fp[:16]}.json").is_file())
        out2 = Terminal()
        self.assertEqual(approve_hosts.main(Terminal(fp[:16] + "\n"), out2, self.pkg, self.repo), 0)
        self.assertIn("exists already", out2.getvalue())

    def test_describe_shows_stop_rule_s0(self):
        """The text before the question shows whether the stage-0 suite passed on exactly this code (gate.s0_status;
        the stage driver refuses to start without it)."""
        self.assertIn("NOT YET", approve_hosts.describe(approve_hosts.prepare(self.pkg, self.repo)))
        path = self.repo / "runs" / "shift_hosts" / "stage0" / "SUMMARY.json"
        path.parent.mkdir(parents=True)
        body = dict(schema=gate.S0_SCHEMA, status=0, s0=dict(passed=True), made_up_arrays_only=True, suite_run="test",
                    manifest=dict(code_manifest_sha256=gate.manifest_sha256(gate.code_manifest(self.pkg, self.repo))))
        data = json.dumps(body).encode("utf8")
        path.write_bytes(data)
        (path.parent / "SUMMARY.json.sha256").write_text(f"{hashlib.sha256(data).hexdigest()}  SUMMARY.json\n")
        text = approve_hosts.describe(approve_hosts.prepare(self.pkg, self.repo))
        self.assertIn("passed: the stage-0 suite test ran on exactly this code", text)
        (self.pkg / "module.py").write_text("x = 9\n")                 # other code: S0 no longer holds
        self.assertFalse(gate.s0_status(package=self.pkg, repo=self.repo)["passed"])

    def test_main_stops_on_empty_or_wrong_answer(self):
        for answer in ("\n", "abcd\n"):
            with self.subTest(answer=answer):
                out = Terminal()
                self.assertEqual(approve_hosts.main(Terminal(answer), out, self.pkg, self.repo), 1)
                self.assertIn("Not approved", out.getvalue())
        self.assertFalse(list(self.pkg.glob("approval_*")))

    def test_main_refuses_a_change_while_reading(self):
        fp = self.fp()

        def change():
            (self.pkg / "module.py").write_text("x = 5\n")
            git(self.repo, "commit", "-qam", "changed while reading")

        out = Terminal()
        self.assertEqual(approve_hosts.main(Terminal(fp[:16] + "\n", on_read=change), out, self.pkg, self.repo), 1)
        self.assertIn("changed while you were reading", out.getvalue())
        self.assertFalse(list(self.pkg.glob("approval_*")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
