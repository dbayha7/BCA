"""Tests for the CI checks. Run: python3 -m unittest discover -s .github/scripts"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_claude_md as claude_md  # noqa: E402
import check_frozen as frozen  # noqa: E402

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


class Repo:
    """A throwaway git repository."""

    def __init__(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.git("init", "-q", "-b", "main")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, env=GIT_ENV, check=True, capture_output=True, text=True).stdout

    def write(self, path, text):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf8")

    def commit(self, message="change"):
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD").strip()

    def close(self):
        self._dir.cleanup()


def lines(n, prefix="line"):
    return "".join(f"{prefix} {i}\n" for i in range(n))


class ClaudeMdTest(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.close)

    def measure(self):
        return claude_md.measure(claude_md.WorkTree(self.repo.root))

    def test_limit_is_under_200(self):
        self.repo.write("CLAUDE.md", lines(199))
        self.assertEqual(self.measure()[0], 199)
        self.repo.write("CLAUDE.md", lines(200))
        self.assertGreaterEqual(self.measure()[0], claude_md.LIMIT)

    def test_imports_count_relative_to_the_importing_file(self):
        self.repo.write("CLAUDE.md", "See @docs/guide.md\n")
        self.repo.write("docs/guide.md", "Also @more.md\n")
        self.repo.write("docs/more.md", lines(3))
        total, loaded, _ = self.measure()
        self.assertEqual(loaded, {"CLAUDE.md": 1, "docs/guide.md": 1, "docs/more.md": 3})
        self.assertEqual(total, 5)

    def test_code_spans_fences_quotes_and_emails_are_not_imports(self):
        self.repo.write(
            "CLAUDE.md",
            'Use `@/` and `@a.md`, "@a.md", mail me@a.md\n```\n@a.md\n```\n~~~~\n@a.md\n~~~~\n',
        )
        self.repo.write("a.md", lines(50))
        total, loaded, _ = self.measure()
        self.assertEqual(list(loaded), ["CLAUDE.md"])
        self.assertEqual(total, 7)

    def test_html_comments_are_stripped_but_not_inside_code(self):
        self.repo.write("CLAUDE.md", "kept\n<!-- one-line note -->\n<!--\nlong\nnote\n-->\n```\n<!-- code -->\n```\n")
        self.assertEqual(self.measure()[0], 4)

    def test_imports_stop_after_four_hops(self):
        self.repo.write("CLAUDE.md", "@f1.md\n")
        for hop in range(1, 6):
            self.repo.write(f"f{hop}.md", f"@f{hop + 1}.md\n")
        _, loaded, notes = self.measure()
        self.assertEqual(list(loaded), ["CLAUDE.md", "f1.md", "f2.md", "f3.md", "f4.md"])
        self.assertTrue(any("more than 4 hops" in note for note in notes))

    def test_import_cycles_count_each_file_once(self):
        self.repo.write("CLAUDE.md", "@a.md\n")
        self.repo.write("a.md", "@CLAUDE.md\n@a.md\n")
        self.assertEqual(self.measure()[0], 3)

    def test_missing_and_outside_imports_are_noted_not_counted(self):
        self.repo.write("CLAUDE.md", "@missing.md @~/notes.md @../up.md\n")
        total, _, notes = self.measure()
        self.assertEqual(total, 1)
        self.assertEqual(len(notes), 3)

    def test_unconditional_rules_count_and_path_scoped_rules_do_not(self):
        self.repo.write("CLAUDE.md", lines(1))
        self.repo.write(".claude/CLAUDE.md", lines(2))
        self.repo.write(".claude/rules/always.md", "---\ndescription: x\n---\n" + lines(4))
        self.repo.write(".claude/rules/web/scoped.md", "---\npaths:\n  - src/**\n---\n" + lines(90))
        self.repo.write(".claude/rules/notes.txt", lines(90))
        _, loaded, _ = self.measure()
        self.assertEqual(loaded, {"CLAUDE.md": 1, ".claude/CLAUDE.md": 2, ".claude/rules/always.md": 4})

    def test_git_tree_reads_committed_files_only(self):
        self.repo.write("CLAUDE.md", "@docs/a.md\n")
        self.repo.write("docs/a.md", lines(5))
        self.repo.commit()
        self.repo.write("docs/a.md", lines(500))  # uncommitted
        total, _, _ = claude_md.measure(claude_md.GitTree("HEAD", self.repo.root))
        self.assertEqual(total, 6)

    def test_main_fails_over_the_limit(self):
        self.repo.write("CLAUDE.md", lines(250))
        original = claude_md.ROOT
        claude_md.ROOT = self.repo.root
        try:
            with open(os.devnull, "w") as sink:
                stdout, stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = sink
                try:
                    self.assertEqual(claude_md.main([]), 1)
                finally:
                    sys.stdout, sys.stderr = stdout, stderr
        finally:
            claude_md.ROOT = original


class FrozenTest(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.close)
        self.repo.write(frozen.LIST, "# comment\npinned/\nsingle.txt\n")
        self.repo.write("pinned/receipt.json", "{}\n")
        self.repo.write("single.txt", "a\n")
        self.repo.write("free/work.py", "x = 1\n")
        self.base = self.repo.commit("base")

    def check(self, head=None):
        return frozen.violations(self.base, head, cwd=self.repo.root)

    def test_untouched_and_free_edits_pass(self):
        self.repo.write("free/work.py", "x = 2\n")
        self.assertEqual(self.check(self.repo.commit()), [])

    def test_adding_files_under_a_frozen_folder_passes(self):
        self.repo.write("pinned/new_result.json", "{}\n")
        self.assertEqual(self.check(self.repo.commit()), [])

    def test_editing_a_frozen_file_fails(self):
        self.repo.write("pinned/receipt.json", '{"changed": true}\n')
        self.assertEqual(self.check(self.repo.commit()), [("M", "pinned/receipt.json")])

    def test_deleting_or_moving_a_frozen_file_fails(self):
        self.repo.git("mv", "single.txt", "free/single.txt")
        self.repo.git("rm", "-q", "pinned/receipt.json")
        found = self.check(self.repo.commit())
        self.assertEqual(sorted(found), [("D", "pinned/receipt.json"), ("D", "single.txt")])

    def test_unfreezing_and_editing_in_one_change_fails(self):
        self.repo.write(frozen.LIST, "single.txt\n")
        self.repo.write("pinned/receipt.json", '{"changed": true}\n')
        self.assertEqual(self.check(self.repo.commit()), [("M", "pinned/receipt.json")])

    def test_unfreezing_first_then_editing_passes(self):
        self.repo.write(frozen.LIST, "single.txt\n")
        self.base = self.repo.commit("unfreeze, reviewed on its own")
        self.repo.write("pinned/receipt.json", '{"changed": true}\n')
        self.assertEqual(self.check(self.repo.commit()), [])

    def test_uncommitted_edits_in_the_working_tree_fail(self):
        self.repo.write("single.txt", "b\n")
        self.assertEqual(self.check(), [("M", "single.txt")])

    def test_a_prefix_is_not_a_folder_match(self):
        self.repo.write("pinned_other/file.txt", "a\n")
        self.repo.write("single.txt.bak", "a\n")
        self.base = self.repo.commit()
        self.repo.write("pinned_other/file.txt", "b\n")
        self.repo.write("single.txt.bak", "b\n")
        self.assertEqual(self.check(self.repo.commit()), [])


if __name__ == "__main__":
    unittest.main()
