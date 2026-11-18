#!/usr/bin/env python3
"""Fail when the instructions Claude Code loads at the start of every session reach 200 lines.

Why: every one of these lines is in context for every session, and Anthropic's guidance is to keep
CLAUDE.md under 200 lines because longer files reduce adherence
(https://code.claude.com/docs/en/memory).

Counted, because Claude Code loads them at launch:
  - ./CLAUDE.md and ./.claude/CLAUDE.md;
  - every repository file they import with @path, up to four hops (imported files load with the file
    that imports them, so moving lines into an import does not save context);
  - .claude/rules/**/*.md files without a `paths:` field in their frontmatter.
Not counted: block-level HTML comments (stripped before loading), rule frontmatter, path-scoped rules
and subdirectory CLAUDE.md files (those two load only when Claude works on matching files).

Usage:
  python3 .github/scripts/check_claude_md.py            # the working tree, as an agent sees it
  python3 .github/scripts/check_claude_md.py --rev HEAD # a commit (CI uses this on a sparse checkout)
"""

import argparse
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path

LIMIT = 200  # fail at this many lines: the guidance is "under 200"
MAX_HOPS = 4
ROOTS = ("CLAUDE.md", ".claude/CLAUDE.md")
RULES = ".claude/rules"

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(`+).+?\1")
# An import is "@" at the start of a line or after whitespace, then a path; "\ " escapes a space.
IMPORT = re.compile(r"(?:^|(?<=\s))@((?:\\ |\S)+)")
ROOT = Path(__file__).resolve().parents[2]


class WorkTree:
    """Files as they are on disk."""

    def __init__(self, root):
        self.root = Path(root)

    def read(self, path):
        file = self.root / path
        return file.read_text(encoding="utf8") if file.is_file() else None

    def files(self, folder):
        base = self.root / folder
        if not base.is_dir():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())


class GitTree:
    """Files as committed at a revision; works on sparse and blob-less checkouts."""

    def __init__(self, rev, root):
        self.rev, self.root = rev, Path(root)

    def _git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True)

    def read(self, path):
        result = self._git("cat-file", "blob", f"{self.rev}:{path}")
        return result.stdout.decode("utf8") if result.returncode == 0 else None

    def files(self, folder):
        result = self._git("ls-tree", "-r", "--name-only", self.rev, "--", folder)
        return sorted(result.stdout.decode("utf8").splitlines()) if result.returncode == 0 else []


def split_frontmatter(text):
    """Return (frontmatter lines or None, body text) for a rule file."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for end in range(1, len(lines)):
            if lines[end].strip() == "---":
                return lines[1:end], "\n".join(lines[end + 1 :])
    return None, text


def scan(text):
    """Return (number of lines that load into context, @import targets in load order)."""
    counted, imports = 0, []
    fence, in_comment = None, False
    for line in text.splitlines():
        if fence:
            counted += 1
            if re.match(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$", line):
                fence = None
            continue
        if in_comment:
            in_comment = "-->" not in line
            continue
        opening = FENCE.match(line)
        if opening:
            fence = opening.group(1)
            counted += 1
            continue
        if line.lstrip().startswith("<!--"):
            in_comment = "-->" not in line.split("<!--", 1)[1]
            continue
        counted += 1
        prose = CODE_SPAN.sub("", line)
        imports += [target.replace("\\ ", " ") for target in IMPORT.findall(prose)]
    return counted, imports


def resolve(importer, target):
    """Repository path an import points to, or None for home, absolute or outside paths."""
    if target.startswith(("~", "/")):
        return None
    path = posixpath.normpath(posixpath.join(posixpath.dirname(importer), target))
    return None if path == ".." or path.startswith("../") else path


def measure(tree):
    """Return (total lines, {path: lines} in load order, notes about imports that load nothing)."""
    loaded, notes = {}, []
    queue = [(root, 0, False) for root in ROOTS]
    for path in tree.files(RULES):
        if path.endswith(".md"):
            queue.append((path, 0, True))
    while queue:  # breadth first, so each file is reached by its shortest import chain
        path, hops, is_rule = queue.pop(0)
        if path in loaded:
            continue
        text = tree.read(path)
        if text is None:
            if hops:
                notes.append(f"@{path} is imported but does not exist, so nothing loads")
            continue
        if is_rule:
            frontmatter, text = split_frontmatter(text)
            if frontmatter and any(re.match(r"\s*paths\s*:", line) for line in frontmatter):
                continue  # path-scoped: loads only with matching files
        loaded[path], imports = scan(text)
        for target in imports:
            child = resolve(path, target)
            if child is None:
                notes.append(f"{path}: @{target} is outside the repository and is not counted")
            elif hops + 1 > MAX_HOPS:
                notes.append(f"{path}: @{target} is more than {MAX_HOPS} hops deep, so it does not load")
            else:
                queue.append((child, hops + 1, False))
    return sum(loaded.values()), loaded, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rev", help="read files from this git revision instead of the working tree")
    opt = parser.parse_args(argv)
    tree = GitTree(opt.rev, ROOT) if opt.rev else WorkTree(ROOT)
    total, loaded, notes = measure(tree)
    for path, lines in loaded.items():
        print(f"{lines:5d}  {path}")
    for note in notes:
        print(f"note: {note}")
    if total < LIMIT:
        print(f"OK: {total} lines load at the start of every session (limit: under {LIMIT}).")
        return 0
    message = (
        f"{total} lines load at the start of every session; keep them under {LIMIT}, because longer "
        "files reduce adherence. Make room with the strongest fix that works: turn a rule into a type, "
        "schema, lint rule, CI check or test, then delete the rule; or move guidance that applies to "
        "only some files into a path-scoped rule (.claude/rules/*.md with `paths:` frontmatter) or a "
        "CLAUDE.md in that folder."
    )
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error file=CLAUDE.md,title=CLAUDE.md is too long::{message}")
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
