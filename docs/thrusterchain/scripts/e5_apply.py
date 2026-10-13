"""Edit 1 of the tooltip pass: the tooltip component, the Tooltips switch and the table tooltips.

Exact, asserted replacements on thrusterchain.html (each old string matched exactly once):
  1. the component's CSS, before the only </style>;
  2. the switch and its note in the walkthrough bar at the top;
  3. the component's script (with the column descriptions of e5_cols.py) after the last </script>.
Never touches thrusterchain.v9.html. Run:
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/e5_apply.py'
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e5_cols import COLS  # noqa: E402

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v9.html"


def once(s, old, new, what):
    n = s.count(old)
    assert n == 1, f"{what}: expected exactly 1 match, found {n}"
    return s.replace(old, new, 1)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    backup_md5 = md5(BACKUP)
    s = open(PAGE, encoding="utf-8").read()
    assert 'id="tt-switch"' not in s and "thrusterchain-tooltips" not in s, "the tooltip edit is already applied"

    # 1. CSS before the only </style>
    css = open(os.path.join(HERE, "e5_tooltips.css"), encoding="utf-8").read()
    s = once(s, "</style>", css + "</style>", "style end")

    # 2. the switch beside the walkthrough button
    old_bar = (
        '    <div class="wt-bar" hidden>\n'
        '      <button type="button" class="btn wt-open" id="wt-open" aria-haspopup="dialog" aria-controls="wt">Full walkthrough mode</button>\n'
        '      <span class="note">Read the page one section at a time, in order, with Back and Next buttons. The normal page stays as it is.</span>\n'
        '    </div>\n'
    )
    new_bar = (
        '    <div class="wt-bar" hidden>\n'
        '      <button type="button" class="btn wt-open" id="wt-open" aria-haspopup="dialog" aria-controls="wt">Full walkthrough mode</button>\n'
        '      <button type="button" class="tt-switch" id="tt-switch" role="switch" aria-checked="true" aria-describedby="tt-switch-note">'
        '<span class="tt-track" aria-hidden="true"><span class="tt-knob"></span></span>Tooltips '
        '<span class="tt-state" aria-hidden="true">on</span></button>\n'
        '      <span class="bar-notes">\n'
        '        <span class="note wt-note">Read the page one section at a time, in order, with Back and Next buttons. The normal page stays as it is.</span>\n'
        '        <span class="note" id="tt-switch-note">Tooltips explain terms, controls and table columns when you hover over, focus or tap them; a table cell\'s tooltip shows its whole row.</span>\n'
        '      </span>\n'
        '    </div>\n'
    )
    s = once(s, old_bar, new_bar, "walkthrough bar")

    # 3. the script, after the last </script> (the walkthrough script)
    js = open(os.path.join(HERE, "e5_tooltips.js"), encoding="utf-8").read()
    cols = json.dumps(COLS, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    assert js.count("__COLS__") == 1
    js = js.replace("__COLS__", cols)
    assert s.endswith("</script>\n"), "the page no longer ends with its last script"
    assert s.rstrip("\n").endswith("})();\n</script>")
    s = s + "\n" + js

    open(PAGE, "w", encoding="utf-8").write(s)
    assert md5(BACKUP) == backup_md5, "the backup changed"
    print(f"applied: {len(s):,} bytes, md5 {md5(PAGE)}; backup untouched ({backup_md5})")
    print(f"column descriptions: {sum(len(v) for v in COLS.values())} tables, "
          f"{sum(len(e['cols']) for v in COLS.values() for e in v)} names")


if __name__ == "__main__":
    main()
