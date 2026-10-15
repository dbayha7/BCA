"""Edit 1 runner: rebuild thrusterchain.html from the backup with e1_moves.py then e1_content.py.

It refuses to write unless the page is still either identical to the backup or identical to what this runner wrote
last, so it can never overwrite another pass's edits. The backup is only read.
"""
import os
import subprocess
import sys

from e1_common import BACKUP, PAGE, md5

HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = os.path.join(HERE, ".e1_last_md5")
TMP = os.path.join(os.path.dirname(PAGE), "tmp", "e1_a.html")

allowed = {md5(BACKUP)}
if os.path.exists(STAMP):
    allowed.add(open(STAMP).read().strip())
cur = md5(PAGE)
if cur not in allowed:
    sys.exit(f"refusing: {PAGE} was changed by someone else (md5 {cur})")
os.makedirs(os.path.dirname(TMP), exist_ok=True)
bk = md5(BACKUP)
subprocess.run([sys.executable, os.path.join(HERE, "e1_moves.py"), BACKUP, TMP], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "e1_content.py"), TMP, PAGE], check=True)
assert md5(BACKUP) == bk, "the backup changed"
open(STAMP, "w").write(md5(PAGE))
print("page md5", md5(PAGE), "| backup md5", bk)
