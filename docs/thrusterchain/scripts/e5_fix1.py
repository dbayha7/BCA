"""Edit 1 fix: the visually hidden column descriptions stay in the flow of their header cell (an absolutely positioned
box escaped the sideways-scrolling tables and widened the page by 377 px at 390 px)."""
import hashlib

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v9.html"
OLD = ".tt-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }\n"
NEW = ("/* visually hidden, but kept in the flow of its cell: an absolute box would escape a sideways-scrolling table */\n"
       ".tt-sr { display: inline-block; width: 1px; height: 1px; margin-right: -1px; overflow: hidden; clip-path: inset(50%);\n"
       "  white-space: nowrap; vertical-align: top; }\n")
b0 = hashlib.md5(open(BACKUP, "rb").read()).hexdigest()
s = open(PAGE, encoding="utf-8").read()
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW, 1)
open(PAGE, "w", encoding="utf-8").write(s)
assert hashlib.md5(open(BACKUP, "rb").read()).hexdigest() == b0
print("fixed; md5", hashlib.md5(s.encode("utf-8")).hexdigest())
