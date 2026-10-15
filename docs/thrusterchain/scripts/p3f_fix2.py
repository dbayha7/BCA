"""Pass 3, final fixes, follow-up 2 (found while reading the final screenshots).

1. The weighting test's pass mark "Width tracks error" still said "the critic's exact error"; it now names the error
   it uses, the critic's error against the exact value, as the rest of the section does (finding on the two kinds of
   critic error).
2. A sideways-scrolling table of Stage 1 lost its caption's words off the left edge once scrolled, like its group
   headings did; the caption now stays in view the same way.
Exact and asserted; the v11 backup is only read.

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/p3f_fix2.py'
"""
import hashlib

P = "/mnt/c/Users/David/dev/bca-shift-work/page/"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
assert md5(P + "thrusterchain.v11.html") == "79f00557ba8803625b9f36bc497f7cc6"
assert md5(P + "thrusterchain.html") == "a1bdcddaa2daab19339e2babc07808a5", md5(P + "thrusterchain.html")
s = open(P + "thrusterchain.html", encoding="utf-8").read()
EDITS = [
    ("of width with the critic's exact error, averaged; its lower bound is above 0",
     "of width with the critic's error against the exact value, averaged; its lower bound is above 0"),
    ("/* the walkthrough changes sections without moving anything sideways",
     "/* a sideways-scrolling table's caption keeps its words in view, like its group headings */\n"
     ".subsec table.rows8 caption .cap { position: sticky; left: 10px; }\n"
     "/* the walkthrough changes sections without moving anything sideways"),
]
for old, new in EDITS:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(P + "thrusterchain.html", "w", encoding="utf-8", newline="").write(s)
assert md5(P + "thrusterchain.v11.html") == "79f00557ba8803625b9f36bc497f7cc6"
print("page md5", md5(P + "thrusterchain.html"))
