"""Pass 3, final fixes, follow-up 1: the walkthrough's entering section only fades in.

p3f_apply.py moved the entering section 10 px vertically. The walkthrough scrolls to a heading right after it mounts
a section, while that move is still under way, so a link to a heading inside another section landed 10 px off
(s1_wt_test.py: "a link to a heading inside a section scrolls to it ['The task', -10]"). The entering section now
only fades in; the leaving section still moves 10 px up or down as it fades. Nothing moves sideways, so the pop-out is
never wider than its box. Exact and asserted; the v11 backup is only read.

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/p3f_fix1.py'
"""
import hashlib

P = "/mnt/c/Users/David/dev/bca-shift-work/page/"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
assert md5(P + "thrusterchain.v11.html") == "79f00557ba8803625b9f36bc497f7cc6"
assert md5(P + "thrusterchain.html") == "8388ed951558135eb8657c271463c774", md5(P + "thrusterchain.html")
s = open(P + "thrusterchain.html", encoding="utf-8").read()
old = """/* the walkthrough changes sections with a short vertical move, so nothing is ever wider than the pop-out */
.wt-stage.wt-leave-next { transform: translateY(-10px); }
.wt-stage.wt-leave-back { transform: translateY(10px); }
.wt-stage.wt-enter-next { transform: translateY(10px); }
.wt-stage.wt-enter-back { transform: translateY(-10px); }
"""
new = """/* the walkthrough changes sections without moving anything sideways, so nothing is ever wider than the pop-out:
   the leaving section fades with a short vertical move, and the entering one only fades in (it is scrolled to a
   heading at once, so it must already sit in place) */
.wt-stage.wt-leave-next { transform: translateY(-10px); }
.wt-stage.wt-leave-back { transform: translateY(10px); }
.wt-stage.wt-enter-next, .wt-stage.wt-enter-back { transform: none; }
"""
assert s.count(old) == 1
s = s.replace(old, new)
open(P + "thrusterchain.html", "w", encoding="utf-8", newline="").write(s)
assert md5(P + "thrusterchain.v11.html") == "79f00557ba8803625b9f36bc497f7cc6"
print("page md5", md5(P + "thrusterchain.html"))
