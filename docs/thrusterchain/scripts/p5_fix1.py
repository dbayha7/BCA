"""Version 14 fix: the estimated-weights row of finding F, medium-expert strong test, is 92.2496 points: 92.2, not 92.3."""
from pathlib import Path
PAGE = Path(__file__).resolve().parent.parent / "thrusterchain.html"
s = PAGE.read_text(encoding="utf-8")
old = '<tr><td>Estimated weights</td><td class="n">92.3</td>'
new = '<tr><td>Estimated weights</td><td class="n">92.2</td>'
assert s.count(old) == 1
PAGE.write_text(s.replace(old, new), encoding="utf-8", newline="
")
print("fixed")
