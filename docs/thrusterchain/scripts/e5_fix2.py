"""Edit 1 fix 2: a superscript of more than one character reads e^(2c), not e^2c, in a tooltip."""
import hashlib

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v9.html"
REPL = [
    ('  const mapAll = (s, m) => [...s].every(ch => m[ch]) ? [...s].map(ch => m[ch]).join("") : null;\n',
     '  const mapAll = (s, m) => [...s].every(ch => m[ch]) ? [...s].map(ch => m[ch]).join("") : null;\n'
     '  const pow = s => mapAll(s, SUP) || (s.length > 1 ? "^(" + s + ")" : "^" + s);\n'),
    ('    .replace(/\\^\\{?([\\w+−-]+)\\}?/g, (m, g) => mapAll(g, SUP) || "^" + g)\n',
     '    .replace(/\\^\\{?([\\w+−-]+)\\}?/g, (m, g) => pow(g))\n'),
    ('      else if (tag === "sup") s += c.getAttribute("aria-label") ? " (" + c.getAttribute("aria-label") + ")" : '
     '(mapAll(plain(c).trim(), SUP) || "^" + plain(c).trim());\n',
     '      else if (tag === "sup") s += c.getAttribute("aria-label") ? " (" + c.getAttribute("aria-label") + ")" : '
     'pow(plain(c).trim());\n'),
]
b0 = hashlib.md5(open(BACKUP, "rb").read()).hexdigest()
s = open(PAGE, encoding="utf-8").read()
for old, new in REPL:
    assert s.count(old) == 1, (s.count(old), old[:60])
    s = s.replace(old, new, 1)
open(PAGE, "w", encoding="utf-8").write(s)
assert hashlib.md5(open(BACKUP, "rb").read()).hexdigest() == b0
print("fixed; md5", hashlib.md5(s.encode("utf-8")).hexdigest())
