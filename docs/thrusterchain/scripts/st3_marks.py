"""Read-only: count and show the exact boundary markers a move script would assert on."""
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
src = open(P, encoding="utf-8").read()
marks = ["  </header>\n", '\n    <figure class="panel" id="sim"', '\n  <section id="task">', '\n  <section id="action-space">',
         '\n  <section id="description">', '\n  <section id="version-history">', '\n  <section id="code-names">',
         '<a href="#version-history">', '  <a href="#version-history">Version History</a>\n',
         'Section 1 of 22', '\n    </figure>\n', '\n  </section>\n']
for mk in marks:
    i = src.find(mk)
    print(repr(mk), "count", src.count(mk), "first at", i, "ctx", repr(src[max(0, i - 50):i + len(mk) + 30]))
