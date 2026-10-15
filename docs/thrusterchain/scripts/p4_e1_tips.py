#!/usr/bin/env python3
"""Read-only: print the page's data-tip text for each given data-tip-title (every distinct text, with its count)."""
import collections
import html
import re
import sys
s = open('thrusterchain.html', encoding='utf-8').read()
for title in sys.argv[1:]:
    t = html.escape(title, quote=True).replace("'", '&#x27;')
    pat = re.compile(r'data-tip-title="%s" data-tip="([^"]*)"' % re.escape(t))
    c = collections.Counter(pat.findall(s))
    print(f'## {title}: {sum(c.values())} uses, {len(c)} texts')
    for k, v in c.most_common():
        print(f'  [{v}] {k}')
