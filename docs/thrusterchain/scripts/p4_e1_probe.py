#!/usr/bin/env python3
"""Read-only probe for pass-4 edit 1: search the page's visible text (formulas, scripts and styles stripped) for
patterns, or print raw HTML around a pattern.
  python3 scripts/p4_e1_probe.py text PAT [PAT ...]       visible-text context around the first hits
  python3 scripts/p4_e1_probe.py all PAT                  every visible-text hit
  python3 scripts/p4_e1_probe.py raw PAT [N]              raw HTML around each hit (N chars each side)
  python3 scripts/p4_e1_probe.py count PAT                count in raw HTML
"""
import re
import sys

PAGE = sys.argv[-1] if sys.argv[-1].endswith('.html') else 'thrusterchain.html'
args = [a for a in sys.argv[1:] if not a.endswith('.html')]
s = open(PAGE, encoding='utf-8').read()


def visible(src):
    t = re.sub(r'<mjx-container.*?</mjx-container>', '[M]', src, flags=re.S)
    t = re.sub(r'<script.*?</script>', '', t, flags=re.S)
    t = re.sub(r'<style.*?</style>', '', t, flags=re.S)
    t = re.sub(r'<template.*?</template>', '', t, flags=re.S)
    t = re.sub(r'<[^>]+>', '', t)
    return re.sub(r'\s+', ' ', t)


mode = args[0]
if mode in ('text', 'all'):
    t = visible(s)
    for pat in args[1:]:
        hits = [m for m in re.finditer(re.escape(pat), t)]
        print(f'## {pat!r}: {len(hits)} hits')
        for m in (hits if mode == 'all' else hits[:2]):
            print('   ', t[max(0, m.start() - 220):m.end() + 160])
            print()
elif mode == 'raw':
    pat = args[1]
    n = int(args[2]) if len(args) > 2 else 300
    for m in re.finditer(re.escape(pat), s):
        line = s.count('\n', 0, m.start()) + 1
        print(f'-- line {line} @ {m.start()}')
        print(s[max(0, m.start() - n):m.end() + n])
        print()
elif mode == 'count':
    print(s.count(args[1]))
