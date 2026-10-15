"""Helpers for putting re-typeset math (from mathbuild/render.js) into thrusterchain.html with exact, asserted edits.

The renderer runs on Windows node (WSL cannot execute node.exe here), so the flow is:
  1. write a job file mathbuild/jobs/<name>.json: [{"key": ..., "tex": ..., "display": bool}, ...]
  2. Git Bash:  cd /c/Users/David/dev/bca-shift-work/mathbuild &&
                /c/Users/David/AppData/Local/hermes/node/node.exe render.js jobs/<name>.json out/<name>.json
  3. WSL Python apply script:
        import mjx
        R = mjx.load('<name>')                                   # {key: container HTML}
        s = mjx.swap(s, old_label, R['key'], count=1)            # replaces whole containers, retagging ids
        mjx.check(s)                                             # every glyph id unique, every <use> resolves
Retagging: a rendered container's glyph ids "MJX-<k>-TEX-..." are renamed to "<PREFIX>-<n>-TEX-..." with n one above
the highest number that prefix already has on the page (default prefix MJXC; the page holds MJX-1..289 and
MJXB-29..325), so new containers never collide with old ones or with each other.
"""
import html as H
import json
import re
from collections import Counter

MATHBUILD = '/mnt/c/Users/David/dev/bca-shift-work/mathbuild'
CONTAINER = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>'
                       r'(.*?)</mjx-container>', re.S)
IDNUM = re.compile(r'(id="|href="#)(MJX[A-Z]*)-(\d+)-')


def load(name):
    """Rendered containers of mathbuild/out/<name>.json, as {key: html}."""
    return json.load(open(f'{MATHBUILD}/out/{name}.json', encoding='utf-8'))


def esc_label(tex):
    """The aria-label form of a TeX string, as the page writes it."""
    return tex.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;')


def label(container):
    """The TeX held in a container's aria-label (unescaped)."""
    return H.unescape(CONTAINER.match(container).group(1))


def next_number(page, prefix='MJXC'):
    nums = [int(m.group(3)) for m in IDNUM.finditer(page) if m.group(2) == prefix]
    return max(nums, default=0) + 1


def retag(container, page, prefix='MJXC'):
    """Rename the container's glyph ids to <prefix>-<n>- with n not yet used on the page."""
    found = {(m.group(2), m.group(3)) for m in IDNUM.finditer(container)}
    assert len(found) <= 1, f'one id number per container expected, found {found}'
    n = next_number(page, prefix)
    return IDNUM.sub(lambda m: f'{m.group(1)}{prefix}-{n}-', container)


def find(page, old_label_tex, display=None):
    """Spans (start, end) of every container whose aria-label is old_label_tex (TeX, unescaped)."""
    out = []
    for m in CONTAINER.finditer(page):
        if H.unescape(m.group(1)) == old_label_tex and (display is None or bool(m.group(2)) == display):
            out.append((m.start(), m.end()))
    return out


def swap(page, old_label_tex, new_container, count=1, display=None, lo=0, hi=None, prefix='MJXC'):
    """Replace every container labelled old_label_tex inside page[lo:hi] by new_container (each copy retagged to its
    own fresh id number). Asserts the number of containers replaced equals count, and that the display flag matches
    the old one's unless display is given."""
    hi = len(page) if hi is None else hi
    spans = [(a, b) for a, b in find(page, old_label_tex, display) if lo <= a and b <= hi]
    assert len(spans) == count, f'{old_label_tex!r}: expected {count} container(s) in range, found {len(spans)}'
    new_display = 'display="true"' in new_container.split('>', 1)[0]
    for a, b in reversed(spans):
        old_display = 'display="true"' in page[a:b].split('<svg', 1)[0]
        assert old_display == new_display, f'{old_label_tex!r}: display flag differs (old {old_display}, new {new_display})'
        page = page[:a] + retag(new_container, page, prefix) + page[b:]
    return page


def check(page):
    """Every glyph id on the page is unique, and every <use> inside a math SVG points to a glyph in that same SVG."""
    ids = re.findall(r'<path id="(MJX[A-Z]*-[^"]+)"', page)
    dup = {i for i, c in Counter(ids).items() if c > 1}
    assert not dup, f'duplicate glyph ids: {sorted(dup)[:10]}'
    for m in CONTAINER.finditer(page):
        body = m.group(3)
        own = set(re.findall(r'<path id="([^"]+)"', body))
        refs = set(re.findall(r'href="#([^"]+)"', body))
        assert refs <= own, f'{H.unescape(m.group(1))!r}: references outside its own svg: {sorted(refs - own)[:5]}'
        assert len({(x.group(2), x.group(3)) for x in IDNUM.finditer(body)}) <= 1, f'{m.group(1)!r}: mixed id numbers'
    return len(ids)
