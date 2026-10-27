"""Keep navigation CSS and its two controllers on the same layout breakpoints."""
import re

BREAKPOINTS = [('1240', '1600'), ('1239', '1599'), ('1000', '1200'), ('999', '1199')]

def navigation_breakpoints(text):
    for old, new in BREAKPOINTS:
        text = re.sub(r'\b' + old + r'(?=\s*px\b)', new, text)
    return text

def patch_responsive(source):
    changed = {'style': 0, 'script': 0}
    def patch(match):
        tag, attrs, body = match.group(1, 2, 3)
        selected = ((tag == 'style' and '/* Navigation. Contents (#nv)' in body)
                    or (tag == 'script' and ('/* Before the page is drawn.' in body
                        or 'const mqWide = mm(' in body)))
        if selected:
            changed[tag] += 1
            body = navigation_breakpoints(body)
        return '<' + tag + attrs + '>' + body + '</' + tag + '>'
    result = re.sub(r'<(style|script)(\b[^>]*)>(.*?)</\1>', patch, source, flags=re.S)
    assert changed == {'style': 1, 'script': 2}, changed
    return result
