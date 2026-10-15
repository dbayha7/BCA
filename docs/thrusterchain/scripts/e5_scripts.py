import re
P='/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s=open(P,encoding='utf-8').read()
for m in re.finditer(r'<script[^>]*>', s):
    e=s.index('</script>', m.end())
    print(m.start(), e, e-m.end(), repr(m.group(0)), repr(s[m.end():m.end()+200]))
print(repr(s[-3000:]))
