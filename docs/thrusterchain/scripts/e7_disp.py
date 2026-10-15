import re
s = open("/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html", encoding="utf-8").read()
ms = list(re.finditer(r'<mjx-container[^>]*display="true"', s))
print(len(ms))
for m in ms:
    pre = s[max(0, m.start() - 120):m.start()]
    print(repr(pre[-70:]))
