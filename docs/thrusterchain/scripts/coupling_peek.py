import re, html
s = open('/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html', encoding='utf-8').read()
i = s.find('<section', s.rfind('<section', 0, s.find('id="dynamics"')+1))
i = s.rfind('<section', 0, s.find('id="dynamics"')+1)
j = s.find('</section>', i)
seg = s[i:j]
print('len', len(seg))
seg = re.sub(r'<mjx-container[^>]*?aria-label="([^"]*)"[^>]*>.*?</mjx-container>', lambda m: '[$' + html.unescape(m.group(1)) + '$]', seg, flags=re.S)
seg = re.sub(r'<svg.*?</svg>', '[svg]', seg, flags=re.S)
seg = re.sub(r'<style.*?</style>', '[style]', seg, flags=re.S)
print(seg)
