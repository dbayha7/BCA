import re, sys
P='/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s=open(P,encoding='utf-8').read()
def mj(x):
    return re.sub(r'<mjx-container[^>]*aria-label="([^"]*)"[^>]*>.*?</mjx-container>', r'$\1$', x, flags=re.S)
tabs=[m.start() for m in re.finditer(r'<table', s)]
for k in map(int, sys.argv[1].split(',')):
    a=tabs[k]; e=s.index('</table>',a)+8
    lim=int(sys.argv[2]) if len(sys.argv)>2 else 4000
    print(f'===== #{k}')
    print(mj(s[a:e])[:lim])
