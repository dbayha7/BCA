import re, sys
P='/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s=open(P,encoding='utf-8').read()
pat=sys.argv[1]; w=int(sys.argv[2]) if len(sys.argv)>2 else 200
lo=int(sys.argv[3]) if len(sys.argv)>3 else 0
hi=int(sys.argv[4]) if len(sys.argv)>4 else len(s)
for m in re.finditer(pat, s[lo:hi]):
    p=m.start()+lo
    print('@',p, repr(s[max(0,p-w):p+w]))
    print()
