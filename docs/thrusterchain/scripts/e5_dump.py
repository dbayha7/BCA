import sys
P='/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html'
s=open(P,encoding='utf-8').read()
a=int(sys.argv[1]); b=int(sys.argv[2])
sys.stdout.write(s[a:b])
