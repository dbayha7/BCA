# Edit 3 adds one table (the push table in Transition Dynamics): e5_test.py's table count goes from 47 to 48.
p = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/e5_test.py'
s = open(p, encoding='utf-8').read()
old = 'check(len(cov) == 47, f"{len(cov)} tables found")'
new = 'check(len(cov) == 48, f"{len(cov)} tables found")  # 48 since edit 3 (the push table in Transition Dynamics)'
assert s.count(old) == 1
open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new, 1))
print('e5_test.py: table count 47 -> 48')
