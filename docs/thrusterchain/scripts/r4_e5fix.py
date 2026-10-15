# Edit 4 adds three tables (the caution variants in Results; the data qualities and the exploratory BCA runs in the
# weighting test): e5_test.py's table count goes from 48 to 51.
p = '/mnt/c/Users/David/dev/bca-shift-work/page/scripts/e5_test.py'
s = open(p, encoding='utf-8').read()
old = 'check(len(cov) == 48, f"{len(cov)} tables found")  # 48 since edit 3 (the push table in Transition Dynamics)'
new = ('check(len(cov) == 51, f"{len(cov)} tables found")  # 51 since edit 4 (48 after edit 3; +1 in Results, +2 in the '
       'weighting test)')
assert s.count(old) == 1
open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new, 1))
print('e5_test.py: table count 48 -> 51')
