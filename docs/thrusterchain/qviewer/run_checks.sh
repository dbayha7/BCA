#!/bin/bash
# All checks of the critic-in-action section on a fresh copy of the page (CPUs 2-3, nice 10). Writes only in qviewer/.
# 1. copy the live page to instest/fresh_page.html and test_page.html; 2. insert the section into test_page.html;
# 3. insertion checks (refusals, --replace, strip); 4. browser checks (qv_test.py, qv_wt_test.py, final_check.py).
Q=/mnt/c/Users/David/dev/bca-shift-work/qviewer
cd $Q || exit 1
cp ../page/thrusterchain.html instest/fresh_page.html && cp instest/fresh_page.html test_page.html || exit 1
export PYTHONDONTWRITEBYTECODE=1
taskset -c 2-3 nice -n 10 /opt/bca-shift-venv/bin/python insert_qviewer.py test_page.html || exit 1
bash ins_test.sh
S="taskset -c 2-3 nice -n 10 /opt/shot-venv/bin/python"
$S qv_test.py | tail -3
$S qv_wt_test.py
$S final_check.py
