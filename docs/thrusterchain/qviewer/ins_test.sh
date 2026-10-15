#!/bin/bash
# Checks of insert_qviewer.py on the page copy (CPU 2, nice 10): it refuses, and writes nothing, on pages without its
# anchors; --replace gives identical bytes; taking the insertion out gives back the original page byte for byte.
# Usage: bash ins_test.sh   (needs instest/fresh_page.html = the page as copied, and test_page.html = it with the section)
cd /mnt/c/Users/David/dev/bca-shift-work/qviewer/instest || exit 1
PY="taskset -c 2 nice -n 10 /opt/bca-shift-venv/bin/python"
INS=../insert_qviewer.py
SRC=fresh_page.html
mkdir -p bad
mk() { cp $SRC bad/$1.html; }
mk nomethods; sed -i 's/<section id="methods">/<section id="methodz">/' bad/nomethods.html
mk twomethods; sed -i '0,/<section id="harm-checks"/s//<section id="methods"><\/section>\n<section id="harm-checks"/' bad/twomethods.html
mk notoc; sed -i 's/<nav class="toc"/<nav class="tocx"/' bad/notoc.html
mk nolink; sed -i 's/<a href="#methods"/<a href="#methodsx"/' bad/nolink.html
mk twolink; sed -i '/<nav class="toc"/,/<\/nav>/ s/\(<a href="#methods"[^>]*>[^<]*<\/a>\)/\1\n  \1/' bad/twolink.html   # a second Methods link inside the rail
mk twowrap; sed -i '0,/<div class="wrap">/s//<div class="wrap"><\/div><div class="wrap">/' bad/twowrap.html
mk nowrap; sed -i 's/<div class="wrap">/<div class="wrapx">/' bad/nowrap.html
cp ../test_page.html bad/already.html
printf '<title>x</title><p>hello</p>\n' > bad/unrelated.html
mk dupid; sed -i 's/<section id="harm-checks"/<section id="critic-in-action"><\/section><section id="harm-checks"/' bad/dupid.html
for f in nomethods twomethods notoc nolink twolink twowrap nowrap already unrelated dupid; do
  rm -f bad/$f.out.html
  msg=$($PY $INS bad/$f.html --out bad/$f.out.html 2>&1); rc=$?
  w=$([ -e bad/$f.out.html ] && echo WROTE || echo nothing-written)
  echo "[$f] exit=$rc $w :: $msg"
done
mkdir -p page; cp $SRC page/thrusterchain.html; a=$(md5sum page/thrusterchain.html | cut -c1-32)
msg=$($PY $INS page/thrusterchain.html 2>&1); rc=$?; b=$(md5sum page/thrusterchain.html | cut -c1-32)
echo "[in-place inside a folder named page, a stand-in] exit=$rc unchanged=$([ $a = $b ] && echo yes || echo NO) :: $msg"
$PY $INS ../test_page.html --out bad/replaced.html --replace >/dev/null 2>&1; cmp ../test_page.html bad/replaced.html && echo "[replace] byte-identical"
$PY $INS $SRC --out bad/again.html >/dev/null 2>&1; cmp ../test_page.html bad/again.html && echo "[insert twice from the copy] byte-identical"
$PY - <<'PYEOF'
import importlib.util
spec = importlib.util.spec_from_file_location("ins", "../insert_qviewer.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p = open("../test_page.html", encoding="utf-8", newline="").read(); o = open("fresh_page.html", encoding="utf-8", newline="").read()
s, n = m.strip_previous(p)
print("[strip] blocks taken out:", n, "| restores the original byte for byte:", s == o, "| lengths", len(s), len(o))
PYEOF
