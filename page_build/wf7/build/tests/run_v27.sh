#!/bin/sh
# Test only (page version 27, built on the live page after version 25; made by make_tests_v27.py from run_v26.sh): every
# component's suite on the integrated page, the old full check, the regression test against the live page, then the
# timing tests last (nothing else running). Pages are set up by setup_tests_v27.py. One headless Edge at a time (each
# test launches its own and closes it). Usage: sh run_v27.sh
T=/c/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests
PY=/c/Users/David/dev/BCA/.venv/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $T/int/out $T/out $T/regression
echo "== start $(date -u +%H:%M:%S)"
sha256sum $T/../core_v27.html $T/../page_v27_preview.html $T/nav/build/nav_lesson_live.html $T/lesson/work/v27_core.html
# 0. static checks
cd $T && $PY -X utf8 static_v27.py > $T/out/static_v27.txt 2>&1; echo "static exit $?"
# 1. the lesson: the full test on the integrated page, the extra checks
cd $T/lesson/test && timeout 3600 node test_full.mjs $T/lesson/work/v27_core.html v27 > $T/lesson/test/full_v27_out.txt 2>&1; echo "lesson full exit $?"
cd $T/lesson/test && timeout 3600 node test_extra.mjs > $T/lesson/test/extra_v27_out.txt 2>&1; echo "lesson extra exit $?"
# 2. color: the live page rendered as the learning curves' reference; all four views of the integrated page with the Whole
#    page switch on, the plan's checks, text shots
PRE="const w = document.getElementById('nv-whole'); if (w.getAttribute('aria-checked') !== 'true') w.click()"
cd $T/color/tools && PAGE_FILE=$T/color/work/l92ba_preview.html TAG=l92ba SHOTS=0 RUNS=d1280_light,d1280_dark,m390_light,m390_dark timeout 3600 node render_test.mjs > $T/color/work/log_l92ba.txt 2>&1; echo "color render live exit $?"
cd $T/color/tools && PAGE_FILE=$T/../page_v27_preview.html TAG=v27int RUNS=d1280_light,d1280_dark,m390_light,m390_dark PRE="$PRE" timeout 3600 node render_test.mjs > $T/color/work/log_v27int.txt 2>&1; echo "color render exit $?"
cd $T/color/tools && node verify_render.mjs $T/color/work/render_v27int.json $T/color/work/render_l92ba.json > $T/color/work/verify_v27int.txt 2>&1; echo "color verify exit $?"
cd $T/color/tools && PAGE_FILE=$T/../page_v27_preview.html TAG=v27int PRE="$PRE" timeout 1800 node text_shots.mjs > $T/color/work/log_text_v27int.txt 2>&1; echo "color text exit $?"
# 3. the old full check, adapted to the navigation
cd $T/int && timeout 1800 node check3_v27.mjs > $T/int/out/check3_v27.log 2>&1; echo "check3 exit $?"
# 4. the regression test against the live page (ids, lines, features in 4 views)
cd $T && timeout 3600 node regression_live.mjs > $T/regression/regression_live.log 2>&1; echo "regression exit $?"
# 4b. the toy section (the toy integration): its own test on the integrated page in six views, and inside the navigation
mkdir -p $T/toy
cd $T/toy && timeout 1800 node test_toy.mjs $T/../page_v27_preview.html $T/toy/result_v27.json > $T/toy/test_toy_v27.log 2>&1; echo "toy exit $?"
cd $T/toy && timeout 900 node nav_smoke.mjs $T/../page_v27_preview.html $T/toy/nav_v27.json > $T/toy/nav_smoke_v27.log 2>&1; echo "toy nav exit $?"
# 5. the navigation (all but the timing test)
sh $T/nav/test/run_all.sh t_lists t_links t_walk t_history t_startbar t_deep t_kbd t_panels t_layout t_entries t_shots
# 6. timing, alone: the navigation's T-18, the load-time comparison with the live page, the lesson's load cost
sh $T/nav/test/run_all.sh t_timing
cd $T && timeout 1800 node timing_v27.mjs 5 > $T/out/timing_v27.log 2>&1; echo "timing exit $?"
V23=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/live6/core_92ba.html
V24=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests/lesson/work/v27_core.html
cd $T/lesson/test && RUNS="[[\"live a\",\"$V23\"],[\"v27 a\",\"$V24\"],[\"live b\",\"$V23\"],[\"v27 b\",\"$V24\"]]" timeout 1800 node perf.mjs > $T/lesson/test/perf_v27_out.txt 2>&1; echo "lesson perf exit $?"
cd $T/nav/test && $PY -X utf8 summarize.py > $T/nav/test/out/summary.txt 2>&1; echo "summary exit $?"
echo "== end $(date -u +%H:%M:%S)"
