#!/bin/sh
# Test only (page version 26, built on version 25; made by make_tests_v26.py from run_v24.sh): every component's suite on
# the integrated page, the old full check, the regression test against version 25, then the timing tests last (nothing
# else running). Pages are set up by setup_tests_v26.py. One headless Edge at a time (each test launches its own and
# closes it). Usage: sh run_v26.sh
T=/c/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests
PY=/c/Users/David/dev/BCA/.venv/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $T/int/out $T/out $T/regression
echo "== start $(date -u +%H:%M:%S)"
sha256sum $T/../core_v26.html $T/../page_v26_preview.html $T/nav/build/nav_lesson_live.html $T/lesson/work/v26_core.html
# 0. static checks
cd $T && $PY -X utf8 static_v26.py > $T/out/static_v26.txt 2>&1; echo "static exit $?"
# 1. the lesson: the full test on the integrated page, the extra checks
cd $T/lesson/test && timeout 3600 node test_full.mjs $T/lesson/work/v26_core.html v26 > $T/lesson/test/full_v26_out.txt 2>&1; echo "lesson full exit $?"
cd $T/lesson/test && timeout 3600 node test_extra.mjs > $T/lesson/test/extra_v26_out.txt 2>&1; echo "lesson extra exit $?"
# 2. color: version 25 rendered as the learning curves' reference; all four views of the integrated page with the Whole
#    page switch on, the plan's checks, text shots
PRE="const w = document.getElementById('nv-whole'); if (w.getAttribute('aria-checked') !== 'true') w.click()"
cd $T/color/tools && PAGE_FILE=$T/color/work/v25_preview.html TAG=v25 SHOTS=0 RUNS=d1280_light,d1280_dark,m390_light,m390_dark timeout 3600 node render_test.mjs > $T/color/work/log_v25.txt 2>&1; echo "color render v25 exit $?"
cd $T/color/tools && PAGE_FILE=$T/../page_v26_preview.html TAG=v26int RUNS=d1280_light,d1280_dark,m390_light,m390_dark PRE="$PRE" timeout 3600 node render_test.mjs > $T/color/work/log_v26int.txt 2>&1; echo "color render exit $?"
cd $T/color/tools && node verify_render.mjs $T/color/work/render_v26int.json $T/color/work/render_v25.json > $T/color/work/verify_v26int.txt 2>&1; echo "color verify exit $?"
cd $T/color/tools && PAGE_FILE=$T/../page_v26_preview.html TAG=v26int PRE="$PRE" timeout 1800 node text_shots.mjs > $T/color/work/log_text_v26int.txt 2>&1; echo "color text exit $?"
# 3. the old full check, adapted to the navigation
cd $T/int && timeout 1800 node check3_v26.mjs > $T/int/out/check3_v26.log 2>&1; echo "check3 exit $?"
# 4. the regression test against version 25 (ids, lines, features in 4 views)
cd $T && timeout 3600 node regression_v25.mjs > $T/regression/regression_v25.log 2>&1; echo "regression exit $?"
# 5. the navigation (all but the timing test)
sh $T/nav/test/run_all.sh t_lists t_links t_walk t_history t_startbar t_deep t_kbd t_panels t_layout t_entries t_shots
# 6. timing, alone: the navigation's T-18, the load-time comparison with version 25, the lesson's load cost
sh $T/nav/test/run_all.sh t_timing
cd $T && timeout 1800 node timing_v26.mjs 5 > $T/out/timing_v26.log 2>&1; echo "timing exit $?"
V23=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/live5/core_4933.html
V24=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests/lesson/work/v26_core.html
cd $T/lesson/test && RUNS="[[\"v25 a\",\"$V23\"],[\"v26 a\",\"$V24\"],[\"v25 b\",\"$V23\"],[\"v26 b\",\"$V24\"]]" timeout 1800 node perf.mjs > $T/lesson/test/perf_v26_out.txt 2>&1; echo "lesson perf exit $?"
cd $T/nav/test && $PY -X utf8 summarize.py > $T/nav/test/out/summary.txt 2>&1; echo "summary exit $?"
echo "== end $(date -u +%H:%M:%S)"
