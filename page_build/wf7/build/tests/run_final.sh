#!/bin/sh
# Test only (integration of page version 20): every component's suite on the integrated page, the old full check, then
# the timing tests last (nothing else running). Pages are set up by setup_tests.py. Usage: sh run_all.sh
T=/c/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests
PY=/c/Users/David/dev/BCA/.venv/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $T/int/out $T/out
echo "== start $(date -u +%H:%M:%S)"
sha256sum $T/../core_v20.html $T/../page_v20_preview.html $T/nav/build/nav_lesson_live.html $T/lesson/work/v20_core.html
# 0. static checks
cd $T && $PY -X utf8 static_v20.py > $T/out/static_v20.txt 2>&1; echo "static exit $?"
# 1. the lesson: the full test on the integrated page, the extra checks
cd $T/lesson/test && timeout 3600 node test_full.mjs $T/lesson/work/v20_core.html v20 > $T/lesson/test/full_v20_out.txt 2>&1; echo "lesson full exit $?"
cd $T/lesson/test && timeout 3600 node test_extra.mjs > $T/lesson/test/extra_v20_out.txt 2>&1; echo "lesson extra exit $?"
# 2. color: all four views of the integrated page with the Whole page switch on, the plan's checks, text shots
PRE="const w = document.getElementById('nv-whole'); if (w.getAttribute('aria-checked') !== 'true') w.click()"
cd $T/color/tools && PAGE_FILE=$T/../page_v20_preview.html TAG=v20int RUNS=d1280_light,d1280_dark,m390_light,m390_dark PRE="$PRE" timeout 3600 node render_test.mjs > $T/color/work/log_v20int.txt 2>&1; echo "color render exit $?"
cd $T/color/tools && node verify_render.mjs $T/color/work/render_v20int.json $T/color/work/render_v19.json > $T/color/work/verify_v20int.txt 2>&1; echo "color verify exit $?"
cd $T/color/tools && PAGE_FILE=$T/../page_v20_preview.html TAG=v20int PRE="$PRE" timeout 1800 node text_shots.mjs > $T/color/work/log_text_v20int.txt 2>&1; echo "color text exit $?"
# 3. the old full check, adapted to the navigation
cd $T/int && timeout 1800 node check3_v20.mjs > $T/int/out/check3_v20.log 2>&1; echo "check3 exit $?"
# 4. the navigation (all but the timing test)
sh $T/nav/test/run_all.sh t_lists t_links t_walk t_history t_startbar t_deep t_kbd t_panels t_layout t_entries t_shots
# 5. timing, alone
sh $T/nav/test/run_all.sh t_timing
V19=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/live3/core_9369.html
V20=C:/Users/David/AppData/Local/Temp/claude/C--Users-David-dev-BCA/3bdab4f0-86ee-439c-a05a-34828d6b0a71/scratchpad/wf7/build/tests/lesson/work/v20_core.html   # a Windows path: node reads it (fix round 1)
cd $T/lesson/test && RUNS="[[\"v19 a\",\"$V19\"],[\"v20 a\",\"$V20\"],[\"v19 b\",\"$V19\"],[\"v20 b\",\"$V20\"]]" timeout 1800 node perf.mjs > $T/lesson/test/perf_v20_out.txt 2>&1; echo "lesson perf exit $?"
cd $T/nav/test && $PY -X utf8 summarize.py > $T/nav/test/out/summary.txt 2>&1; echo "summary exit $?"
echo "== end $(date -u +%H:%M:%S)"
