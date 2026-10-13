#!/bin/bash
# Rebuild the page: inject data and fragments, typeset math, wrap for local viewing, dump reader text.
set -e
cd "$(dirname "$0")"
# the After-the-stop numbers and status; stops when the state behind the page's status text has changed
/tmp/claude-0/bcavenv/bin/python make_beyond_data.py
# the learned actors' and the room check's arms' points, for the controller guide and the controller tooltips
/tmp/claude-0/bcavenv/bin/python make_controllers_data.py
/tmp/claude-0/bcavenv/bin/python build.py
NODE_PATH=../mathbuild/node_modules node ../mathbuild/render.js thrusterchain.html thrusterchain.math.html
mv thrusterchain.math.html thrusterchain.html
(echo '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style></head><body>'; cat thrusterchain.html; echo '</body></html>') > wrapped.html
node dumptext.js wrapped.html page_text.txt
