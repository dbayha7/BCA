# ThrusterChain page build

The sources that built the ThrusterChain page (https://claude.ai/artifact/8VbJdfNyQr1qDmr7JN1Ha7) in the cloud
container, copied here so they survive it. Built screenshots, backups and the 4.6 MB built HTML are left out.

`rebuild.sh` runs, in order: `make_beyond_data.py` and `make_controllers_data.py` (they read the gitignored
`runs/` folders), `build.py` (fills `template.html` with the `*_data.json` files), the math typesetting in
`mathbuild/render.js` (run `npm ci` in `mathbuild/` first), and `dumptext.js` (the page's plain text, for checks).
The scripts expect `/home/user/bca` and the Python at `/tmp/claude-0/bcavenv`.

The live page has moved on since this copy (13 October, about 12:43 UTC): a separate process has been editing it
from comments. Read the live page before rebuilding from this template, or the newer edits are lost.
