#!/usr/bin/env node
/*
 * render.js: TeX -> <mjx-container> HTML matching the ThrusterChain page's pre-rendered math.
 *
 * Engine: mathjax-full 3.2.2 (installed here), TeX input with base + ams + newcommand + noundefined,
 * SVG output with fontCache 'local' (each <svg> carries its own <defs> of glyph paths), lite DOM adaptor.
 * Wrapper written exactly as on the page:
 *   <mjx-container role="img" aria-label="TEX" class="MathJax" jax="SVG"[ display="true"]><svg ...>...</svg></mjx-container>
 * aria-label = the TeX source (or the given label) with & -> &amp;, " -> &quot;, < -> &lt; (the page leaves > as is).
 * Glyph ids are "<PREFIX>-<N>-TEX-<variant>-<codepoint>", e.g. MJX-29-TEX-I-1D461 or MJXB-291-TEX-I-1D44F.
 * Every container on the page must use its own <PREFIX>-<N>; page/scripts/mjx.py renames the ids of a rendered
 * container to a number not yet on the page, at apply time (retag).
 *
 * Verified: re-rendering all 325 containers of thrusterchain.html (version 11) from their aria-labels, display flags,
 * prefixes and numbers gives 322 byte-identical containers; the other 3 are byte-identical when their bars |...| are
 * typeset as \vert ... \vert with the label kept (repro_test.js, repro_variants.js).
 *
 * Usage (Windows node; WSL cannot execute node.exe here, so run this from Git Bash or PowerShell):
 *   NODE=/c/Users/David/AppData/Local/hermes/node/node.exe
 *   $NODE render.js --tex "\hat{a}"                       # inline; prints the container HTML
 *   $NODE render.js --tex "x = 1" --display --prefix MJXC --id 7 [--label "aria text"]
 *   $NODE render.js in.json out.json                      # batch
 * Batch input: a JSON array; each item is either a TeX string (inline) or an object
 *   {"key": "...", "tex": "...", "display": false, "prefix": "MJXC", "id": 7, "label": "..."}
 * All fields but tex are optional: key defaults to the item's index, id to the item's 1-based position, prefix to
 * "MJX", label to tex (give label only when the aria-label text differs from the TeX typeset).
 * Batch output: a JSON object {key: containerHTML}. A TeX error stops the run with the offending TeX.
 */
'use strict';
const fs = require('fs');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
require('mathjax-full/js/input/tex/base/BaseConfiguration.js');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');
require('mathjax-full/js/input/tex/newcommand/NewcommandConfiguration.js');
require('mathjax-full/js/input/tex/noundefined/NoUndefinedConfiguration.js');

const PACKAGES = ['base', 'ams', 'newcommand', 'noundefined'];

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

function escLabel(tex) {
  return tex.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

// One MathJax document per call, so the localID option fully controls the glyph-id number.
function render(tex, opts) {
  opts = opts || {};
  const display = !!opts.display;
  const id = opts.id == null ? 1 : opts.id;
  const prefix = opts.prefix || 'MJX';
  if (!/^MJX[A-Z]*$/.test(prefix)) throw new Error('prefix must look like MJX, MJXB, MJXC ...: ' + prefix);
  const input = new TeX({packages: PACKAGES, formatError: (jax, err) => { throw new Error(err.message + ' in: ' + tex); }});
  const output = new SVG({fontCache: 'local', localID: id});
  const doc = mathjax.document('', {InputJax: input, OutputJax: output});
  const node = doc.convert(tex, {display, em: 16, ex: 8, containerWidth: 1280});
  let svg = adaptor.outerHTML(adaptor.firstChild(node));
  if (/data-mjx-error|data-mml-node="merror"/.test(svg)) throw new Error('TeX error in: ' + tex);
  if (prefix !== 'MJX') svg = svg.replace(/(id="|href="#)MJX-/g, '$1' + prefix + '-');
  const label = opts.label == null ? tex : opts.label;
  return '<mjx-container role="img" aria-label="' + escLabel(label) + '" class="MathJax" jax="SVG"' +
         (display ? ' display="true"' : '') + '>' + svg + '</mjx-container>';
}

function arg(argv, name) {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : undefined;
}

function main(argv) {
  if (argv[0] === '--tex') {
    const id = arg(argv, '--id');
    process.stdout.write(render(argv[1], {display: argv.includes('--display'), prefix: arg(argv, '--prefix'),
                                          id: id == null ? undefined : Number(id), label: arg(argv, '--label')}) + '\n');
    return;
  }
  if (argv.length !== 2) {
    process.stderr.write('usage: node render.js --tex "TeX" [--display] [--prefix MJXC] [--id N] [--label "aria text"]\n' +
                         '       node render.js in.json out.json\n');
    process.exit(2);
  }
  const items = JSON.parse(fs.readFileSync(argv[0], 'utf8'));
  const out = {};
  items.forEach((it, i) => {
    const o = typeof it === 'string' ? {tex: it} : it;
    const key = o.key == null ? String(i) : String(o.key);
    if (key in out) throw new Error('duplicate key ' + key);
    out[key] = render(o.tex, {display: o.display, prefix: o.prefix, label: o.label, id: o.id == null ? i + 1 : o.id});
  });
  fs.writeFileSync(argv[1], JSON.stringify(out, null, 1), 'utf8');
  process.stdout.write('rendered ' + Object.keys(out).length + ' -> ' + argv[1] + '\n');
}

module.exports = {render, escLabel};
if (require.main === module) main(process.argv.slice(2));
