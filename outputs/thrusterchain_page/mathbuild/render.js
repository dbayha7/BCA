// Render TeX in an HTML file to inline SVG at build time (MathJax 3, no runtime library, no fonts).
// Inline math: \( ... \)   Display math: \[ ... \]   Only the part before the first <script> is processed.
const fs = require('fs');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const [src, dst] = process.argv.slice(2);
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages.filter(p => !['bussproofs', 'autoload', 'require'].includes(p)) });
const svg = new SVG({ fontCache: 'local' });
const doc = mathjax.document('', { InputJax: tex, OutputJax: svg });
const esc = s => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const decode = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
let n = 0, errors = [];
function render(t, display) {
  const node = doc.convert(decode(t), { display, em: 16, ex: 8, containerWidth: 1200 });
  let out = adaptor.outerHTML(node);
  if (/data-mjx-error|merror/.test(out)) errors.push(t);
  n++;
  return out.replace('<mjx-container', `<mjx-container role="img" aria-label="${esc(decode(t).trim())}"`);
}
let html = fs.readFileSync(src, 'utf8');
const cut = html.indexOf('<script>');
let head = html.slice(0, cut), tail = html.slice(cut);
head = head.replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => `<div class="math-display">${render(t, true)}</div>`);
head = head.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => render(t, false));
const css = adaptor.textContent(svg.styleSheet(doc));
head = head.replace('/*__MATHCSS__*/', css);
fs.writeFileSync(dst, head + tail);
console.log(`rendered ${n} expressions` + (errors.length ? `; ERRORS in: ${errors.join(' | ')}` : '; no errors'));
if (errors.length) process.exit(1);
