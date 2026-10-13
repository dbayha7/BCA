// Usage: node dumptext.js <wrapped.html> <out.txt>  (rendered page text, as a reader sees it, phone width)
const { chromium } = require('playwright');
(async () => {
  const [src, out] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
  await p.goto("file://" + require('path').resolve(src)); await p.waitForTimeout(1000);
  const txt = await p.evaluate(() => {
    // math is SVG with aria-label holding its TeX; show it inline as [math: ...]
    document.querySelectorAll('mjx-container').forEach(m => m.replaceWith(document.createTextNode(' [math: ' + m.getAttribute('aria-label') + '] ')));
    document.querySelectorAll('details').forEach(d => d.open = true);
    return document.body.innerText;
  });
  require('fs').writeFileSync(out, txt); await b.close(); console.log("wrote", out, txt.length, "chars");
})();
