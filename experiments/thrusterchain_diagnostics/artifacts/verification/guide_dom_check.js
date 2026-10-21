// Structural DOM check; this is deliberately not a browser-rendering test.
// Usage: node guide_dom_check.js path/to/ThrusterChain_diagnostics.html
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[2], 'utf8');
const data = JSON.parse(html.match(/<script type="application\/json" id="diagnostics-oracle-data">([\s\S]*?)<\/script>/)[1]);
const script = html.match(/<script id="diagnostics-oracle-interaction">([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.listeners = {}; this.value = ''; this.textContent = ''; this.className = ''; }
  appendChild(child) { this.children.push(child); if (child.selected) this.value = child.value; return child; }
  replaceChildren() { this.children = []; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
}
const nodes = {};
for (const id of ['diagnostics-oracle-data','diag-regime','diag-load','diag-position','diag-time','diag-retention','diag-oracle-rows','diag-oracle-context','diag-oracle-witness']) nodes[id] = new Element('div');
nodes['diagnostics-oracle-data'].textContent = JSON.stringify(data);
nodes['diag-regime'].value = '0'; nodes['diag-load'].value = '1'; nodes['diag-retention'].value = 'zero';
const document = {getElementById: id => nodes[id], createElement: tag => new Element(tag)};
vm.runInNewContext(script, {document});
assert.equal(nodes['diag-position'].children.length, 9);
assert.equal(nodes['diag-time'].children.length, 8);
assert.equal(nodes['diag-position'].value, '7');
let selections = 0, supportedEmpty = 0, removed = 0;
for (let r=0;r<2;r++) for (let l=0;l<2;l++) for (let p=0;p<9;p++) for (let t=0;t<8;t++) for (const mode of ['full','rare','zero']) {
  Object.entries({regime:r,load:l,position:p,time:t,retention:mode}).forEach(([k,v]) => nodes['diag-'+k].value=String(v));
  nodes['diag-retention'].listeners.change();
  const s=data.states.findIndex(x=>x[0]===r&&x[1]===l&&x[2]===data.positions[p]);
  const rows=nodes['diag-oracle-rows'].children;
  assert.equal(rows.length,5);
  for(let a=0;a<5;a++) {
    const cells=rows[a].children.map(c=>c.textContent);
    assert.equal(Number.parseFloat(cells[1]),Number(data.q[t][s][a].toFixed(6)));
    assert.equal(cells[2],data.critical[t][s][a]?'Yes':'No');
    assert.equal(Number.parseFloat(cells[3]),100*data.retention[mode][t][s][a]);
    assert.equal(Number(cells[5]),data.counts[mode][t][s][a]);
    const eligible=data.logging_occupancy[t][s][a]>0&&data.retention[mode][t][s][a]>0;
    assert.equal(cells[4]==='Yes',eligible);
    if(eligible && Number(cells[5])===0)supportedEmpty++;
    if(cells[4]==='No · removed')removed++;
    assert.equal(rows[a].className==='diag-best',data.q[t][s][a]===Math.max(...data.q[t][s]));
  }
  assert(nodes['diag-oracle-context'].textContent.includes((8-t)+' steps remaining'));
  selections++;
}
console.log(JSON.stringify({selections,actionRows:selections*5,supportedZeroCountRows:supportedEmpty,removedRows:removed,passed:true,browserTest:false}));
