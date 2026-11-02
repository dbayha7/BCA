/* @ds-bundle: {"format":4,"namespace":"BCA","components":[{"name":"Button"},{"name":"Option"},{"name":"Tag"},{"name":"Key"},{"name":"Callout"},{"name":"Finding"},{"name":"SymbolLegend"},{"name":"Table"},{"name":"Readout"},{"name":"DotRange"},{"name":"Chart"},{"name":"Tooltip"},{"name":"PipelineStrip"},{"name":"DeckSlide"}]} */
/* BCA — React components over the ThrusterChain reference page's own markup and classes (components/bundle.css).
   Hand-written from page_build/wf7/build/core_v27.html; reads window.React and window.ReactDOM (18), assigns window.BCA. */
(function () {
  'use strict';
  var React = window.React;
  if (!React) { if (window.console) console.error('BCA: load React 18 before components/bundle.js'); return; }
  var h = React.createElement, Fragment = React.Fragment;
  var useState = React.useState, useEffect = React.useEffect, useLayoutEffect = React.useLayoutEffect,
      useRef = React.useRef, useId = React.useId;

  /* ------------------------------------------------------------------ helpers */
  function cx() {
    var out = [];
    for (var i = 0; i < arguments.length; i++) if (arguments[i]) out.push(arguments[i]);
    return out.join(' ');
  }
  function omit(p, keys) {
    var o = {};
    for (var k in p) if (Object.prototype.hasOwnProperty.call(p, k) && keys.indexOf(k) < 0) o[k] = p[k];
    return o;
  }
  /* data-sit / data-world / data-k: the attribute that names a colored thing */
  function thing(p) {
    var a = {};
    if (p.sit) a['data-sit'] = p.sit;
    if (p.world) a['data-world'] = p.world;
    if (p.k) a['data-k'] = p.k;
    return a;
  }
  function useControlled(value, defaultValue, onChange) {
    var s = useState(defaultValue);
    var controlled = value !== undefined;
    var cur = controlled ? value : s[0];
    function set(v) { if (!controlled) s[1](v); if (onChange) onChange(v); }
    return [cur, set];
  }
  /* a number with a true minus and fixed decimals */
  function num(v, d) {
    if (typeof v !== 'number' || !isFinite(v)) return String(v);
    var s = Math.abs(v).toFixed(d == null ? 1 : d);
    return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function para(c) { return typeof c === 'string' ? h('p', null, c) : c; }

  /* ------------------------------------------------------------------ Button and Transport */
  function Button(p) {
    var rest = omit(p, ['variant', 'className', 'children']);
    return h('button', Object.assign({ type: 'button' }, rest, {
      className: cx('btn', p.variant === 'ghost' && 'ghost', p.className)
    }), p.children);
  }

  function Transport(p) {
    var max = p.max == null ? 99 : p.max;
    var c = useControlled(p.t, p.defaultT || 0, p.onChange), t = c[0], setT = c[1];
    var pl = useState(false), playing = pl[0], setPlaying = pl[1];
    var tRef = useRef(t), setRef = useRef(setT);
    tRef.current = t; setRef.current = setT;
    useEffect(function () {
      if (!playing) return undefined;
      var id = setInterval(function () {
        var n = tRef.current + 1;
        if (n > max) { setPlaying(false); return; }
        setRef.current(n);
      }, p.stepMs || 120);
      return function () { clearInterval(id); };
    }, [playing, max, p.stepMs]);
    var label = p.label || function (v) { return 't = ' + v; };
    return h('div', { className: 'transport' },
      h(Button, {
        'aria-pressed': playing ? 'true' : 'false',
        onClick: function () {
          if (!playing && t >= max) setT(0);
          setPlaying(!playing);
        }
      }, playing ? 'Pause' : 'Play'),
      h(Button, { variant: 'ghost', onClick: function () { setPlaying(false); setT(Math.min(t + 1, max)); } }, 'Step'),
      h(Button, { variant: 'ghost', onClick: function () { setPlaying(false); setT(0); } }, 'Reset'),
      h('input', {
        type: 'range', min: 0, max: max, value: t, 'aria-label': p.ariaLabel || 'Time',
        onChange: function (e) { setPlaying(false); setT(Number(e.target.value)); }
      }),
      h('span', { className: 'tlabel', 'aria-live': 'off' }, label(t)));
  }

  /* ------------------------------------------------------------------ Option, OptionGroup, Panel, KeyValue */
  function Option(p) {
    var name = h('b', null,
      p.group === 'ctrl' ? h('span', { className: cx('sw', p.dash && 'dash') }) : null,
      p.name);
    return h('button', {
      type: 'button', className: cx('opt', p.className), 'data-group': p.group, 'data-value': p.value,
      'aria-pressed': p.selected ? 'true' : 'false', onClick: p.onSelect
    }, name,
      p.description ? h('span', null, p.description) : null,
      p.step ? h('span', { className: 'opt-step' }, p.step) : null);
  }

  function OptionGroup(p) {
    var c = useControlled(p.value, p.defaultValue, p.onChange), val = c[0], set = c[1];
    return h('div', { className: 'optgroup', role: 'group', 'aria-label': p.question },
      h('p', { className: 'optlbl' },
        p.n != null ? h('span', { className: 'optn' }, p.n) : null,
        p.question,
        p.hint ? h('span', { className: 'q' }, p.hint) : null),
      h('div', { className: 'optbtns' }, (p.options || []).map(function (o) {
        return h(Option, Object.assign({ key: o.value, group: p.group }, o, {
          selected: o.value === val, onSelect: function () { set(o.value); }
        }));
      })));
  }

  function Panel(p) {
    return h('div', { className: cx('panel', p.className) },
      p.title ? h('p', { className: 'panel-title' }, p.title) : null,
      p.children);
  }

  function KeyValue(p) {
    return h('dl', { className: cx('kv', p.stacked && 'kv1', p.pro && 'pro', p.className) },
      (p.items || []).map(function (it, i) {
        return h(Fragment, { key: i }, h('dt', null, it.label), h('dd', null, it.value));
      }));
  }

  /* ------------------------------------------------------------------ Tag and StatusList */
  function Tag(p) {
    return h('span', { className: cx('tag', p.variant, p.className) }, p.children);
  }
  function StatusList(p) {
    return h('ul', { className: 'g-status' }, (p.items || []).map(function (it, i) {
      return h('li', { key: i }, h(Tag, { variant: it.variant }, it.tag), it.text);
    }));
  }

  /* ------------------------------------------------------------------ Key, Mention, Legend */
  function Key(p) {
    return h('i', Object.assign({ className: cx('key', p.shape || 'dot', p.className), 'aria-hidden': 'true' }, thing(p)));
  }
  function Mention(p) {
    return h('span', Object.assign({ className: cx('km', p.className) }, thing(p)), p.children);
  }
  function Legend(p) {
    return h('div', { className: 'legend' }, (p.items || []).map(function (it, i) {
      return h('span', { key: i },
        it.swatch ? h('i', { className: 'mk', style: { background: 'var(--' + it.swatch + ')' }, 'aria-hidden': 'true' })
          : it.ring ? h('i', { className: 'mk ring', 'aria-hidden': 'true' })
          : it.line ? h('i', { className: 'sw', style: { background: 'var(--' + it.line + ')' }, 'aria-hidden': 'true' })
          : h(Key, it),
        it.label);
    }));
  }

  /* ------------------------------------------------------------------ Callout */
  function Callout(p) {
    var v = p.variant || 'warn';
    if (v === 'warn') {
      return h('div', { className: 'callout', role: p.role }, h('p', null, p.lead ? h('strong', null, p.lead) : null, p.lead ? ' ' : null, p.children));
    }
    return h('div', { className: cx('box', v === 'stop' && 'stop') },
      p.title ? h('h3', null, p.title) : null, para(p.children));
  }

  /* ------------------------------------------------------------------ Finding */
  function Finding(p) {
    return h('div', { className: 'finding' },
      h('p', { className: 'f-title' }, p.title),
      h(KeyValue, { items: (p.parts || []).map(function (x) { return { label: x.label, value: x.text }; }), stacked: p.layout !== 'columns', pro: true }));
  }

  /* ------------------------------------------------------------------ SymbolLegend */
  function SymbolLegend(p) {
    var heading = p.heading || 'What the symbols mean';
    var body = h('div', { className: 'symkey' },
      p.collapsible ? null : h('p', { className: 'symkey-h' }, heading),
      (p.groups || []).map(function (g, gi) {
        return h(Fragment, { key: gi },
          g.title ? h('p', { className: 'symkey-g' }, g.title) : null,
          h('dl', null, g.items.map(function (it, i) {
            return h('div', { key: i },
              h('dt', null, it.key ? h(Key, it.key) : null, it.symbol),
              h('dd', null, it.meaning));
          })));
      }),
      p.example ? h('p', { className: 'symkey-ex' }, h('b', null, 'Worked example.'), ' ', p.example) : null);
    if (!p.collapsible) return body;
    return h('details', { open: p.defaultOpen !== false },
      h('summary', null, heading),
      h('div', { className: 'inner' }, body));
  }

  /* ------------------------------------------------------------------ the floating tooltip (shared by Tooltip, Table) */
  function FloatingTip(p) {
    var ref = useRef(null);
    var s = useState(null), pos = s[0], setPos = s[1];
    useLayoutEffect(function () {
      var el = ref.current, r = p.anchor;
      if (!el || !r) return;
      var w = el.offsetWidth, hh = el.offsetHeight;
      var vw = document.documentElement.clientWidth, vh = window.innerHeight;
      var left = clamp(r.left, 8, Math.max(8, vw - w - 8));
      var top = r.bottom + 8;
      if (top + hh > vh - 8) top = Math.max(8, r.top - hh - 8);
      setPos({ left: left, top: top });
    }, [p.anchor && p.anchor.left, p.anchor && p.anchor.top, p.title, p.text]);
    return h('div', {
      ref: ref, id: p.id, role: 'tooltip', className: cx('tt', pos && 'on'),
      style: pos ? { left: pos.left + 'px', top: pos.top + 'px' } : { left: '-9999px', top: '0' }
    },
      p.title ? h('div', { className: 'tt-h' }, p.title) : null,
      p.text ? h('div', { className: 'tt-b' }, p.text) : null,
      p.more ? h('div', { className: 'tt-col' }, p.more) : null);
  }

  function useTip() {
    var s = useState(null), tip = s[0], setTip = s[1];
    function show(e, content) { setTip(Object.assign({ anchor: e.currentTarget.getBoundingClientRect() }, content)); }
    function hide() { setTip(null); }
    return [tip, show, hide];
  }

  /* ------------------------------------------------------------------ Tooltip: a term that explains itself */
  function Tooltip(p) {
    var id = useId();
    var t = useTip(), tip = t[0], show = t[1], hide = t[2];
    var content = { title: p.title, text: p.text, more: p.more };
    return h(Fragment, null,
      h(p.as || 'i', {
        'data-tip': '', tabIndex: 0, 'aria-describedby': tip ? id : undefined,
        onMouseEnter: function (e) { show(e, content); }, onMouseLeave: hide,
        onFocus: function (e) { show(e, content); }, onBlur: hide,
        onKeyDown: function (e) { if (e.key === 'Escape') hide(); }
      }, p.children),
      tip ? h(FloatingTip, Object.assign({ id: id }, tip)) : null);
  }

  /* a static tooltip box, for documentation */
  function TooltipBox(p) {
    return h('div', { className: 'tt on', role: 'tooltip', style: { position: 'static' } },
      p.title ? h('div', { className: 'tt-h' }, p.title) : null,
      p.text ? h('div', { className: 'tt-b' }, p.text) : null,
      p.more ? h('div', { className: 'tt-col' }, p.more) : null);
  }

  /* ------------------------------------------------------------------ Table */
  function Table(p) {
    var cols = p.columns || [];
    var boxRef = useRef(null);
    var sc = useState({ scrolls: false, atEnd: false }), scroll = sc[0], setScroll = sc[1];
    var t = useTip(), tip = t[0], show = t[1], hide = t[2];
    var tipId = useId();
    useEffect(function () {
      var el = boxRef.current;
      if (!el) return undefined;
      function check() {
        var s = el.scrollWidth > el.clientWidth + 1;
        var end = el.scrollLeft + el.clientWidth >= el.scrollWidth - 2;
        setScroll(function (o) { return o.scrolls === s && o.atEnd === end ? o : { scrolls: s, atEnd: end }; });
      }
      check();
      el.addEventListener('scroll', check, { passive: true });
      var ro = window.ResizeObserver ? new ResizeObserver(check) : null;
      if (ro) ro.observe(el);
      return function () { el.removeEventListener('scroll', check); if (ro) ro.disconnect(); };
    }, []);
    function head(c) {
      var tipProps = c.desc ? {
        'data-tt-desc': '', tabIndex: 0, 'aria-describedby': tip && tip.col === c.key ? tipId : undefined,
        onMouseEnter: function (e) { show(e, { title: c.label, text: c.desc, col: c.key }); }, onMouseLeave: hide,
        onFocus: function (e) { show(e, { title: c.label, text: c.desc, col: c.key }); }, onBlur: hide
      } : {};
      return h('th', Object.assign({ key: c.key, scope: 'col', className: c.numeric ? 'n' : undefined }, tipProps), c.label);
    }
    var rows = (p.rows || []).map(function (r, i) {
      if (r.group != null) {
        var label = (r.sit || r.world || r.k) ? h(Mention, { sit: r.sit, world: r.world, k: r.k }, r.group) : r.group;
        return h('tr', Object.assign({ key: i, className: 'sub' }, r.sit ? { 'data-sit': r.sit } : {}),
          h('td', { colSpan: cols.length }, label));
      }
      return h('tr', { key: i, className: cx(r.highlight && 'hl', r.anchor && 'anchor') || undefined },
        cols.map(function (c) {
          var v = r.cells ? r.cells[c.key] : undefined;
          return h('td', { key: c.key, className: c.numeric ? 'n' : undefined }, v == null ? '' : v);
        }));
    });
    return h('div', { className: cx('tblwrap', scroll.scrolls && 'scrolls', scroll.atEnd && 'at-end') },
      h('span', { className: 'hint' }),
      h('div', { className: 'tbl', ref: boxRef },
        h('table', { className: p.className },
          p.caption ? h('caption', null, p.caption) : null,
          h('thead', null, h('tr', null, cols.map(head))),
          h('tbody', null, rows))),
      tip ? h(FloatingTip, Object.assign({ id: tipId }, tip)) : null);
  }

  /* ------------------------------------------------------------------ Readout */
  function Readout(p) {
    return h('div', { className: 'readouts', 'aria-live': p.live ? 'polite' : 'off' }, (p.items || []).map(function (it, i) {
      var v = it.tone === 'stall' ? h('span', { className: 'stallv' }, it.value)
        : it.tone === 'region' ? h('span', { style: { color: 'var(--k-region-ink)' } }, it.value)
        : it.value;
      return h('div', { key: i, className: 'ro' }, h('span', { className: 'k' }, it.name), h('span', { className: 'v' }, v));
    }));
  }

  /* ------------------------------------------------------------------ DotRange */
  function DotRange(p) {
    var lo = p.min, hi = p.max, span = hi - lo;
    function pct(v) { return clamp((v - lo) / span * 100, 0, 100); }
    var fmt = p.format || function (v) { return num(v, 1); };
    var tickFmt = p.tickFormat || function (v) { return num(v, 0); };
    var ticks = p.ticks || [];
    var grid = ticks.map(function (v) {
      return v === 0 ? 'linear-gradient(var(--rule-strong), var(--rule-strong))' : 'linear-gradient(var(--rule), var(--rule))';
    }).join(', ');
    var gridPos = ticks.map(function (v) { return pct(v).toFixed(2) + '% 0'; }).join(', ');
    var trackStyle = ticks.length ? { backgroundImage: grid, backgroundPosition: gridPos } : undefined;
    var body = (p.rows || []).map(function (r, i) {
      if (r.group != null) return h('div', { key: i, className: 'dr-grp' }, r.group);
      var dot = pct(r.mean), l = pct(r.lo), w = pct(r.hi) - l;
      var label = typeof r.name === 'string' ? r.name : (r.label || '');
      var tipText = label + ': ' + fmt(r.mean) + '; lowest to highest seed: ' + fmt(r.lo) + ' to ' + fmt(r.hi);
      return h('div', { key: i, className: cx('dr-row', r.kind || 'lo'), tabIndex: 0, 'aria-label': tipText },
        h('span', { className: 'dr-name' }, r.strong ? h('b', null, r.name) : r.name, r.note ? ' ' + r.note : null),
        h('span', { className: 'dr-val' }, fmt(r.mean)),
        h('span', { className: 'dr-track', style: trackStyle },
          h('span', { className: 'dr-rng', style: { left: l.toFixed(2) + '%', width: w.toFixed(2) + '%' } }),
          h('span', { className: 'dr-dot', style: { left: dot.toFixed(2) + '%' } }),
          h('span', { className: 'dr-tip', style: { right: (100 - dot).toFixed(2) + '%' }, 'aria-hidden': 'true' }, tipText)));
    });
    return h('figure', { className: 'chartbox drbox' },
      h('div', { className: 'dr' },
        h('div', { className: 'dr-axis dr-head' },
          h('span', { className: 'dr-name' }, p.nameHeading || 'Arm'),
          h('span', { className: 'dr-val' }, p.valueHeading || 'Mean'),
          h('span', { className: 'dr-ticks' }, ticks.map(function (v) {
            return h('span', { key: v, style: { left: pct(v).toFixed(2) + '%' } }, tickFmt(v));
          }))),
        body,
        p.axisLabel ? h('div', { className: 'dr-axis dr-axlab' }, h('span', { className: 'dr-name' }), h('span', { className: 'dr-val' }),
          h('span', { className: 'dr-ticks' }, h('span', { className: 'lab' }, p.axisLabel))) : null),
      p.legend ? h(Legend, { items: p.legend }) : null,
      p.caption ? h('figcaption', null, p.caption) : null);
  }

  /* ------------------------------------------------------------------ Chart */
  function valueAt(pts, xv) {
    if (!pts.length) return NaN;
    if (xv <= pts[0][0]) return pts[0][1];
    for (var i = 1; i < pts.length; i++) {
      if (xv <= pts[i][0]) {
        var a = pts[i - 1], b = pts[i];
        return b[0] === a[0] ? b[1] : a[1] + (b[1] - a[1]) * (xv - a[0]) / (b[0] - a[0]);
      }
    }
    return pts[pts.length - 1][1];
  }

  function Chart(p) {
    var W = p.width || 560, H = p.height || 304;
    var m = Object.assign({ l: 48, r: 16, t: 16, b: 48 }, p.margin || {});
    var X = p.x, Y = p.y, pw = W - m.l - m.r, ph = H - m.t - m.b;
    function sx(v) { return m.l + (v - X.min) / (X.max - X.min) * pw; }
    function sy(v) { return m.t + (1 - (v - Y.min) / (Y.max - Y.min)) * ph; }
    var fx = X.format || function (v) { return num(v, 2); };
    var fy = Y.format || function (v) { return num(v, 2); };
    var series = p.series || [];
    var s = useState(null), hx = s[0], setHx = s[1];
    var svgRef = useRef(null);
    var step = p.step || (X.max - X.min) / 100;
    function snap(v) { return clamp(Math.round((v - X.min) / step) * step + X.min, X.min, X.max); }
    function onMove(e) {
      var r = svgRef.current.getBoundingClientRect();
      var px = (e.clientX - r.left) * W / r.width;
      setHx(snap(X.min + (px - m.l) / pw * (X.max - X.min)));
    }
    function onKey(e) {
      var cur = hx == null ? X.min : hx, n = null;
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') n = cur + step;
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') n = cur - step;
      else if (e.key === 'Home') n = X.min;
      else if (e.key === 'End') n = X.max;
      else if (e.key === 'Escape') { setHx(null); return; }
      if (n != null) { e.preventDefault(); setHx(snap(n)); }
    }
    function path(pts) {
      return pts.map(function (pt, i) { return (i ? 'L' : 'M') + sx(pt[0]).toFixed(1) + ',' + sy(pt[1]).toFixed(1); }).join(' ');
    }
    var kids = [];
    (Y.ticks || []).forEach(function (v) {
      if (v !== Y.min) kids.push(h('line', { key: 'g' + v, className: 'grid', x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }));
    });
    kids.push(h('line', { key: 'ax', className: 'axis', x1: m.l, x2: W - m.r, y1: sy(Y.min), y2: sy(Y.min) }));
    kids.push(h('line', { key: 'ay', className: 'axis', x1: m.l, x2: m.l, y1: m.t, y2: sy(Y.min) }));
    (Y.ticks || []).forEach(function (v) {
      kids.push(h('text', { key: 'yt' + v, className: 'tick', x: m.l - 8, y: sy(v) + 4, textAnchor: 'end' }, (Y.tickFormat || fy)(v)));
    });
    (X.ticks || []).forEach(function (v) {
      kids.push(h('text', { key: 'xt' + v, className: 'tick', x: sx(v), y: sy(Y.min) + 18, textAnchor: 'middle' }, (X.tickFormat || fx)(v)));
    });
    if (X.label) kids.push(h('text', { key: 'xl', className: 'axlab', x: m.l + pw / 2, y: H - 6, textAnchor: 'middle' }, X.label));
    if (Y.label) kids.push(h('text', { key: 'yl', className: 'axlab', transform: 'translate(12 ' + (m.t + ph / 2) + ') rotate(-90)', textAnchor: 'middle' }, Y.label));
    (p.guides || []).forEach(function (g, i) {
      if (g.x != null) kids.push(h('line', { key: 'gd' + i, className: g.className || 'knee', x1: sx(g.x), x2: sx(g.x), y1: m.t, y2: sy(Y.min) }));
      else if (g.y != null) kids.push(h('line', { key: 'gd' + i, className: g.className || 'knee', x1: m.l, x2: W - m.r, y1: sy(g.y), y2: sy(g.y) }));
    });
    series.forEach(function (se) {
      kids.push(h('path', {
        key: 's' + se.id, className: se.className, d: path(se.points),
        style: se.className ? undefined : { stroke: 'var(--' + (se.color || 'ink-2') + ')', strokeWidth: 2.6 }
      }));
    });
    (p.marks || []).forEach(function (mk, i) {
      kids.push(h('circle', { key: 'm' + i, className: mk.className, cx: sx(mk.x), cy: sy(mk.y), r: mk.r || 4.5 }));
    });
    (p.labels || []).forEach(function (lb, i) {
      kids.push(h('text', { key: 'l' + i, className: lb.className || 'lab', x: sx(lb.x), y: sy(lb.y), textAnchor: lb.anchor || 'start' }, lb.text));
    });
    if (hx != null) {
      kids.push(h('line', { key: 'xh', className: 'xhair', x1: sx(hx), x2: sx(hx), y1: m.t, y2: sy(Y.min) }));
      series.forEach(function (se) {
        var v = valueAt(se.points, hx);
        kids.push(h('circle', { key: 'xd' + se.id, className: 'xdot', cx: sx(hx), cy: sy(v), r: 4.5, style: { fill: 'var(--' + (se.color || 'ink-2') + ')' } }));
      });
    }
    var read = hx == null
      ? (p.hint || 'Point at the chart, or focus it and use the arrow keys, to read every line at one point.')
      : h(Fragment, null, h('b', null, (X.readFormat || fx)(hx)), series.map(function (se) {
        return h('span', { key: se.id, className: 'chart-read-i' },
          h('i', { className: 'sw', style: { background: 'var(--' + (se.color || 'ink-2') + ')' }, 'aria-hidden': 'true' }),
          se.label + ' ' + (se.format || fy)(valueAt(se.points, hx)));
      }));
    return h('figure', { className: cx('chartbox', p.wide && 'wide') },
      h('svg', {
        ref: svgRef, className: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', tabIndex: 0,
        'aria-label': p.ariaLabel || p.caption || 'Chart',
        onPointerMove: onMove, onPointerLeave: function () { setHx(null); }, onKeyDown: onKey,
        onBlur: function () { setHx(null); }
      }, kids),
      p.legend ? h(Legend, { items: p.legend }) : null,
      h('div', { className: 'chart-read', 'aria-live': 'polite' }, read),
      p.caption ? h('figcaption', null, p.caption) : null);
  }

  /* ------------------------------------------------------------------ PipelineStrip and StepChip */
  var STEP_ORDER = ['collect', 'remove', 'train', 'band', 'test'];
  function StepChip(p) {
    var n = STEP_ORDER.indexOf(p.step) + 1;
    return h('span', { className: cx('stepchip', p.inline && 'in'), 'data-step': p.step }, p.children || (n ? 'Step ' + n : ''));
  }

  function PipelineStrip(p) {
    var steps = p.steps || [];
    var c = useControlled(p.here, p.defaultHere || [], p.onHereChange), here = c[0], setHere = c[1];
    if (!p.compact) {
      return h('nav', { className: 'pipe', 'aria-label': p.ariaLabel || 'The experiment in five steps' },
        h('ol', null, steps.map(function (st, i) {
          return h('li', { key: st.step, className: 'pipe-step', 'data-step': st.step },
            h('div', { className: 'pipe-card' },
              h('p', { className: 'pipe-head' }, h('span', { className: 'pipe-n' }, i + 1), h('span', { className: 'pipe-t' }, st.title)),
              [].concat(st.text || []).map(function (tx, j) { return h('p', { key: j, className: 'pipe-d' }, tx); }),
              st.why ? h('p', { className: 'pipe-why' }, h('b', null, 'Why:'), ' ', st.why) : null,
              st.href ? h('a', { className: 'pipe-go', href: st.href }, st.go || st.title) : null));
        })));
    }
    return h('nav', { className: 'pipe pipe-sm', 'aria-label': p.ariaLabel || 'Where this sits in the experiment' },
      h('ol', null, steps.map(function (st, i) {
        var on = here.indexOf(st.step) >= 0;
        return h('li', { key: st.step, className: cx('pipe-step', on && 'is-here'), 'data-step': st.step },
          h('a', {
            href: st.href || '#', 'aria-current': on ? 'step' : undefined,
            onClick: p.selectable ? function (e) { e.preventDefault(); setHere([st.step]); if (p.onSelect) p.onSelect(st.step); } : p.onSelect ? function (e) { p.onSelect(st.step, e); } : undefined
          }, h('span', { className: 'pipe-n' }, i + 1), st.short || st.title));
      })));
  }

  /* ------------------------------------------------------------------ DeckSlide */
  var SLIDE_W = 1920, SLIDE_H = 1080;
  function DeckSlide(p) {
    var scale = p.scale || 0.25, navy = p.kind !== 'paper';
    var base = {
      position: 'absolute', left: 0, top: 0, width: SLIDE_W + 'px', height: SLIDE_H + 'px', boxSizing: 'border-box',
      transform: 'scale(' + scale + ')', transformOrigin: '0 0', fontFamily: 'var(--font-deck-sans)', fontSize: '30px', lineHeight: 1.4,
      display: 'flex', flexDirection: 'column', textAlign: 'left'
    };
    var title = { fontFamily: 'var(--font-deck-serif)', fontWeight: 600, margin: 0, padding: 0, border: 0, letterSpacing: 0 };
    var kicker = { fontSize: '24px', fontWeight: 600, letterSpacing: '3px', textTransform: 'uppercase', margin: 0 };
    var inner;
    if (navy) {
      var prog = p.progress;
      inner = h('section', {
        style: Object.assign({}, base, { background: 'var(--deck-ink)', color: 'var(--deck-on-ink)', padding: '128px', justifyContent: 'center', gap: '36px' })
      },
        (prog || p.kicker) ? h('div', { style: { display: 'flex', gap: '14px', alignItems: 'center' } },
          prog ? Array.apply(null, Array(prog.total)).map(function (_, i) {
            var cur = i + 1 === prog.current;
            return h('div', {
              key: i, style: {
                width: '56px', height: '56px', borderRadius: '50%', boxSizing: 'border-box', display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: cur ? 'var(--deck-step)' : 'transparent', border: cur ? '0' : '2px solid var(--deck-ring)',
                fontSize: '24px', fontWeight: cur ? 600 : 400, color: cur ? 'var(--deck-ink)' : 'var(--deck-step-off)'
              }
            }, i + 1);
          }) : null,
          p.kicker ? h('p', { style: Object.assign({}, kicker, { color: 'var(--deck-kicker)', paddingLeft: prog ? '16px' : 0 }) }, p.kicker) : null) : null,
        p.lead ? h('p', { style: { fontSize: p.cover ? '36px' : '30px', lineHeight: p.cover ? 1.35 : 1.4, color: 'var(--deck-on-ink-2)', width: '1400px', margin: 0 } }, p.lead) : null,
        h('h2', { style: Object.assign({}, title, { fontSize: p.cover ? '104px' : '76px', lineHeight: p.cover ? 1.05 : 1.1, color: 'var(--deck-on-ink)', width: '1500px' }) }, p.title),
        p.cover ? h('div', { style: { width: '160px', height: '6px', background: 'var(--deck-orange)' } }) : null,
        p.body ? h('p', { style: { fontSize: '30px', lineHeight: 1.4, color: 'var(--deck-on-ink)', width: '1400px', margin: 0 } }, p.body) : null,
        p.footnote ? h('p', { style: { position: 'absolute', left: '128px', bottom: '96px', width: '1200px', fontSize: '24px', color: 'var(--deck-on-ink-2)', margin: 0 } }, p.footnote) : null);
    } else {
      inner = h('section', {
        style: Object.assign({}, base, { background: 'var(--deck-paper)', color: 'var(--deck-ink)', padding: '128px 128px 160px', gap: '20px' })
      },
        p.kicker ? h('p', { style: Object.assign({}, kicker, { color: 'var(--deck-blue)' }) }, p.kicker) : null,
        h('h2', { style: Object.assign({}, title, { fontSize: '52px', lineHeight: 1.12, color: 'var(--deck-ink)' }) }, p.title),
        p.body ? h('p', { style: { fontSize: '30px', color: 'var(--deck-soft)', margin: 0 } }, p.body) : null,
        p.children,
        p.caution ? h('p', { style: { fontSize: '26px', lineHeight: 1.35, color: 'var(--deck-caution)', fontWeight: 500, margin: 0 } }, p.caution) : null,
        p.footnote ? h('p', { style: { position: 'absolute', left: '128px', bottom: '64px', width: '1664px', fontSize: '24px', color: 'var(--deck-footnote)', margin: 0 } }, p.footnote) : null);
    }
    return h('div', {
      role: 'img', 'aria-label': (p.kicker ? p.kicker + ': ' : '') + p.title,
      style: { position: 'relative', width: SLIDE_W * scale + 'px', height: SLIDE_H * scale + 'px', overflow: 'hidden', borderRadius: '6px', flex: 'none' }
    }, inner);
  }

  window.BCA = {
    Button: Button, Transport: Transport,
    Option: Option, OptionGroup: OptionGroup, Panel: Panel, KeyValue: KeyValue,
    Tag: Tag, StatusList: StatusList,
    Key: Key, Mention: Mention, Legend: Legend,
    Callout: Callout, Finding: Finding, SymbolLegend: SymbolLegend,
    Table: Table, Readout: Readout, DotRange: DotRange, Chart: Chart,
    Tooltip: Tooltip, TooltipBox: TooltipBox,
    PipelineStrip: PipelineStrip, StepChip: StepChip,
    DeckSlide: DeckSlide,
    format: { num: num }
  };
})();
