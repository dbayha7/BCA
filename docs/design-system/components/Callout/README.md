# Callout

Three boxes for text that must stand apart from the running column: a warning, a neutral box, and a decision.

**Use for**
- `.callout`: a warning about the stall or a result that could mislead, its bold lead in `warn` on `warn-soft`.
- `.box`: a self-contained aside with its own `h3`, on `surface` with a `rule` border.
- `.box.stop`: a rule that decides (a pre-registered stop rule, a kill rule). It is neutral, a 4px `ink-2` edge, because red marks the stall and nothing else.

**Markup**

```html
<div class="callout"><p><strong>Past 80% the push falls.</strong> …</p></div>
<div class="box"><h3>The recording controller</h3><p>…</p></div>
<div class="box stop"><h3>The stop rule</h3><p>…</p></div>
```

All three are `radius-8`, padded `space-14` by `space-16`, with `space-8`–`space-10` between their children, and capped at 74–80ch. Text inside stays `ink`.

**Don't** stack callouts, put a link on the warning tint in dark (it falls to 6.3:1), or add a colored left border to any other box.
