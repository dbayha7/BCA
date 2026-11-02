# Table

A bordered table in its own scrolling box, numbers in mono and right-aligned, with group rows that carry a situation's color.

**Use for** every set of numbers a reader compares, and beside every chart whose exact values matter.

**Markup**

```html
<div class="tblwrap"><span class="hint"></span>
<div class="tbl"><table>
  <caption>…how to read it, with the scale…</caption>
  <thead><tr><th>Controller</th><th class="n">States past 80%</th>…<th class="n">Points</th></tr></thead>
  <tbody>
    <tr class="sub" data-sit="strong_current"><td colspan="5"><span class="km" data-sit="strong_current">Strong current</span></td></tr>
    <tr class="anchor"><td>Stall-aware controller</td><td class="n">0.0%</td>…<td class="n">100.0</td></tr>
    <tr class="hl"><td>Rising-thrust controller</td><td class="n">84.0%</td>…<td class="n">−54.1</td></tr>
  </tbody>
</table></div></div>
```

- `.tbl`: `surface`, `rule` border, `radius-8`, scrolls sideways; `.tblwrap.scrolls` shows "Wider than the screen: scroll the table sideways →" in `accent-ink`.
- Cells 7px by 11px, `rule` lines between rows and columns, 15px with tabular figures; header on `surface-2`, 14px/600.
- `td.n`, `th.n`: right-aligned mono 13.5px. A number column's header stays in the body face.
- `tr.sub`: a group row on `surface-2`, 600; with `data-sit` it takes the situation's tint and a 4px edge in its color.
- `tr.hl`: a row on `warn-soft`, for a row about the stall; `tr.anchor`: the reference row (the stall-aware controller), its first cell 600.
- `table.grouped` adds a centered group-heading row (`tr.grp`) underlined in `rule-strong`.
- `caption` sits on top, 14px `ink-2`, and says how to read the table.
