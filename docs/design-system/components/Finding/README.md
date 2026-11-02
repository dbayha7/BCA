# Finding

A result written as a title and four labeled parts: the idea, what was tested (and the rule that would stop it), what happened, and why.

**Use for** every result in a Results section, in the order the reader needs them. The title states the finding as a sentence a reader could repeat ("Placement is almost the whole effect").

**Markup**

```html
<div class="finding">
  <p class="f-title">A harder pull at chosen rows (BCA's dose, as published)</p>
  <dl class="kv kv1 pro">
    <dt>The idea</dt><dd>…</dd>
    <dt>What we tested (and the kill rule)</dt><dd>…</dd>
    <dt>What happened</dt><dd>…</dd>
    <dt>Why</dt><dd>…</dd>
  </dl>
</div>
```

- A 3px `rule-strong` left edge, 14px in; the title in ink, 700.
- `dl.kv.kv1.pro` stacks each label over its text; labels in `accent-ink` 600, text in `ink`.
- `dl.kv` without `kv1` is the two-column form (label left, text right), used in the live panel's explanation box.

**Do** give numbers with their scale and seeds, and the date a test stopped. **Don't** soften a negative result or leave out the rule that decided it.
