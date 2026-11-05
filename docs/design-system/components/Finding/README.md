# Finding

A result written as a title and labeled parts that keep the formal verdict, what the data show, and the open questions apart.

**Use for** every result in a Results section. The parts, in order: **The idea** · **The test and its rule** · **Verdict** · **What the data show** · **Open questions**. The title states what was learned as a sentence a reader could repeat ("Placement is almost the whole effect"), not the verdict's label.

- **Verdict** is the predefined test's outcome under its own rule, exactly, and nothing else: a statement about that test, not about the idea.
- **What the data show** reads every tested value and arm on its own terms, including what the rule did not score. It is written for every result, whatever the verdict.
- **Open questions** names hypotheses as hypotheses, each with what would test it. A new claim needs its own rule written before its runs; the verdict above never changes.

**Markup** (CSS only; the React component renders the same)

```html
<div class="finding">
  <p class="f-title">A harder pull at chosen rows (BCA's dose, as published)</p>
  <dl class="kv kv1 pro">
    <dt>The idea</dt><dd>…</dd>
    <dt>The test and its rule</dt><dd>…</dd>
    <dt>Verdict</dt><dd>…</dd>
    <dt>What the data show</dt><dd>…</dd>
    <dt>Open questions</dt><dd>…</dd>
  </dl>
</div>
```

- A 3px `rule-strong` left edge, 14px in; the title in ink, 700.
- `dl.kv.kv1.pro` stacks each label over its text; labels in `accent-ink` 600, text in `ink`.
- `dl.kv` without `kv1` is the two-column form (label left, text right), used in the live panel's explanation box.

**Do** give numbers with their scale and seeds, and the date a test stopped. **Don't** soften a verdict or leave out the rule that decided it, and don't let the verdict stand in for the reading: an inconclusive or failed test still gets its "What the data show" and its open questions.

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { Finding } = window.BCA;
<Finding title="Placement is almost the whole effect"
  parts={[{ label: 'The idea', text: '…' }, { label: 'The test and its rule', text: '…' },
          { label: 'Verdict', text: '…' }, { label: 'What the data show', text: '…' },
          { label: 'Open questions', text: '…' }]} />
```

`layout="columns"` puts each label beside its text instead of over it.
