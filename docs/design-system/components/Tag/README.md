# Tag

One or two lowercase words of status, outlined in the current text color, beside the thing they qualify.

**Use for** the state of a test or a claim: `pass`, `exploratory`, `never ran`, `continue`. In a status list (`ul.g-status`) each line starts with a tag and says the rest in a sentence.

**Markup** (CSS only; the React component renders the same)

```html
<span class="tag ok">pass</span>
<span class="tag acc">exploratory</span>
<span class="tag warn">stalls</span>
<span class="tag">never ran</span>

<ul class="g-status">
  <li><span class="tag ok">pass</span>The environment checks: all nine world checks hold.</li>
</ul>
```

- `.tag.ok` in `ok`, `.tag.acc` in `accent-ink`, `.tag.warn` in `warn`; a plain `.tag` takes the text color.
- 12px/600, 0.04em tracking, `radius-pill`, a 1px border in `currentColor`.
- In `ul.g-status` every tag is ink except `.ok`, and each line has a 3px `rule-strong` left edge.

**Do** keep red (`.warn`) for the stall and real warnings. A test that stopped or failed is a decision, so its tag stays neutral and its words say why. **Don't** fill tags or use them as buttons.

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { Tag, StatusList } = window.BCA;
<Tag variant="ok">pass</Tag>  <Tag variant="acc">exploratory</Tag>  <Tag>never ran</Tag>
<StatusList items={[{ tag: 'pass', variant: 'ok', text: 'The environment: all nine world checks hold.' }]} />
```

`variant` is `ok`, `acc`, `warn` or none (the text color). In a `StatusList` every tag is ink except `ok`.
