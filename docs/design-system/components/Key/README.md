# Key

A small CSS mark in front of a word: the shape says what kind of thing it is, the color says which one.

**Use for** legends, option names, table row headers and the first mention of a thing in running text. A color never appears without its label, so a key always sits beside its word.

| shape | class | means |
|---|---|---|
| dot | `.key.dot` | a test situation |
| line | `.key.line` | a world, a controller or a learned part |
| dashed line | `.key.dash` | no throttle |
| dotted line | `.key.dots` | the throttle cost |
| tile | `.key.tile` | a region (an area) |
| hatched tile | `.key.removed` | data removed |
| ring | `.key.ring` | a float, or no signal |
| four-quarter dot | `.key.sitall` | the situation you picked (all four) |

**Markup** (CSS only; the React component renders the same)

```html
<i class="key dot" data-sit="big_waves"></i>Big waves
<i class="key removed" data-k="region"></i>Removed from the data
<span class="km" data-sit="strong_current">strong current</span>   <!-- in running text -->
```

- Color comes from one attribute: `data-sit` (`recorded`, `strong_current`, `big_waves`, `big_kicks`), `data-world` (`stall`, `no_stall`) or `data-k` (`region`, `stall`, `learn`, `hold`, `recording`, `stall_aware`, `rising_thrust`, `no_throttle`). It sets `--c` and `--c-soft` for anything inside.
- `.km` is the running-text form: the word stays in `ink` and the mark sits in the span's own padding, so a line break never separates them. Use it at a first mention, in lead-ins and in captions, not on every repetition.
- A gold tile always carries a 1px ring in `k-region-ink`.
- Group keys in a `.legend` row (14px, `ink-2`).

**React** (`window.BCA`, React 18; hand-written from the page's markup in `page_build/wf7/build/core_v27.html`)

```js
const { Key, Mention, Legend } = window.BCA;
<Key shape="dot" sit="big_waves" />Big waves
<Mention sit="strong_current">strong current</Mention>          // in running text
<Legend items={[{ shape: 'line', world: 'stall', label: 'Stalling world' },
                { swatch: 'dr-lo', label: 'untargeted penalty' }, { ring: true, label: 'no signal' }]} />
```

Give one of `sit`, `world` or `k`. A `Legend` item is a `Key`'s props plus `label`, or a `swatch`, `ring` or `line` (a color token's name) for chart marks.
