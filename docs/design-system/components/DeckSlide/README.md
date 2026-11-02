# DeckSlide

A 1920 by 1080 slide in the deck's own look: navy slides for the cover, parts and questions, paper slides for content, a code panel for code.

**Use for** talks and decks about the project. Slides and the figures on them share one palette (the `deck-` colors, which do not change with the theme) and one type system (IBM Plex Sans, Source Serif 4 for titles, JetBrains Mono for code).

**Navy slide** (cover, part, question): `deck-ink` ground, 128px padding, content centered vertically.
- Kicker: `dk-kicker` (24px/600, 3px tracking, uppercase) in `deck-kicker`, words joined by " · ".
- Title: `dk-cover` (104px) or `dk-title` (76px) in `deck-on-ink`; lead `dk-lead` or `dk-body` in `deck-on-ink-2`.
- Question slides open with a row of seven 56px dots: the current one filled `deck-step` with its number in `deck-ink`, the others outlined 2px `deck-ring` with numbers in `deck-step-off`.
- The cover adds a 160 by 6px rule in `deck-orange` under the title and the byline at the foot.

**Paper slide** (content): `deck-paper` ground, padding 128px (160px at the bottom).
- Kicker in `deck-blue`; title in Source Serif 4 52px/1.12 in `deck-ink`; body 24–30px in `deck-ink` or `deck-soft`.
- What is not yet shown, in `deck-caution` 26px/500. Footnotes 24px in `deck-footnote`, 64px from the bottom.
- Boxes on `deck-card`; a highlighted table row on `deck-row`.

**Code panel**: `deck-code`, `radius-deck`, 16px by 28px; code in `dk-code` `deck-code-ink`, strings `deck-code-string`, numbers `deck-step`; the file-and-lines caption above it in mono `deck-step-off`.

**Rules**: nothing under 24px; one idea per slide; the slide title carries the figure's title, so figures have none.
