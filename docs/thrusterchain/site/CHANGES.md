# Change summary, 27 October 2026

## What was built

- One canonical, self-contained research interface: `docs/thrusterchain/index.html` (about 0.95 MB; no network
  access, no external fonts or scripts; works from the file system). Six parts: Start here, Environment explorer,
  BCA explained, Experimental setup, Results, Evidence and downloads. Each part has a section sidebar with clickable
  subsections, previous/next controls, a one-part-at-a-time view with a show-all toggle, light/dark themes, and a
  phone layout with a bottom bar.
- A validated results manifest (`site/manifest.json`, resolved copy `site/data/manifest_resolved.json`): 24 entries
  with status, era, machine, dates, question, plain summary, sources, and 98 headline numbers that the build resolves
  from JSON paths or verbatim text in the cited report files. Missing evidence is marked unavailable, never zero.
- Extracted data: `site/data/environment.json` (contract, matrices, fixtures, five playback trajectories; formula
  validated against all fixtures with zero error) and `site/data/results.json` (128 states × 9 candidates with
  reference V, Q1, Q2, U, commands; official selector tables for both batches; matched-pilot returns and paired
  comparisons; cached privileged controls; operator-alignment strata). Fourteen recomputed aggregates were checked
  against the diagnostics before writing.
- `site/BCA_AUDIT.md`: source-anchored audit of the Unifloral host and every BCA component with line numbers, the
  update order, the receipts, and a claim-to-evidence table.
- `site/SOURCE_MAP.md`: where every section came from, what was preserved untouched, what is absent.
- Four project skills installed under `.claude/skills/` from the user's ZIP (no existing skill overwritten).

## What changed in scope during the task

- The user instructed "Unifloral base only, not CORL". The existing main page (version 16), the v3–v7 diagnostics
  pages, the 32-state saved-bank Explorer, the critic viewer and the BCA mechanics page all report the CORL-derived
  host, so they were preserved on disk unchanged and listed as archived rather than merged. The Environment Explorer
  was therefore rebuilt from the core v4 benchmark contract instead of carving the old page.
- The brief's five Titan-only evidence folders are not present on this machine (Zeus). Their rows show the user's
  description as a report and no numbers; the support-pilot completion is likewise marked unavailable here.

## Verification performed

- Build checks (all passing): 98 manifest numbers resolved; 32 prose placeholders filled; 14 kappa-grid
  cross-checks matched; 10 fixture cases and 505 trajectory steps reproduced with error 0.0; 113 unique ids; 11
  fragment links all resolve; no external references; size below the 16 MB artifact limit.
- Rendered in the built-in browser over a local static server at 1280 px and at the 375 px phone preset: top tabs,
  section sidebar, bottom bar, episode player (Play/pause/step), one-step worked arithmetic, stall and start-law
  figures, state explorer with strength buttons and case shortcuts, error and decision charts, policy-return chart,
  evidence table and status board. No horizontal overflow at either width after the fixes. No console errors on a
  fresh load (the only error seen was from an earlier build and was fixed).
- Independent verification by a multi-agent workflow (run wf_bfc30cfb-9e0): each of the 20 local evidence folders was
  extracted into a structured entry by one agent and re-checked by another against the source files (about 850 number
  checks); no number used by this interface was refuted, and the two status labels the verifiers questioned concern the
  agents' own drafts, not this manifest. The Unifloral host description was checked line by line (67 citations, none
  refuted). Verification of the auxiliary and penalty components and the map of the old page were still running when
  this summary was written; their findings, if any, arrive as a later note.
- Direct file:// opening could not be exercised by the available browser tools (both block the file scheme); the
  page's self-containment was confirmed by the build's external-reference check and by observing that loading it
  issues a single network request.

## Not done, by instruction

No training, reruns, frozen-protocol changes, deletions, publishing, pushing or permission changes. Nothing was
committed; all new files are untracked under `docs/thrusterchain/site/`, `docs/thrusterchain/index.html`,
`.claude/skills/` and `.claude/launch.json`.
