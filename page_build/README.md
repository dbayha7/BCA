# ThrusterChain page, version 27: the build handed to the cloud

The owner's laptop battery was running out on 16 Oct 2026, so this build moved to the cloud. These are the source files of
page version 27 of the ThrusterChain artifact (https://claude.ai/code/artifact/3cadefd0-4bec-414c-b1ef-f32473e337b0),
copied from the laptop session's scratchpad (`scratchpad/wf7/...`). Paths inside the scripts assume this folder layout.

- `base/core_92ba.html`: the live page this build starts from (service version 1792156350-92ba: the page's version 25
  with its status box removed at the owner's request). `base/live_92ba.html` is the same page with the publish wrapper:
  its first 531 bytes are the wrapper head, and its last 15 bytes (`\n</body></html>`) are the wrapper tail.
- `wf7/build/core_v27.html`: version 27, built by `wf7/build/apply_all.py base/core_92ba.html OUT --version 27`. It
  applies the toy patch, then the color, navigation and lesson patches, then the integration edits.
- `wf7/build/BUILD.md`, `REBASE.md` and `FIX_LOG.md`: what the build does, every merge with versions 20-25, and review
  round 1's findings and fixes.
- `wf7/toy/`: the full results of all five toy notebooks (`TOY.md` lists the sources and the re-run checks).
- `wf7/build/tests/`: the laptop's test scripts. They drive headless Microsoft Edge on Windows, so in the cloud use
  Chromium or Playwright instead; the checks they make are the reference.

Not here: screenshots, test outputs, stage copies and the lesson's number-regeneration pipeline (laptop only). The
lesson's numbers are frozen in `wf7/lesson/lesson_data.json`.
