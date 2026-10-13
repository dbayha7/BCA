# Deck tools: WBCP_Experiments_Eight_Studies

The scripts that turned the uploaded deck (WBCP_Experiments_Seven_Studies.pptx, not stored here) into
WBCP_Experiments_Eight_Studies_notes.pptx on 13 October (the finished deck was sent directly and is not stored here).

- `build.py`: the builder (clones template slides, adds the Study 6-8 slides, edits and reorders the rest); helpers
  in `extra_slides.py`, `splits.py`, `scorecard.py`, `tables4.py`, `patch4.py`.
- `charts.py`: the matplotlib charts the builder places.
- `notes_work/`: `slides.json` (a text dump of the slides), `mine.json` (speaker notes keyed by slide title) and
  `apply.py` (merges the notes into the deck).
- `dump.py`, `outline.py`, `detail.py`, `runs.py`: small inspection helpers.

Paths inside the scripts point at the cloud container's scratchpad.
