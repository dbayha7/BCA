# decision/: the decision files of 20 October 2026 (history, not pinned)

Copies of the session's working files from which version 9 was built, kept beside the package so that the verdict
table and the numbers the pre-registration quotes have a committed source. Nothing here is pinned by the code
manifest; `approve_local_near80.py` allows this folder as it allows `V9_BUILD_NOTES.md` (the only folder besides
`code/` that may hold Python files: the two calculation scripts below), and `freeze.py` does not list it. Nothing of
the near-80 code reads it.

| file | what it is |
|---|---|
| `DECISION.md` | the decision after three reviews (revised 20 October 2026); its section 3 is copied word for word into `PREREGISTRATION.md` section 14.6 |
| `REVISION_LOG.md` | what the statistics, fairness and feasibility reviews changed in it |
| `credit.md` | the credit analysis of version 8's records that sections 9 and 17 of the pre-registration quote |
| `c3_rule.py`, `c3_rule.out` | the sign-count and power calculations of C2 and C3 (section 9.3), and their output |
| `review_stats_calc.py`, `review_stats_calc.out` | the statistics review's calculations (the refusal expectations, the power numbers), and their output |
| `PREVIEW.md` | the preview of the pooled groups on the spent seeds 0-4 and 10-14 (section 3.5; nothing trained) |

The two scripts are copies for reading: they ran in the session's scratchpad against version 8's results folder at
the path they name, and their `.out` files are what they printed. Known stale line, left as it was:
`DECISION.md` section 2.2 (line 109) says "Both heights are reported on seeds 0-4"; seeds 0-4 have no second critic
on the desktop, so the heights are reported on seeds 10-14 (`PREREGISTRATION.md` 17.1, `PREVIEW.md` section 0).
`DECISION.md` section 3 has no row for the validity conditions; section 14.6 of the pre-registration adds one.
