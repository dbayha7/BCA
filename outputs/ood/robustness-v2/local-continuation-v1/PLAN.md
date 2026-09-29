# Local Hopper continuation implementation plan

**Goal:** Continue the interrupted TD3+BC Hopper171 pair from its exact saved state, without repeating completed trajectories or adding a repeat check.

**Architecture:** A new one-shot worker holds the original local lease and reopens the existing cumulative extension through its unchanged `ExtensionLedger.open` interface. A small adapter uses the existing framed journal for new evidence, with the original SQLite extension as the sole accounting authority. The old archive remains read-only; a selected, hash-bound input capsule carries the already reviewed states, banks, completed returns and partial trajectory.

**Tech stack:** Python, existing NumPy/JAX/MuJoCo runtime, original frozen scientific interfaces, existing framed archive and SQLite extension.

The user's explicit no-subagent and immediate-execution directions take precedence over the planning skill's delegation and handoff suggestions. All implementation files go in `work/standard_bca_noiw_campaign_v1/monitor_20260929T094838Z`; running and closed sources remain unchanged.

- [ ] Prepare a bounded capsule from the terminal attempt, binding actual exit1, the closed full-prefix review, old archive identity, original ledger snapshot, source/checkpoint/training/support receipts and existing precommit. Preserve a byte-for-byte stopped ledger snapshot before reopening the original.
- [ ] Implement `continuation_storage.py`: permanent claim under the original pair; original shared lease; exact local TD3 scope; cumulative original phase/global caps; framed reserve and ack mirrors explicitly subordinate to the original extension. Failures retain charges and consume the claim.
- [ ] Implement `continue_panel.py`: reuse 740 completed returns and all52 panel files; continue only host/state026/slot12 from ordinary step247, retaining its247 rewards and already verified first/repeat. Execute untouched slots with the original driver. No collection, candidate generation, key generation or completed trajectory replay.
- [ ] Test new resume equivalence, duplicate/scope/cap refusal, lease exclusion and crash accounting with synthetic inputs. Preserve every executed source snapshot and actual exit. No closed suite rerun.
- [ ] Independently review the concrete route and input bindings before dispatch. Charge a new constructor and two paired engineering checks to the original engineering allowance; independently read the new journal's engineering prefix before release to outcomes. Require action1e-6, reward1e-7, exact full record and full-state repeat, plus exact restoration of the interrupted full state.
- [ ] Launch once under a saved supervisor/command/start identity. Retain original failure and cumulative charges, then observe real continuation progress without opening its active SQLite ledger. Publish small owned sources and receipts after integrity checks. Final pair acceptance remains pending complete independent outcome review.

Interrupted trajectory disposition: the original verified first/repeat and247 ordinary records form a contiguous prefix. New records must begin at247 from the exact saved full after-state, with the same frozen host actor. The first/repeat is never repeated in the repeat phase. Its completed return is accepted only after a saved-data reviewer joins both attempt segments and unchanged keys. New engineering is separate, explicitly charged, and does not replace the old repeat evidence.
