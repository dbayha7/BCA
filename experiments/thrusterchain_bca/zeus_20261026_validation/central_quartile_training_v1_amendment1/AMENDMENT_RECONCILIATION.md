# Amendment 1: authoritative prepared plan and exact-state repair

The authoritative prepared scientific plan is this package's `protocol.json`, `PROTOCOL.md` and `DATA_PLAN.json`, unchanged from `central_quartile_training_v1`. It is a proposal with production execution disabled, not an approved 12-hour run. The user approved reconciliation and this implementation repair on 2026-10-28 at 20:50:41 UTC. No training, runtime retry or larger workload was authorized by that approval.

## Plans reconciled without expanding scope

Retain the existing three seeds, three data conditions, three policy arms, nine separate 10k native warmups and 27 branches ending at 20k. Retain original direct-Q BCA kappa 3, matched distance control, normalization, episode roles, masks, calibration rows, optimizer/RNG state, evaluation operators, return references and stopping rules.

| Item | Authoritative prepared plan | Historical alternative in v3 decision form |
|---|---|---|
| Calibration times | Shared 10k, branch 15k | Shared 10k, branch 15k and 20k |
| Total calibrations | 9 + 27 = 36 | 9 + 27 + 27 = 63 |
| Posterior draws | 36,000 | 63,000 |
| Evaluated checkpoints | 0 / 10k / 15k / 20k | Also branch 11k |
| Evaluated policy snapshots | 72 | 99 |

The alternative adds 27 final calibrations, changing the final uncertainty widths and adjusted scores even when training updates are unchanged. Applying the current evaluator to its extra 27 snapshots would add 1,728 episodes, 10,368 forced-action Q-pi references and 820,800 simulator transitions, increasing the current cap from 1,975,304 to 2,796,104. This arithmetic illustrates the cost; it is not an adopted alternative protocol or runtime estimate. It would need a separate scientific and resource decision. No such expansion is made here.

Current totals remain 360,000 critic, 180,000 actor and 360,000 auxiliary updates; 36 calibrations; 4,608 free evaluation episodes; 1,975,304 simulator transitions; proposed global ceiling 43,200 seconds. Twelve-hour production approval remains pending, independently of target validation.

The v3 blank decision form, its 63-calibration/11k schedule, old options memo and superseded gate READMEs remain historical alternatives, not current launch instructions. Their files and commits are preserved. The v2 and zero-slab gates must not be frozen or launched; the completed central-quartile gate remains development design evidence, not BCA efficacy or joint-action support proof. The options memo's claim that better continuation must shrink action gaps is a hypothesis, not a general property; the prepared protocol already allows magnitude and sign changes.

## Exact-state comparison, not a tolerance adjustment

The second target validation stopped because `serialization.to_bytes(restored) == raw` failed. JAX can reorder mapping keys while Flax restores the template's order. A two-leaf CPU reproduction showed unequal MessagePack bytes with bit-identical state. The failed target attempt did not retain the raw/restored synthetic checkpoint, so the actual differing fields cannot be retrospectively proved. Its failure remains recorded.

`state_exact.assert_exact_state` compares the decoded saved schema against the restored Flax state dictionary. It ignores only dictionary insertion order. It requires equal key sets, empty/nested container structure, primitive and array/scalar kinds, shapes, dtypes, metadata and exact leaf bits. It detects extra fields Flax may otherwise omit during restoration. It reports the mismatch path, without numerical tolerances. Static functions/classes/optimizer definitions remain bound by the authenticated template and source hashes; serialized state does not encode them.

The same helper is used for runtime full-state restoration, production warmup restoration, and evaluation's full-state versus actor-export check. Original artifact hashes, manifest checks, state/counter validation and exact restored-next-step comparisons remain in place. The synthetic runtime report now identifies the full-state restoration stage and records successful exact-comparison diagnostics. No policy, critic, auxiliary objective, calibration rule or numerical threshold changes.

## Immutable lineage and validation attempt 3

The original package, archive of attempt 1, completed failed attempt 2 and their sources/receipts are pinned in `AMENDMENT_LINEAGE.json` and the new preparation seal. They are not moved, overwritten, reclassified as passes or reset. The amendment is a separate directory with no copied reservation or run output.

The new supervisor retains the reviewed interruption-safe behavior: record-only signal handlers through child creation, cleanup and terminal receipt writes; child outcome saved before final verification; exclusive validation reservation; scoped same-user process checks including both old and new validators; unknown child liveness if cleanup fails. It never reserves the production attempt or automatically retries. Original legacy supervisor tests remain inherited evidence; `launch.py --validate-runtime` routes to `validation_supervisor.py`, whose dedicated fixtures cover the current path.

Attempt 3 has the same 900-second worker/postcheck cap. Owned-child cleanup may take up to 10 additional seconds plus filesystem latency; read-only prechecks precede reservation. Unchanged synthetic caps are 64 critic / 32 actor / 64 auxiliary updates and 4 calibrations; read-only initialization cap is 4,480 repeated episode reads / 448,000 source tuples; Gymnasium cap is 512 transitions / 128 resets. There are zero real-data optimizer updates. CUDA/model restoration, next-step continuation, source initialization and Gymnasium must actually complete before this validation can pass.

Prior work is not erased: attempt 1 has unknown final counts and a 19.75672-second observed elapsed lower bound. Attempt 2 used 8 critic, 4 actor, 4 auxiliary updates and 1 calibration, entirely synthetic, in 62.324150886 seconds including postcheck. Their cumulative elapsed lower bound is 82.080870886 seconds, but the exact cumulative total remains unknown. A successful attempt 3 cannot make prior accounting exact.

CPU tests and a new seal prepare a manual runtime validation; they do not satisfy production launch conditions. Assistant-side WSL access remains denied and is not retried. No installation or push is part of this amendment.
