# OOD adapter engineering checks — September 27, 2026

The initial **TD3+BC/ReBRAC × Hopper/Walker** adapter tests pass on the existing
WSL CPU runtime. This establishes that the measurement tools agree with the
original host calculations and can repeat a simulator transition from a saved
state. It does not establish that BCA detects harmful actions or improves return.
All 40 required scientific 1M checkpoints remain pending.

Validation: **6 adapter/oracle tests + 19 protocol tests passed**. The repository
check also passes, including all 40 bundled Unifloral source files and 47 traced
BCA integration snippets. The 280-run training declaration matrix is unchanged.

Every setting and section in [ood.yaml](../configs/ood.yaml) has an explanatory
comment. Its parsed values are unchanged: semantic SHA256
`491e5662bfac93bbe58d0318ab133f98c4a93bfedc4ae9f898c59f1c6fb2e35e`.
Collection-specific `pending` fields deliberately remain pending; the tests do
not change the scientific protocol or accept its future artifacts.

## What was checked

| Check | Result and meaning |
| --- | --- |
| Four checkpoint paths | TD3+BC and ReBRAC, each host-only and BCA, reproduce original actor, target, width and dose outputs. Fixtures have zero optimizer updates. |
| Independent initialization anchor | All four initialized state hashes match the archived CPU host parity fixtures. Direct reference outputs are saved before adapter loading. |
| Frozen BCA queries | Querying actions, targets and widths leaves serialized model state and checkpoint bytes unchanged. Native width is N/A. A synthetic radius refresh makes the BCA fixture ready; no scale fitting occurs. |
| Target arithmetic | Rewards, terminal masking, target smoothing, twin-critic minimum and ReBRAC's recorded-next-action penalty agree with the original host. Missing ReBRAC next actions are refused. |
| Invalid checkpoint inputs | Wrong file hashes, missing parameters, wrong update counts and malformed RNG shape/dtype are refused. Step-zero fixtures cannot pass the regular 1M budget requirement. |
| Saved-action gate | Actual/reference/error/device arrays are written before absolute `1e-6` is checked, including deliberate failing and wrong-shape fixtures. Existing files cannot be overwritten. |
| Full simulator state | Removing each required field or changing tested shapes/dtypes/identity is refused. Snapshot restoration and repeated observations match exactly, including a non-reset state. |
| Action and reward arithmetic | Float32/float64 proposed actions, transformed applied actions and simulator controls are retained. Rewards reconstruct from measured displacement and independently squared applied actions under absolute `1e-7`. |
| Episode endings | Natural termination and the original 1,000-step time limit are distinguished. Stepping after either ending or past the engineering budget is refused. |
| Shared stochastic streams | Per-step keys reconstruct exactly; a changed key bank is refused. Candidate actions reuse the same bank. This tests the stream contract, not an unimplemented sampled-policy adapter. |
| Exact toy oracle | Familiar action `-1` loses 9.1 return units relative to action `0`; unfamiliar action `+1` gains 2.9. Familiarity and harm are separate. Constant/reversed/oracle score fixtures are prepared; ranking metrics remain Task C. |

The accepted Hopper maximum reward error is **2.920619301960414e-11**; Walker's
is **2.192258596878105e-10**. Both are below the unchanged `1e-7` gate.
These small hand-selected engineering fixtures are not a broad physics audit.

## Preserved execution attempts

Each attempt has a unique directory and actual process-exit receipt under the
ignored local `runs/ood/` tree. Failures have not been overwritten or reclassified.

| Attempt | Actual exit | Observation |
| --- | --- | --- |
| `adapters-runtime-v1` | 1 | Four checkpoint paths passed; simulator setup found the additional Gym `OrderEnforcing` wrapper. Two constructor transitions, no explicit test transitions. |
| `adapters-runtime-v2` | 1 | State validation encountered inactive MuJoCo fields represented by `None`. Two constructor transitions, no explicit test transitions. |
| `adapters-runtime-v3` | 1 | The installed legacy Gym RNG could not be copied through its NumPy reducer. Two constructor transitions, no explicit test transitions. |
| `adapters-runtime-v4` | 0 | Simulator checks passed after validating RNG state with a temporary native NumPy generator. Twenty explicit and two constructor transitions. |
| `adapters-runtime-v5` | 0 | Full six-test suite passed, including stronger malformed-input tests. Twenty explicit and two constructor transitions. |
| `adapters-runtime-v6` | 0 | Final six-test suite passed after moving the control-equality gate after evidence writes, so a future failure retains the actual control vector. Twenty explicit and two constructor transitions. |

Across all six attempts: **60 explicit + 12 constructor = 72 environment
transitions, 288 physics steps**, within the predeclared 130-transition ceiling.
No optimizer updates or scientific policy outcomes were produced. The compact
[verification receipt](../experiments/ood/validation.json) records source/evidence
hashes, test exits, counters and model identities.

Corrections account for actual wrapper state, preserve inactive fields as `None`,
and restore the exact saved RNG state. They do not alter host equations, rewards,
gates or training settings. Source hashes are saved for every attempt; exact OOD
source copies are additionally saved from v5 onward. The original runtime's
`get_mjb` temporary-file warnings remain in logs; upstream source was not edited.
Fixture checkpoints and numerical arrays stay local, outside Git.

## Reproduce the engineering checks

The protocol-only suite needs Python and PyYAML:

```bash
python -m unittest experiments.ood.test_protocol -v
python check.py
```

Checkpoint/simulator tests require the existing Linux/WSL JAX/Flax and
D4RL/MuJoCo 2.1 runtime. The adapter checks exact installed Python source hashes
and records its binary model hash; an incompatible installation is refused.
A fresh installation or GPU replay has not been verified. In that environment,
from this repository:

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Use a NEW output path for every attempt. This creates engineering evidence.
mkdir -p runs/ood
set -C # Refuse to overwrite an existing log or exit receipt.
export BCA_OOD_TEST_OUTPUT=runs/ood/my-adapter-check-01
python -m unittest experiments.ood.test_outcomes -v > runs/ood/my-adapter-check-01.log 2>&1
adapter_exit=$?
printf '{"exit_code":%s,"kind":"engineering_tests"}\n' "$adapter_exit" > runs/ood/my-adapter-check-01-exit.json
exit "$adapter_exit"
```

The test module also chooses a unique output directory if the environment
variable is omitted. It declares a ceiling of 130 engineering transitions,
including constructor steps, before loading a model or creating a simulator.
A passing full suite currently uses 22 transitions (88 physics steps). Repeated
attempts must be charged cumulatively to the approved engineering allowance.
No dataset is downloaded, optimizer updated or GPU worker started.

## Remaining work

1. Implement and verify harm-ranking, coverage and clustered uncertainty metrics;
   render complete/missing/failed/sparse synthetic reports and check their figures.
2. Bind the actual 1M checkpoints, training journals, paired data/normalization,
   saved action references and real process exits before accepting research rows.
3. Freeze actual collection candidates, state identities, random keys and the
   cumulative resource ledger; pass the collection-specific engineering gates.
4. Resolve ReBRAC's fresh live coverage target separately. Its training target
   requires a recorded next action, which a new live transition does not supply
   under the same offline contract. The adapter never substitutes actor actions.

IQL, CQL and Unifloral reference adapters are outside this first tranche. No old
native outcome, failed recovery, training queue or main thesis PDF was changed.
