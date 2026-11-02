# Bounded host checks

## Strict-offline TD3+BC runner adapter

`td3bc_adapter.TD3BCAdapter` connects the repository-native
`algorithms/td3_bc.py` to the inherited v5 `train_offline` and `run_mode` interfaces. It is
not the pinned UniFloral standalone or unified implementation. Its configuration
is fixed: gamma .98, raw rewards, the host's default actor/critic architecture,
noise, alpha, Adam, delayed actor and target schedule. No setting overrides or
online updates are supported. The adapter fits `(mean, std + .001)` using the
runner's training split only, applies that fixed transform to current and next
observations, and sends executed actions and explicit termination flags to the
host. Truncation alone does not zero the bootstrap.

Every primary update is one critic step; actor and target steps follow the
host's delay. Checkpoints bind optimizer, target, actor/critic, noise RNG, and
preprocessing state. Their metadata identifies this adapter, native host,
shared network/reward sources, resolved arguments, and Python package versions.
The full 64-bit initialization seed is encoded in the JAX key's high and low
words before the host's fold-in; for seeds below 2**32 this matches native
initialization exactly. The runner's full generated 63-bit seed is retained.
Restoration rejects changed source hashes or package versions. The enclosing
runner records the dataset and train-split hashes and sampling RNG. Frozen
inference contains only the actor and fixed preprocessing. The evaluation guard
is cooperative instrumentation, as documented by the runner.

The bounded test checks real dataset updates, exact uninterrupted versus resumed
checkpoint bytes, save/reload before one frozen evaluation episode, inference
against the host, source-identity rejection, distinct terminal and truncation
targets, and nonfinite update refusal. These checks do not establish scientific
performance, GPU portability, arbitrary-runtime compatibility, or broad learner
equivalence. Run only bounded checks before a separately authorized scientific
experiment.

## Strict-offline ReBRAC runner adapter

`rebrac_adapter.ReBRACAdapter` connects the repository-native ReBRAC host to the
same runner with fixed settings, train-only normalization and gamma .98. It uses
the host's zero-based schedule: the actor updates on critic updates 1, 3, and so
on. It accepts a terminal row without a next action because the entire successor
term is masked, but refuses initialization if any bootstrapped row lacks its
recorded next action. Consequently the current continuing-time-limit datasets
are explicitly unsupported by this adapter rather than silently dropping rows
or changing its objective. No online updates are implemented.

The bounded tests cover native low-seed initialization parity, full-width seed
handling, an independently calculated critic target, counters, exact offline
resume, and a reloaded frozen evaluation. These establish only those tested
paths, not full experimental results or external ReBRAC equivalence.

## Strict-offline IQL runner adapter

`iql_adapter.IQLAdapter` uses the repository-native Gaussian IQL variant. It
fits V against the minimum target critic on logged actions, uses the host's
pre-update value model for its critic target and advantage weights, and uses
capped beta-multiplied detached weights on the logged-action Gaussian likelihood.
The host updates V, critic, target critic, then actor. Frozen evaluation uses
the deterministic Gaussian mean. Raw rewards, gamma .98 and train-only
observation normalization are fixed. The adapter does not implement online
updates. Bounded checks cover independently reduced losses, terminal/timeout
targets, update counters, full-width seeds, exact resume and frozen evaluation.

## Strict-offline CQL runner adapter

`cql_adapter.CQLAdapter` preserves the repository-native twin-critic CQL
implementation, including its importance-sampled conservative gap, ten
candidate actions, coefficient 10, temperature 1, automatic entropy tuning,
no entropy backup and no Lagrange mode. Its fixed `bc_steps=0` bypasses the
host's non-dataset-action warm-up branch. Gamma is .98, rewards are raw,
normalization is fit on train observations, and frozen inference uses the
deterministic tanh of the Gaussian mean. The host increments both critic,
actor, entropy and disabled alpha-prime optimizer counters on each step. The
adapter provides no online updates. Bounded tests cover terminal/time-limit
targets, low-seed native parity, full-width seeds, exact resume, source
identity and frozen evaluation. These checks do not reproduce another CQL
implementation or establish performance.

## Task-batch numerical probes

`td3bc_probe.py` bridges actual v5 train transitions to the repository's
`algorithms/td3_bc.py` and performs exactly two native TD3+BC updates. It
checks critic losses and delayed actor loss against independently reduced
network outputs, and verifies delayed optimizer and target counters. Train-only
observation normalization is applied to both current and successor fields;
rewards and executed actions are unchanged. This is a task-batch plumbing and
numerical check, not a production adapter, checkpoint, performance result, or
validation of other algorithms. No policy selection or rollout occurs.

`other_probes.py` also checks real repository-native IQL, CQL, and ReBRAC
updates on actual v5 training batches. IQL's target-critic expectile, value
bootstrap, and weighted Gaussian likelihood are checked against NumPy
reductions. CQL's TD errors and density-corrected conservative gaps are checked
using the same sampled proposals and independently reduced network outputs.
The v5 CQL probe explicitly requires `bc_steps=0`: the repository host's
optional warmup actor loss scores a second policy sample instead of the
dataset action. A numerical counterexample confirms its warmup policy loss
is unchanged when logged actions change. This guard applies to this probe;
other callers of the old host are not automatically protected.
ReBRAC's target uses the true same-episode recorded next action, and the check
refuses a missing next action on any row with nonzero bootstrap. A terminal row
may use a finite placeholder only because the entire successor expression is
multiplied by zero; its reward-only target is unchanged. Each probe checks
optimizer counts and identifies the source variant. None is a production
adapter, an independent implementation of the networks or action sampler, or
evidence of algorithm performance.

These repository-native hosts are not the separately pinned UniFloral host.
V6 adds one explicitly resolved unified composition with actor-primary budget
accounting and fixed-batch parity. See [unified integration](unifloral_README.md)
for its different cadence, actual dependencies, compatibility opt-in and limits.

Run on a compatible JAX/Flax/Optax host with the benchmark and repository on
`PYTHONPATH`, and Gymnasium installed. From this v6 directory in the existing
WSL runtime, for example:

```
PYTHONPATH=../..:.:/mnt/c/Users/David/.cache/bca-shift/py \
  JAX_PLATFORMS=cpu /opt/bca-shift-venv/bin/python -B -m unittest \
  host_integration.test_td3bc_probe host_integration.test_other_probes \
  host_integration.test_td3bc_adapter host_integration.test_rebrac_adapter \
  host_integration.test_iql_adapter host_integration.test_cql_adapter -v
```
