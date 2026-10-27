# Unified UniFloral TD3+BC composition in v6

`unifloral_adapter.UnifiedTD3BCAdapter` uses the computational definitions from
the actual pinned `baselines/unifloral/algorithms/unifloral.py`, revision
`f2dc1278eae18ed3e22c92255119c54885414d17`, source SHA-256
`14a9bcdbc6d96fad85544a9d98957e98a7e46db13b89d74d54e3c5ce246517c8`.
The loader checks that hash and compiles selected upstream AST nodes unchanged;
it omits unrelated D4RL, Gym, CLI and logging imports. It does not modify the
upstream source or use a stand-in for Distrax. A local object maps the removed
`jax.tree_map` name to `jax.tree_util.tree_map` without changing the JAX module.

This is **one fixed unified composition**, based on the pinned TD3+BC preset:
deterministic tanh actor, two critics, actor BC coefficient 1, actor Q
coefficient 2.5, first-critic actor Q with detached mean-absolute-Q
normalization, action-summed BC, target smoothing, no value/AWR/entropy/critic
BC/diversity, and two critic updates on one batch after the actor update. Both
targets update after the critics. The task adaptation is raw v6 rewards,
discount 0.98, executed actions, terminal-only bootstrap mask, and native
train-only observation mean/std with 0.001 added inside the model.

One runner primary update is **one actor outer step**, not one critic step:
each advances the actor once and the critic optimizer twice. This differs from
the primary-critic convention of the standalone host adapters. The runner
samples one batch with its own checkpointed PCG64 stream; the internal native
sampler is replaced by an identity boundary. The native split of the learner
JAX key, the shared batch, actor-first order, target noise and target timing are
retained. This preserves a native update conditioned on that batch, but does
not claim identical sampler draws or a full training trajectory to upstream.
The runner records its actual batch size; the adapter's stored `args` are its
architecture/configuration template.

The pinned project's requirements specify JAX `<0.6` and NumPy `<2`. The local
bounded test used actual Distrax 0.1.5 and TensorFlow Probability 0.25.0 with
JAX 0.6.2, NumPy 2.2.6, Flax 0.10.7 and Optax 0.2.8; it is outside those upstream
constraints. `learning_settings={"allow_outside_upstream_constraints": True}`
is required for such a runtime and is retained in the checkpoint. The adapter
checks only the JAX and NumPy upper bounds; its flag is not a certificate that
the entire upstream dependency stack is supported. The adapter
records package versions and source hashes and refuses restoration on a version
or source mismatch. The tests establish only those CPU paths and do not imply
upstream support, GPU validation or full-run performance.

The integration test checks fixed-batch parity against the original native
sampler after reconstructing its selected indices, including all states,
metrics and RNG; separately checks the actor loss and an independently
calculated critic target with distinct heads, true termination and a time-limit
bootstrap; and checks optimizer counts, exact resume and frozen evaluation.
The selected test runtime on Zeus requires real Distrax available on Python's
path; `/opt/bca-shift-venv` does not have it installed. For the bounded local
check, the second existing environment's site-packages was appended **after**
the primary environment, preserving NumPy 2.2.6:

```text
cd /mnt/c/Users/David/dev/BCA/experiments/thrusterchain_benchmark_v6
PYTHONPATH=../..:.:/mnt/c/Users/David/.cache/bca-shift/py JAX_PLATFORMS=cpu \
  /opt/bca-shift-venv/bin/python -B -c 'import sys,unittest; sys.path.append("/opt/bca-venv/lib/python3.10/site-packages"); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName("host_integration.test_unifloral_adapter")); sys.exit(not r.wasSuccessful())'
```

This is not a general adapter for arbitrary unified configurations, canonical
IQL, unified model-based learning, CQL, or online adaptation. No full training,
scientific comparison, or held-out performance claim is included.
