# Prospective CQL objective check

This independent, bounded test is additional evidence for a separately named
compiled execution path. It does not change or pass the existing strict
eager-versus-compiled state tolerance.

Freeze the following before running the test: two observations, two-dimensional
actions, one nonterminal and one terminal row, a linear Gaussian actor, two
distinct linear critics and two distinct target critics; fixed PRNG key 319 and
three proposals from each proposal family. Use the repository CQL defaults for
temperature, alpha, importance correction and target switches, with learning
rate 0.01 and plain SGD solely to expose gradients. The fixture is explicitly
toy-sized and uses no training data or model selection.

Reconstruct in NumPy float64, without calling the CQL objective, the sampled
tanh-Gaussian actions and log densities, min-target-Q Bellman targets,
per-critic TD mean-square errors, proposal-density-corrected logsumexp gaps,
and SAC actor loss. Use float64 central finite differences with step 1e-5 to
check all actor, critic1 and critic2 parameter gradients, compare the compiled
one-step SGD updates to those gradients, and check updated-critic Polyak target
updates and integer counters. Use absolute tolerance 2e-5 and relative 2e-5
for loss/target metrics, absolute 2e-4 and relative 5e-4 for gradients and
Polyak parameters; require finite values throughout. The tolerance is fixed for
this analytical, differently scoped test; it does not supersede the prior
state-parity tolerance. Record all comparison errors and device identity.
