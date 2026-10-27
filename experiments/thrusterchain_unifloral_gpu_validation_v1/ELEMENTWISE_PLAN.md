# Elementwise CPU/GPU update check: predeclared protocol

This additive check uses the same source and ten-episode smoke dataset as the
bounded validation harness. The final CPU reference's initial checkpoint is
SHA-verified and restored by both devices. Each device performs exactly one
unified outer update on fixed training rows `[0, 1, 99, 102, 199]`: one actor
update and two critic updates. The resulting actor, critic, actor target and
critic target parameter leaves and actor/critic optimizer leaves are compared
elementwise after validating tree structure, names, dtypes, shapes and finite
values. Floating-point tolerances are **absolute 1e-4 and relative 0.005**, fixed
before running the final CPU/GPU check. Integer, boolean and PRNG leaves must
match exactly. A mismatch fails; tolerances are not tuned after GPU results.

This is a bounded fixed-batch comparison with common initial parameters and RNG.
It does not prove equivalent training trajectories, scientifically meaningful
policy performance, or general correctness across other hardware or configs.
