# Prospective CQL full-network gradient check

This bounded test supplements, and does not replace, either the independent toy
objective oracle or the original strict eager-versus-compiled parity gate.

Use the actual CQL adapter and default training configuration. Generate ten
feedback-weak episodes with seed 2026102703, initialize the learner with seed
2026102803, and select replay rows `[11, 29, 70, 99, 145, 199, 225, 299]`.
Use a fixed update key from the adapter's normal split. In a single compiled
host update, extract the Adam first moment and derive its first-step gradients.
The fixed coordinates for both actor and critic1 Dense_0 kernels are (1,2),
(3,4), (5,6), (7,8), (9,10); for Dense_1 and Dense_2 use (1,2), (23,34),
(45,56), (67,78), (89,100). For actor Dense_3 use (1,0), (23,1), (45,2),
(67,3), (89,4); for critic1 Dense_3 use (1,0), (23,0), (45,0), (67,0),
(89,0). Perturb each coordinate in the initial state by +/-0.001 and evaluate
the same compiled update and keys. Require at least one gradient in each kernel
to have magnitude above 1e-5; record zeros rather than assuming all coordinates
are active through ReLUs.
Central finite differences compare the reported policy loss for actor kernels
and `qf1_loss + cql_min_qf1_loss` for critic1 kernels to the derived gradients.
Use atol 0.003 and rtol 0.03, reflecting float32 loss subtraction; no observed
values determine these thresholds. Record the selected coordinates, all
gradients and errors. Confirm keys, terminal rows and counters separately in the
existing test suite. Such a local derivative check does not prove long-run
training equivalence or justify calling the strict parity gate passed.
