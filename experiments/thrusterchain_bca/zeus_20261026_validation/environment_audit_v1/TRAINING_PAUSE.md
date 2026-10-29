# Training package paused for environment audit

The user's later instruction, "Env itself first", takes priority. Training-package work stopped after source inspection. No training package, model update, simulator experiment or reservation was created.

Findings to preserve for eventual training work:

1. Both the pinned native initializer and BCA auxiliary initializer enforce observation statistics from their supplied data. Common normalization needs a deliberate common-intersection initialization fixture for both, followed by condition-retained replay; changing one model's statistics alone is insufficient.
2. The existing distance penalty has exactly zero action gradient at its frozen fork actor. Matching gradient RMS on logged training actions is an off-policy match and must never be described as matching the live initial actor gradient.
3. Reusing calibration episodes after their thresholds affect the BCA actor and subsequent critic is adaptive reuse. It does not establish a fresh split-conformal guarantee.
4. Three seeds x three data conditions x (10k warmup + three 10k branches) implies 360k critic updates. No expanded runtime budget, training seeds or launch approval has been granted.

These are deferred training integration issues. They are not defects in the simulator or findings of the environment audit.
