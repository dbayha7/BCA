# V5 bounded integration validation plan

This is a plumbing check, not a scientific study. Generate a fresh smoke dataset
of 10 episodes with the default finite task and seed 20261022. Run the existing
`run_training_example.py` with its fixed affine-SGD configuration: eight offline
optimizer steps; three more on a resumed copy compared with an independent
11-step continuation; and 12 online interactions plus six online updates per
scenario for the four scenario families. The evaluation config runs two frozen
episodes per family for each of the strict, adaptive and zero-online modes.

Expected observations: the 8+3 and 11-step checkpoint bytes agree; the dataset
and source snapshot remain unchanged; strict evaluation performs no optimizer
updates; adaptation records 48 interactions and 24 primary updates total;
the zero-online mode records no adaptation; evaluation records complete
100-step episodes and stable policy state. A miss is a failed validation, not
grounds to change the expected outcome after seeing it. No neural algorithm
performance or cross-method conclusion is inferred.
