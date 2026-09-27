# Standard BCA, no importance weighting

**Current priority: [action-level OOD tests](../ood/README.md).** Existing training
performance is supporting context; accepted checkpoint pairs can enter OOD gates
without waiting for the full training matrix.

The first stage is four hosts × seven datasets × five seeds × host/BCA:
280 physical runs and 315 actor trajectories. IQL's primary host/BCA pair shares
Q/V; its standalone host is additional context, not another independent seed.
Each run has 1,000,000 host updates; delayed TD3+BC/ReBRAC actors have 500,000.
Both Bayesian and conformal components remain active in BCA.

`manifest.json.gz` losslessly stores every resolved configuration, data hash,
seed bank, source hash, expected counter and fixed lane/order. Decompress it
with `gzip -dk manifest.json.gz`. Its uncompressed SHA256 is
`13ae3e6693d1cc71228e7b676243232c52ac6766c6a2c9b0623aaac1145f97fe`.
The repository contains the exact source files named in that manifest; verify
their SHA256 identities before reproduction. Model weights are excluded.

The local lane contains TD3+BC/CQL; the cluster lane contains ReBRAC/IQL.
Both methods and all five seeds for a host/dataset stay on the same lane.
`../standard_runner.py` validates the frozen source and declarations, launches
one worker at a time with an exclusive GPU lock, records actual child exits,
and stops on failure. A started output directory cannot be resumed/retried.
At most one controller per lane and two GPUs total are authorized.

Run a single row with the root `train.py` commands in the main README.
For a whole lane, use `python experiments/standard_runner.py --help`; the
manifest hash, lane, output directory, GPU lock, and cached-data path are explicit.
Do not launch a second controller for an existing lane.

See [preflight evidence](../../docs/validation/standard-bca-preflight.json).
Real OOD collection follows verified checkpoints and separate adapter gates;
training completion alone is neither OOD evidence nor evidence of benefit.
Importance weighting is a later, separately selected comparison. No final
outcomes will be used to pick its recipe.
