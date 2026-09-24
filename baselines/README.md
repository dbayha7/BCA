# Original baseline sources

[unifloral/](unifloral/) contains the **complete, unmodified tracked tree** of
[EmptyJackson/Unifloral](https://github.com/EmptyJackson/Unifloral/tree/f2dc1278eae18ed3e22c92255119c54885414d17)
at commit `f2dc1278eae18ed3e22c92255119c54885414d17`. A normal BCA clone includes
all 40 files. There is no submodule or additional source download.

| Read this | Purpose |
| --- | --- |
| [Original README](unifloral/README.md) | Upstream usage and citations |
| [Original algorithms](unifloral/algorithms/) | Standalone hosts, unified Unifloral algorithm and its helpers |
| [Standalone configs](unifloral/configs/algorithms/) | Original algorithm sweep definitions |
| [Unified configs](unifloral/configs/unifloral/) | Original Unifloral sweep definitions |
| [Requirements](unifloral/requirements.txt), [Dockerfile](unifloral/Dockerfile) | Original environment specifications |
| [License](unifloral/LICENSE) | Original Apache 2.0 license |
| [Reference manifest](../configs/unifloral.json) | Revision, Git blob/SHA256 identities for every file, and four audited standalone default profiles |

From the BCA root, `python check.py` checks every bundled file without importing
Unifloral. `python check.py --reference-config cql hopper` prints the original
default arguments, update counts, data hash and command. Four standalone profiles
are audited: IQL, CQL, TD3+BC and ReBRAC. Including the rest of upstream does not
declare more BCA experiments or validate their runtime behavior.

Unifloral stays a baseline reference. The four paired CORL-derived JAX hosts and
BCA attachments live in the root [algorithms/](../algorithms/). They do not import
these vendored scripts. BCA changes are mapped in [INTEGRATION.md](../INTEGRATION.md).

Use a separate environment for upstream: its dependency ranges differ from BCA's.
For upstream commands, work inside `baselines/unifloral/`; see the root
[reproduction notes](../README.md#unifloral-baseline-references). Installation in a
fresh environment is not yet validated. Original sweeps/defaults do not identify
the exact published table configurations, and 1M outer steps need not mean 1M
critic updates. Do not compare those budgets without the recorded counts.

Treat this directory as a source snapshot. Keep adaptations outside `unifloral/`
and give them explicit provenance. Refreshing upstream requires reviewing a new
pin, every file hash, licensing and defaults; it is never an automatic update.
