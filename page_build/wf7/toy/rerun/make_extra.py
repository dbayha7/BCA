"""Make the two information-only copies of the toy notebooks that the section's builder and number checks read
(rebuilt in the cloud on 16 Oct 2026: the laptop's copies were not handed over). Each copy is the committed notebook
with one code cell added after its last cell; the cell only reads the run's variables and prints, and changes nothing.
Usage: python make_extra.py SRC_DIR RUN_DIR (writes RUN_DIR/extra_discrete.ipynb, RUN_DIR/extra_weights.ipynb and RUN_DIR/extra_channel.ipynb)."""
import json
import sys

src, run = sys.argv[1], sys.argv[2]

DISCRETE = r'''# Information only (added for the ThrusterChain page; changes nothing): seed 7's calibration data.
print("EXTRA cal contexts seed 7:", int((cal["c"] == 0).sum()), int((cal["c"] == 1).sum()))
print("EXTRA weight sums", repr(float(cal_weights.sum())), f"{float(cal_weights @ cal_weights):.2f}")
_v, _n = np.unique(np.round(cal_scores[0], 4), return_counts=True)
print("EXTRA stage-0 scores", [(float(a), int(b)) for a, b in zip(_v, _n)])
_be = (q[1, 0, 1] - q[1, 0, 0]) / (scale[1, 0, 1] - scale[1, 0, 0])
print(f"EXTRA break-even at (1,0): {_be:.4f}")
print("EXTRA BCA widths at (1,0):", [float(x) for x in widths("BCA")[1, 0]], "weighted:", [float(x) for x in widths("weighted BCA")[1, 0]])
'''

WEIGHTS = r'''# Information only (added for the ThrusterChain page; changes nothing): the rates the notebook's figure draws.
import json
_p, _s, _z = PROFILES.index("harm in harsh conditions"), SHIFTS.index("strong"), SIZES.index(120)
print("EXTRA_RUNG_RATES " + json.dumps({g: [float(x) for x in rate(DEC[g])[_p, :, _s, _z]] for g in show}))
print("EXTRA_TRUTH " + json.dumps({"usual_below": round(float(USUAL_BELOW), 9), "usual_mean": round(float(USUAL_MEAN), 6),
                                   "mu": [round(float(x), 6) for x in MU], "bad": [round(float(x), 6) for x in BAD]}))
print(f"EXTRA_COUNTS unsafe worlds {int(UNSAFE_W.sum())} harmful {int(HARMFUL_W.sum())} safe shifted {int(SAFE_SHIFTED.sum())} allowance {ALLOW:.4f}")
print("EXTRA worst unweighted mean world:", float(rate(DEC["BCA mean, unweighted"])[HARMFUL_W].max()))
print("EXTRA wbar", [float(x) for x in WBAR])
'''

CHANNEL = r'''# Information only (added for the ThrusterChain page; changes nothing): the pooled regrets the notebook prints to 4
# decimals, in full, so that 3-decimal roundings on the page can be checked.
print("EXTRA pooled regret", {m: repr(float(pooled[M[m]])) for m in METHODS})
print("EXTRA room", repr(float(room)))
'''

for name, cell, out in (("bca_discrete_toy.ipynb", DISCRETE, "extra_discrete.ipynb"),
                        ("bca_weights_toy.ipynb", WEIGHTS, "extra_weights.ipynb"),
                        ("bca_channel_toy.ipynb", CHANNEL, "extra_channel.ipynb")):
    nb = json.load(open(f"{src}/{name}", encoding="utf-8"))
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": cell.splitlines(keepends=True)})
    json.dump(nb, open(f"{run}/{out}", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out, "with", len(nb["cells"]), "cells")
