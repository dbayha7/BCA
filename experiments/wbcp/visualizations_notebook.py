"""Build and execute experiments/wbcp/visualizations.ipynb: BCA's calibration pipeline and its theory in action.

Six figures, each drawn by experiments/wbcp/viz/fig_<key>.py from the repo's own code (calibration/wbcp.py,
calibration/bank.py, the dose formula) and recorded run data under runs/. Each figure is preceded by its
explanation (experiments/wbcp/viz/text/<key>.md) and followed by the table of every number it shows, with its source.
Figures are also written to runs/wbcp_viz/<key>.png at 200 dpi for slides. The notebook is executed here, so the
saved file carries its outputs (needs matplotlib, pandas, numpy, ipykernel and jupyter_client).

python experiments/wbcp/visualizations_notebook.py [--output experiments/wbcp/visualizations.ipynb] [--no-execute]
"""

import argparse
import json
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from experiments.wbcp.results_notebook import execute  # noqa: E402

FIGURES = [  # (module key, section heading)
    ("pipeline", "1. The whole pipeline, from training to testing"),
    ("wbcp_threshold", "2. WBCP: from a calibration bank to a threshold"),
    ("shift_weighting", "3. Distribution shift: uniform BQ-CP vs importance-weighted WBCP"),
    ("bank_dependence", "4. Episode dependence: why the calibration bank is thinned"),
    ("critic_alignment", "5. Critic alignment: which error the band certifies"),
    ("bc_dose", "6. From calibrated width to BC dose, and why mean dose is not strength"),
]

INTRO = r"""# BCA calibration: the pipeline and its theory in action

This notebook shows how BCA's calibration works and what each part does, using the repo's own code and recorded
runs. The first figure is the whole pipeline. The next five each take one theoretical component and run it on real
data:

1. **Pipeline.** How BCA goes from the offline dataset, through host training and calibration, to the four kinds of
   test, with the status of each test.
2. **WBCP threshold.** How weighted Bayesian conformal prediction (Lou & Luo, Algorithm 1; uniform weights = BQ-CP)
   turns a bank of held-out scores into a certified threshold.
3. **Distribution shift.** What importance weights do when the test law differs from the calibration law, and what
   they cost.
4. **Episode dependence.** Why a bank of whole episodes breaks the guarantee and a thinned bank restores it.
5. **Critic alignment.** Why a band calibrated on min(Q1, Q2) does not certify the Q1 that TD3+BC's actor climbs.
6. **Dose.** How the calibrated width becomes a BC multiplier, why it acts as nearly uniform extra BC, and why matching
   the mean dose does not match strength.

**Conventions used throughout.**
- Calibration settings: target miscoverage α = 0.1 at credibility β = 0.95, M = 1,000 posterior draws.
- A calibration bank **fails** when the threshold it certifies misses more than 10% of test rows. A valid rule fails
  in at most 5% of banks (1 − β), the dashed budget line in the figures.
- Colors: **blue** is the method working as intended (WBCP, the Q1 band); **orange** is the case being explained
  (uniform BQ-CP under shift, whole-episode banks, the min(Q1, Q2) band, the oracle dose). Panels marked
  *Illustration* are constructed examples; everything else is computed from recorded data.
- Under each figure, a table lists every number it shows with its source.

**What these figures do not show.** All D4RL evidence comes from frozen critics (one 100k-update critic per host and
dataset, one seed) at logged actions. No WBCP training run exists yet, so nothing here measures return, and the
step-4 placement study has no outcomes yet.

To rebuild after new runs: `python experiments/wbcp/visualizations_notebook.py`. The figure files are
`experiments/wbcp/viz/fig_<key>.py` and the explanations `experiments/wbcp/viz/text/<key>.md`."""

SETUP = r"""import os
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parents[1]  # this notebook lives in experiments/wbcp
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import importlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from experiments.wbcp.viz import style as S

%matplotlib inline
pd.set_option("display.max_colwidth", None)
print("font:", S.setup())


def draw(key):
    # Run experiments/wbcp/viz/fig_<key>.py, save runs/wbcp_viz/<key>.png, show the figure and its facts.
    module = importlib.import_module(f"experiments.wbcp.viz.fig_{key}")
    fig, facts = module.make()
    path = S.save(fig, key)
    display(fig)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")
    display(pd.DataFrame(facts, columns=["what", "value", "source"]).style.hide(axis="index"))"""

CLOSING = r"""## Provenance

- Code: `experiments/wbcp/viz/` (one module per figure, plus `style.py`). Each module's docstring says which data it
  reads. Every module was reviewed against the repo code and its sources, and each figure re-rendered after the fixes.
- Data: recorded runs under `runs/` (frozen score pools in `runs/wbcp_frozen`, benchmark results in `runs/wbcp_bench`,
  `runs/wbcp_dependence` and `runs/wbcp_hosts`, the signal study in `runs/wbcp_signal`). Nothing here trains a model
  or runs a benchmark; every figure renders in seconds.
- Write-ups behind the numbers: `experiments/wbcp/DEPENDENCE.md`, `experiments/wbcp/CRITIC_ALIGNMENT.md`,
  `experiments/signal/SIGNAL_STUDY.md`, `ALGORITHMS.md`, `INTEGRATION.md`."""


def cell(kind, source):
    lines = source.strip("\n").split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    out = {"cell_type": "markdown" if kind == "md" else "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def notebook_cells():
    cells = [cell("md", INTRO), cell("code", SETUP)]
    for key, heading in FIGURES:
        with open(os.path.join(HERE, "viz", "text", key + ".md"), encoding="utf-8") as handle:
            text = handle.read()
        cells.append(cell("md", f"## {heading}\n\n{text}"))
        cells.append(cell("code", f'draw("{key}")'))
    cells.append(cell("md", CLOSING))
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=os.path.join(HERE, "visualizations.ipynb"))
    parser.add_argument("--no-execute", action="store_true", help="write the notebook without outputs")
    args = parser.parse_args(argv)
    cells = notebook_cells()
    failures = [] if args.no_execute else execute(cells, os.path.dirname(os.path.abspath(args.output)))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
                "nbformat": 4, "nbformat_minor": 5}
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {args.output}: {len(cells)} cells" + (f"; {len(failures)} cell errors: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
