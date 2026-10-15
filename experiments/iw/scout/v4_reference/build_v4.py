"""Build version 4's data with version 4's own code (a frozen copy of world.py and data.py, taken before any version 5
edit) so version 5's expert column can be compared with it array by array.

  python build_v4.py OUT SEED SIZE      SIZE: small | tiny (test_data's TINY)
Writes OUT/seed<k>/... (version 4's layout) and OUT/hashes_seed<k>_<size>.json: the SHA-256 of every array of every
file, the ball, the quadrature and D1."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4ref import data as D  # noqa: E402

TINY = D.Sizes(train=24, calibration=8, held_out=8, test=8, sample=8, grid_points=5, per_point=20, truth_rows=100,
               noise_map_horizon=50, start_bank=20)


def main():
    out, seed, size = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sizes = {"small": D.SMALL, "tiny": TINY}[size]
    world, times = D.build_seed(seed, out, sizes, log=print)
    root = out / f"seed{seed}"
    hashes = {}
    for name in ("train_law.hdf5", "test_mild.hdf5", "test_strong.hdf5", "samples.hdf5", "truth.hdf5"):
        m = json.loads((root / (name + ".manifest.json")).read_text())
        hashes[name] = {k: v["sha256"] for k, v in m["arrays"].items()}
    res = dict(seed=seed, size=size, sizes=sizes.as_dict(), arrays=hashes, gap=world["gap"],
               quadrature=world["quadrature"], D1=world["D1"], removal=world["removal"],
               test_in_ball_share=world["test_in_ball_share"], code_sha256=world["code_sha256"])
    (out / f"hashes_seed{seed}_{size}.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    print("seconds", times)


if __name__ == "__main__":
    main()
