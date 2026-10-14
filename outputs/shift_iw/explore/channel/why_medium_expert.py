"""POST HOC diagnosis (after scoring): where the command-weighted band refuses the final cmd_strong actor's commands,
by current strength and by episode type (random share 0 vs 0.65), on kept training rows. Read only."""
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/channel")
import command_le_train as T  # noqa: E402
from experiments.shift import learners as LR  # noqa: E402

out = {}
for quality in ("medium-expert", "expert"):
    for seed in (0, 1, 2, 3, 4):
        wd = T.World(quality, seed)
        rec = json.loads((T.arm_dir(quality, "cmd_strong", seed) / "record.json").read_text())
        params = LR.load_tree(T.arm_dir(quality, "cmd_strong", seed) / "actor.msgpack", LR.actor_template(),
                              rec["actor_sha256"])
        _, info, ref = wd.refresh(params, "strong")
        c, eps = wd.train.c, wd.train.eps
        bins = np.quantile(c, [0, 1 / 3, 2 / 3, 1])
        tercile = np.clip(np.searchsorted(bins, c, side="right") - 1, 0, 2)
        rand_row = np.asarray(wd.data.train_log.diagnostic("infos/random")[wd.prep.rows], bool)
        r = dict(refused=float(ref.mean()),
                 by_current_tercile=[float(ref[tercile == k].mean()) for k in range(3)],
                 by_episode_random_share={f"{e:g}": float(ref[eps == e].mean()) for e in np.unique(eps)},
                 refused_rows_whose_recorded_command_is_random=float(rand_row[ref].mean()) if ref.any() else None,
                 all_rows_recorded_random=float(rand_row.mean()))
        out[f"{quality} seed {seed}"] = r
        print(quality, seed, json.dumps(r))
