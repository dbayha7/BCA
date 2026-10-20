"""EXPLORATORY: result F's recorded seed-0 numbers (desktop), with file and line. Read only."""
import json
import re
from pathlib import Path

EX = Path("/home/user/bca/outputs/shift_iw/explore")
CH = EX / "channel" / "results"
LAWS = ("strong", "mild", "none")


def key_lines(path):
    """{tuple of keys (list items as ints): line number} for indent-formatted JSON (one key per line)."""
    out, stack = {}, []
    for no, line in enumerate(Path(path).read_text().splitlines(), 1):
        ind = len(line) - len(line.lstrip(" "))
        m = re.match(r'\s*"([^"]+)":', line)
        while stack and stack[-1][0] >= ind:
            stack.pop()
        if m:
            stack.append((ind, m.group(1)))
            out[tuple(k for _, k in stack)] = no
    return out


rows = []
for q in ("medium-expert", "expert"):
    # result F's own arms: evaluate_seed0.json
    for fam, arms in (("train_f1", ("A2_cmd_strong", "B2_cmd_train", "C2_untargeted")),
                      ("train_est", ("A3_estimated", "C3_untargeted")),
                      ("train", ("cmd_strong", "cmd_train", "untargeted"))):
        p = CH / fam / q / "evaluate_seed0.json"
        ev, kl = json.loads(p.read_text())["evaluations"], key_lines(p)
        for a in arms:
            rows.append(dict(quality=q, row=a, source=str(p),
                             **{law: (ev[a][law]["J"], ev[a][law]["points"], kl[("evaluations", a, law, "J")])
                                for law in LAWS}))
    # the references: RESULTS.json per_seed[0] (seeds listed in order 0-4)
    p = EX / "RESULTS.json"
    R = json.loads(p.read_text())
    assert R["seeds"] == [0, 1, 2, 3, 4]
    qi = [x["quality"] for x in R["qualities"]].index(q)
    arms = R["qualities"][qi]["arms"]
    text = p.read_text().splitlines()
    # line of each per_seed list: find the quality block, then scan keys in order
    start = next(i for i, l in enumerate(text) if l.strip() == f'"quality": "{q}",')
    for a in ("le_none", "le_bca_unweighted", "le_oracle"):
        vals = {}
        i = next(j for j in range(start, len(text)) if text[j].strip() == f'"{a}": {{')
        for law in LAWS:
            k = next(j for j in range(i, len(text)) if text[j].strip() == f'"{law}": {{')
            jl = next(j for j in range(k, len(text)) if text[j].strip() == '"per_seed": [')
            vals[law] = (arms[a][law]["J"]["per_seed"][0], arms[a][law]["points"]["per_seed"][0], jl + 2)
        rows.append(dict(quality=q, row=a, source=str(p), **vals))
    # the paired seed-0 differences of the pass marks: SCORE.json
    p = CH / "train_f1" / "SCORE.json"
    S, kl = json.loads(p.read_text()), key_lines(p)
    for k in ("A2_cmd_strong - v5_unweighted | strong", "A2_cmd_strong - v5_unweighted | none",
              "A2_cmd_strong - no_caution | strong", "A2_cmd_strong - no_caution | none",
              "A2_cmd_strong - C2_untargeted | strong"):
        d = S[q][k]
        rows.append(dict(quality=q, row=f"diff {k}", source=str(p), seed0_J=d["J"]["per_seed"][0],
                         seed0_points=d["points"]["per_seed"][0], mean_J=d["J"]["mean"], lower_J=d["J"]["lower"],
                         mean_points=d["points"]["mean"], per_seed_J=d["J"]["per_seed"],
                         line=kl[(q, k, "J", "per_seed")]))
print(json.dumps(rows, indent=1))
