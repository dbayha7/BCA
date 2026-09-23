"""Published config."""

from __future__ import annotations
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_CSV = os.path.abspath(
    os.path.join(_HERE, "..", "configs", "corl_reference_scores.csv")
)
REFERENCE_PROVENANCE = os.path.splitext(REFERENCE_CSV)[0] + ".PROVENANCE"
CORL_UPSTREAM_SHA = "eeeeef2"
_META = frozenset(
    {"tag", "env", "domain", "published_mean", "published_std", "has_config"}
)
_RENAME = {"learning_rate": "lr"}
_RENAME_BY_TAG = {
    "corl_cql": {"reward_scale": "cql_reward_scale", "reward_bias": "cql_reward_bias"}
}
_STEPS_DIRECT = "max_timesteps"
_STEPS_EPOCHS = ("num_epochs", "num_updates_on_epoch")


def _scalar(t):
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return None


def _coerce(text):
    t = text.strip()
    v = _scalar(t)
    if v is not None:
        return v
    if "#" in t:
        v = _scalar(t.split("#", 1)[0].strip())
        if v is not None:
            return v
    return t


def _load():
    if not os.path.exists(REFERENCE_CSV):
        return {}
    out = {}
    with open(REFERENCE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("has_config", "").strip().lower() != "yes":
                continue
            cfg = {}
            per_tag = _RENAME_BY_TAG.get(row["tag"], {})
            for col, raw in row.items():
                if col in _META or not (raw or "").strip():
                    continue
                if col == _STEPS_DIRECT:
                    cfg["num_updates"] = _coerce(raw)
                elif col in _STEPS_EPOCHS:
                    continue
                else:
                    cfg[per_tag.get(col, _RENAME.get(col, col))] = _coerce(raw)
            ep, per = (row.get(_STEPS_EPOCHS[0], ""), row.get(_STEPS_EPOCHS[1], ""))
            if ep.strip() and per.strip():
                cfg["num_updates"] = _coerce(ep) * _coerce(per)
            out[row["tag"], row["env"]] = cfg
    return out


PUBLISHED = _load()


def published_config(tag, dataset):
    return PUBLISHED.get((tag, dataset))


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and (a == b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12 * max(
            1.0, abs(float(a)), abs(float(b))
        )
    return a == b


def diff_against_published(tag, dataset, args):
    cfg = published_config(tag, dataset)
    if cfg is None:
        return None
    bad = []
    for field, want in sorted(cfg.items()):
        if not hasattr(args, field):
            continue
        got = getattr(args, field)
        if not _same(got, want):
            bad.append((field, got, want))
    return bad


def check_published_config(tag, dataset, args, allow_off_config=False):
    bad = diff_against_published(tag, dataset, args)
    if bad is None:
        msg = f"\n{tag} x {dataset}: NO PUBLISHED CONFIGURATION IS ON RECORD.\n  `{os.path.relpath(REFERENCE_CSV)}` carries no row for this cell, so there\n  is nothing to check this run against and no published number to compare\n  it to. A run that cannot be checked must not be able to claim it\n  reproduces anything.\n  Add the row (with its yaml provenance) or pass --allow_off_config and\n  accept that the run is not a reproduction.\n"
        if allow_off_config:
            print(f"[published-config] OFF-CONFIG, DECLARED:{msg}", flush=True)
            return
        sys.exit(msg)
    if bad:
        lines = "\n".join(
            (f"    {f:24s} run has {g!r}, published is {w!r}" for f, g, w in bad)
        )
        msg = f"\n{tag} x {dataset}: this run does NOT carry the published configuration.\n{lines}\n  These come from CORL's per-environment yaml, which OVERRIDES the file\n  defaults. Case I checks the file default against CORL's file default and\n  passes; it cannot see this. A run at the wrong per-environment value\n  trains a different algorithm from the one whose published score it would\n  be compared against.\n  Pass the flags above, or pass --allow_off_config to declare that this run\n  is deliberately not a reproduction.\n"
        if allow_off_config:
            print(f"[published-config] OFF-CONFIG, DECLARED:{msg}", flush=True)
            return
        sys.exit(msg)
    n = len(published_config(tag, dataset))
    print(
        f"[published-config] {tag} x {dataset}: matches the published configuration on all {n} recorded field(s)",
        flush=True,
    )


def coverage():
    by_tag = {}
    for (tag, _env), cfg in PUBLISHED.items():
        by_tag.setdefault(tag, []).append(cfg)
    out = {}
    for tag, cfgs in sorted(by_tag.items()):
        fields = sorted({k for c in cfgs for k in c})
        varies = [f for f in fields if len({repr(c.get(f)) for c in cfgs}) > 1]
        out[tag] = (len(cfgs), varies)
    return out
