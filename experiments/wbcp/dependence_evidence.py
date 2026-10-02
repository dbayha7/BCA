"""Dependence evidence registry: measured no-shift failure of BCA's configured banks.

Reads d4rl_benchmark.py results for banks drawn with BCA's own sampler (--spacing
reservation, --per-episode K) and writes calibration/dependence_evidence.json: one entry
per measured (host, dataset, K, n, sampler, score) with uniform BCA's failure rate without
shift (the BQ-CP arm at gamma 0 on the normalized score), its exact 95% interval and the
run it came from. calibration.bank.dependence_evidence looks a configured bank up there,
and runtime/config.resolve puts the answer in every bca row as `dependence_evidence`, for
the score that host's BCA calibrates (calibration.bank.DEPLOYED_SCORE).

Status rule (calibration.bank.evidence_status, applied at lookup):
  consistent with the 5% budget            the interval's lower bound is at most 0.05
  exceeds the 5% budget                    the lower bound is above 0.05
  not validated for this host/dataset/K    no entry has exactly this host, dataset, K,
                                           n, sampler and score
The budget is 1 - credibility: a beta = 0.95 WBCP threshold may fail in at most 5% of
banks. "Consistent" means the runs cannot reject the budget, not that they prove it.
Evidence never carries over: another host, dataset, K, n, sampler or score is unmeasured,
so a Q1 or max bank on a host's pool is not evidence for the min(Q1, Q2) bank it deploys.

Sources (SOURCES): the configured-bank runs behind DEPENDENCE.md (Changes 9-12) and the
CQL host matrix. A file enters when it is a benchmark result with spacing 'reservation', a
per-episode K, the heldout population, the deployed WBCP settings (alpha 0.1, beta 0.95,
1,000 draws), no shuffled tilt and a gamma = 0 block; any other file is listed as skipped
with its reason. Host and dataset come from the frozen pool's name (TD3+BC:
<dataset>-s..., or hopper-medium-v2-s...; other hosts: <host>-<dataset>-s...) and must
match the pool's own frozen.json record; the score comes from that record's definition
of q (SCORES). Critic health is host_matrix.critic_health on the pool's training log when
the pool is on disk, else None (not assessed). A result drawn with the remainder trim
(calibration/bank.py, 2026-10-01) records the sampler revision as `bank_trim`
(calibration.bank.REMAINDER_TRIM), copied into its entries; a result without it was drawn
before the trim, when a bank could hold up to K ceil(n / K) rows, and its note says
whether, and by how much, the trim changes that design. Bank sizes alone cannot tell the
two apart: a trimmed bank holds at most n rows, like an untrimmed design K divides.

python experiments/wbcp/dependence_evidence.py            # print the table
python experiments/wbcp/dependence_evidence.py --write    # rewrite calibration/dependence_evidence.json
python experiments/wbcp/dependence_evidence.py --check    # exit 1 unless the registry matches the runs
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import yaml
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from calibration.bank import (DEPENDENCE_BUDGET, EVIDENCE_PATH, EVIDENCE_SCHEMA, MIN_Q, REMAINDER_TRIM,  # noqa: E402
                              TARGET_MIN_Q, evidence_status)

SOURCES = (
    "runs/wbcp_dependence/all-datasets/*_resv*.json",
    "runs/wbcp_dependence/other-datasets/w_resv*.json",
    "runs/wbcp_dependence/other-datasets/p_resv*.json",
    "runs/wbcp_dependence/hopper-u100000-reservation/r*_*.json",
    "runs/wbcp_hosts/cql/*_resv*.json",
    "runs/wbcp_hosts/cql/superseded_pre_trim/*_resv*.json",  # CQL runs from before the remainder trim (step 1a)
)
HOSTS = ("td3_bc", "rebrac", "cql", "iql")
DEPLOYED = dict(alpha=0.1, beta=0.95, draws=1000)  # calibration.reference.WBCPConfig, every config's bca block
SCORES = {  # frozen.json score_definition["q"] -> the calibrated score (labels: calibration.bank)
    "min over the two critic heads at (obs, action)": MIN_Q,  # freeze_scores.py (TD3+BC), freeze_rebrac.py
    "min(Q1, Q2) at (obs, action)": MIN_Q,  # freeze_cql.py
    ("min over the twin Polyak target Q heads at (obs, action), the copy IQL's actor advantage reads "
     "(calibration.iql_reference.refresh)"): TARGET_MIN_Q,  # freeze_iql.py
}
POOL_NAME = re.compile(r"^(?:(%s)-)?(.+)-s\d+-u\d+$" % "|".join(HOSTS))
STATUS_RULE = ("consistent with the 5% budget if the 95% interval's lower bound is at most 0.05; "
               "exceeds the 5% budget otherwise; not validated for this host/dataset/K without an entry "
               "matching host, dataset, K, n, sampler and score exactly; a configured bank is looked up "
               "with the score its host's BCA calibrates (calibration.bank.evidence_status, DEPLOYED_SCORE)")


class Skip(Exception):
    """A source file that is not evidence for a configured bank; the message says why."""


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def pool_identity(pool, frozen, root=ROOT):
    """(host, config dataset key, D4RL name) of a frozen pool, from its name, checked
    against the configs and the pool's frozen.json record."""
    match = POOL_NAME.match(os.path.basename(os.path.normpath(pool)))
    if match is None:
        raise ValueError(f"cannot read host and dataset from the pool name {pool!r}")
    host, part = match.group(1) or "td3_bc", match.group(2)
    with open(os.path.join(root, "configs", host + ".yaml"), encoding="utf-8") as handle:
        datasets = yaml.safe_load(handle)["datasets"]
    names = [key for key, entry in datasets.items() if part in (key, entry["environment"])]
    if len(names) != 1:
        raise ValueError(f"pool {pool!r} names no single {host} dataset")
    environment = datasets[names[0]]["environment"]
    if (frozen.get("algorithm"), frozen.get("dataset")) != (host, environment):
        raise ValueError(f"pool {pool!r} records {frozen.get('algorithm')} / {frozen.get('dataset')}, "
                         f"its name says {host} / {environment}")
    return host, names[0], environment


def score_label(frozen):
    q = (frozen.get("score_definition") or {}).get("q")
    if q not in SCORES:
        raise ValueError(f"no score label for q = {q!r}; add its definition to SCORES")
    return SCORES[q]


def trim_note(rows, n, trimmed):
    """How the remainder trim (2026-10-01) bears on a design: `trimmed` when its result
    carries the bank_trim marker, else measured before the trim."""
    held = (f"{rows['max']:,} rows" if rows["min"] == rows["max"]
            else f"{rows['min']:,}-{rows['max']:,} rows (mean {rows['mean']:,.1f})")
    if trimmed:
        return (f"Measured with the remainder trim (2026-10-01): these banks held {held} against n = {n:,}"
                + ("." if rows["min"] == n else "; reserved episodes shorter than K leave fewer than n."))
    if rows["max"] <= n:
        return ("Measured before the remainder trim (2026-10-01). No bank exceeded n rows, so the trim "
                "never fires on this design and a rerun reproduces it bit for bit.")
    return (f"Measured before the remainder trim (2026-10-01): these banks held {held} against n = {n:,}. "
            "The trim now removes the surplus at random; rerun to confirm.")


def entries_from_run(path, root=ROOT, health=None):
    """Registry entries (one per bank size n) from one benchmark result; Skip if it is not
    evidence for a configured bank. health(pool_dir, dataset_key) gives critic health."""
    with open(os.path.join(root, path), encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or not {"settings", "frozen", "blocks"} <= set(result):
        raise Skip("not a d4rl_benchmark.py result")
    settings, frozen = result["settings"], result["frozen"]
    if settings.get("spacing") != "reservation" or not settings.get("per_episode") or settings.get("blocks"):
        raise Skip(f"spacing {settings.get('spacing')!r}, per_episode {settings.get('per_episode')}: not BCA's sampler")
    if settings.get("population") != "heldout" or settings.get("shuffle_tilt") is not None:
        raise Skip("not the heldout population with unshuffled tilts")
    if any(settings.get(name) != value for name, value in DEPLOYED.items()):
        raise Skip("WBCP settings differ from the deployed alpha 0.1, beta 0.95, 1,000 draws")
    host, key, environment = pool_identity(settings["frozen"], frozen, root)
    score, k = score_label(frozen), int(settings["per_episode"])
    marker = result.get("bank_trim")  # absent: drawn before the remainder trim
    if marker not in (None, REMAINDER_TRIM):
        raise ValueError(f"{path}: unknown sampler revision bank_trim = {marker!r}")
    found = {}
    for block in result["blocks"]:
        if block["gamma"] == 0.0 and block["score"] == "normalized":
            found.setdefault(int(block["n"]), []).append(block)
    if not found:
        raise Skip("no gamma = 0 block on the normalized score (shift-only run)")
    entries = []
    for n, blocks in sorted(found.items()):
        arm = blocks[0]["arms"]["BQ-CP"]
        if any(b["arms"]["BQ-CP"]["failures"] != arm["failures"] for b in blocks):
            raise ValueError(f"{path}: uniform BCA differs between gamma = 0 tilts at n = {n}")
        low, high = stats.binomtest(arm["failures"], arm["trials"]).proportion_ci(method="exact")
        if abs(arm["fail"] - arm["failures"] / arm["trials"]) > 1e-12 or max(
                abs(low - arm["ci"][0]), abs(high - arm["ci"][1])) > 1e-9:
            raise ValueError(f"{path}: the recorded failure rate or interval does not match its counts")
        rows = blocks[0]["calibration_size"]
        if marker is not None and rows["max"] > n:
            raise ValueError(f"{path}: a bank drawn with the remainder trim held {rows['max']:,} rows, over n = {n:,}")
        entries.append(dict(
            host=host, dataset=environment, config_dataset=key, rows_per_episode=k, size=n,
            sampler="reservation", score=score,
            uniform_bca_failure=arm["fail"], ci95=list(arm["ci"]), failures=arm["failures"], trials=arm["trials"],
            bank_rows=dict(mean=rows["mean"], min=rows["min"], max=rows["max"]),
            source=path, source_sha256=sha256(os.path.join(root, path)), benchmark_seed=settings["seed"],
            pool=settings["frozen"], pool_npz_sha256=frozen.get("npz_sha256"),
            critic_health=None if health is None else health(settings["frozen"], key),
            bank_trim=marker, note=trim_note(rows, n, marker is not None),
        ))
    return entries


def build(sources=SOURCES, root=ROOT, health=None):
    """(entries sorted by host, dataset, n and K; [(source, reason)] skipped). Two runs of
    the same host, dataset, K, n, sampler and score are an error: pick one explicitly."""
    paths = sorted({os.path.relpath(p, root).replace(os.sep, "/")
                    for pattern in sources for p in glob.glob(os.path.join(root, pattern))})
    entries, skipped = [], []
    for path in paths:
        try:
            entries.extend(entries_from_run(path, root, health))
        except Skip as reason:
            skipped.append((path, str(reason)))
    keys = [(e["host"], e["dataset"], e["rows_per_episode"], e["size"], e["sampler"], e["score"]) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("two runs measure the same design: " + str(sorted(k for k in keys if keys.count(k) > 1)))
    entries.sort(key=lambda e: (e["host"], e["dataset"], e["size"], e["rows_per_episode"]))
    return entries, skipped


def registry(entries, sources=SOURCES):
    return dict(schema=EVIDENCE_SCHEMA, generator="experiments/wbcp/dependence_evidence.py",
                budget=DEPENDENCE_BUDGET, status_rule=STATUS_RULE, sources=list(sources), entries=entries)


def dumps(record):
    return json.dumps(record, indent=1, allow_nan=False) + "\n"


def critic_health(pool, key):
    from experiments.wbcp.host_matrix import critic_health as check
    return check(pool, key)


def print_table(entries, skipped):
    print(f"{'host':<7}{'dataset':<30}{'K':>5}{'n':>7}{'fail':>8}{'95% CI':>17}  {'status':<31}critic")
    for e in entries:
        ci = f"[{e['ci95'][0]:.1%}, {e['ci95'][1]:.1%}]"
        health = "not assessed" if e["critic_health"] is None else e["critic_health"]["status"]
        print(f"{e['host']:<7}{e['dataset']:<30}{e['rows_per_episode']:>5}{e['size']:>7}"
              f"{e['uniform_bca_failure']:>8.1%}{ci:>17}  {evidence_status(e['ci95']):<31}{health}")
    for path, reason in skipped:
        print(f"skipped {path}: {reason}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="rewrite " + os.path.relpath(EVIDENCE_PATH, ROOT))
    action.add_argument("--check", action="store_true", help="exit 1 unless the registry matches the runs")
    args = parser.parse_args(argv)
    entries, skipped = build(health=critic_health)
    print_table(entries, skipped)
    text = dumps(registry(entries))
    if args.write:
        with open(EVIDENCE_PATH, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"wrote {len(entries)} entries to {os.path.relpath(EVIDENCE_PATH, ROOT)}")
    elif args.check:
        with open(EVIDENCE_PATH, encoding="utf-8") as handle:
            same = handle.read() == text
        print("registry matches the runs" if same else "registry differs from the runs; rerun with --write")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
