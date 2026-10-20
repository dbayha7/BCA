"""Near-80 (version 9, main setting): the pre-approval facts (the freeze's step 7; PREREGISTRATION.md sections 4.2
and 8.1, check 5). DRAFT, not frozen. Nothing is trained, no BCA band is fitted, no outcome of any learner is read.

Two kinds of fact, written into ONE record (write-once, outside the repository or under the official output folder):

  the counts on seeds 0-4 (--seeds 0 1 2 3 4): the gate's verified logs through acceptance.load_seed (which
      regenerates each log and checks it, its ball and both gate removals against runs/shift/pre/data/base/<world>/
      seed<k>/record.json, read only), step 1's gap rule (verified bytes) on the ball-kept training commands, and the
      pooled groups' outcome-free count (near80lib.pooled_count_record). Commands and split labels only.

  the spent-seed switch facts (--spent-switch 10 11 12 13 14): for each of version 8's fresh seeds and worlds, the
      seed's log regenerated as version 8's data phase generated it (dataset.generate with the record's recorder,
      the gate's sizes and the record's stall slope) and checked against version 8's data record (every array's
      SHA-256, the ball, both removals' kept-row and bank-row hashes; the record itself against its .sha256 sidecar);
      version 8's recorded second critic loaded from runs/shift_explore/near80/learners/main/<world>/seed<k>/
      critic_second.msgpack and checked against its record (the file's SHA-256, the parameters' SHA-256, the
      survivors' SHA-256); then, with the frozen code, the pooled groups' scores (learner_phases.switch_scores_pooled),
      BCA's certificate on them (certificate_entry: the same tags and draws), the plain 90% gap and twice it, the
      mean difference, the rank test, each height's rank among the group's scores, and the four switches' calls;
      beside them version 8's recorded (unpooled) calls from its switch.json, and the kill rule: BCA's and the
      mean-difference switch's pooled calls equal the recorded ones on every spent seed-world (kill_rule_passed).
      Version 8's outputs are read and never written. Nothing is trained: the critic is version 8's file.

Seeds: 0-4 for the counts and 10-14 for the switch facts, and no other (near80lib.check_seeds, check_spent_seeds); the
fresh seeds 40-49 and the reserved 50-59 are never generated, loaded or read by this script.

Usage (from the work folder, one CPU, the pinned environment):
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=<repo> \
    python <repo>/outputs/shift/beyond-the-data/near80/code/facts_near80.py --out OUT.json \
      [--seeds 0 1 2 3 4] [--spent-switch 10 11 12 13 14] [--worlds stall no_stall]

Cost: about 13 s per seed-world for the counts (the log is regenerated); about 30-60 s per spent seed-world (the
log regenerated, 42,000 rows scored by the recorded critic, 1,000 posterior draws). About 10-15 CPU-minutes in all.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import near80_constants as K  # noqa: E402
import near80lib as N  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

RUNS = N.ROOT / "runs" / "shift"
KIND = "near80-v9-pre-approval-facts"
COUNTS_KIND = "near80-v9-pooled-group-counts"          # the counts' own kind, kept for the record
SPENT_KIND = "near80-v9-spent-seed-switch-facts"
V8_SWITCH_REL = "learners/main/{world}/seed{seed}/switch.json"
V8_CRITIC2_REL = "learners/main/{world}/seed{seed}/critic_second.json"
V8_DATA_REL = "data/main/{world}/seed{seed}/data.json"


# --------------------------------------------------------------------------------------------- the counts (seeds 0-4)

def seed_world_counts(log, gap, world, removal, step1):
    """The outcome-free count record of one seed-world (near80lib.pooled_count_record) with its identifiers."""
    rec = N.pooled_count_record(log, gap, removal, step1)
    rec.update(rep=int(log.meta["rep"]), world=world, kept_training_rows=int(len(removal.kept_train_rows(log))),
               bank_rows_kept=int(len(removal.bank_rows(log))),
               calibration_rows=int(len(log.split_rows("calibration"))),
               evaluation_rows=int(len(log.split_rows("evaluation"))))
    return rec


def _bank(situation, n):
    return TC.bank_starts(situation) if n is None else TC.bank_starts(situation, int(n))


def anchors_for(world, spec=RC.DEFAULT_SPEC, bank_n=None):
    """{situation: (J of the recording controller on the bank, J of the world's top controller)}, exact. The
    points' anchors, as the gate computes them (acceptance.anchors_of) and as stub mode replaces them."""
    from experiments.shift import truth as TR
    physics = TC.world_physics(world)
    top = RC.stall_aware(physics) if world == "stall" else RC.rising_thrust(physics)
    n = TC.BANK_SIZE if bank_n is None else int(bank_n)
    out = {}
    for s in TC.SITUATIONS:
        x0 = _bank(s, bank_n)
        out[s] = (float(TR.rollout(TR.recorder_on_bank(physics, spec, n), x0, physics).J.mean()),
                  float(TR.rollout(top, x0, physics).J.mean()))
    return out


def summary(results):
    """The smallest and largest pooled full-throttle group per world, and how many seed-worlds reach CERT_MIN_ROWS."""
    out = {}
    for w in K.WORLDS:
        rows = [r for r in results if r["world"] == w and r.get("groups")]
        sizes = [r["groups"]["n_full_throttle"] for r in rows]
        out[w] = dict(seed_worlds=len(rows), n_full_throttle_min=min(sizes) if sizes else None,
                      n_full_throttle_max=max(sizes) if sizes else None,
                      reach_min_rows=sum(1 for r in rows if r["certification"]["reaches_min_rows"]),
                      version8_min=min((r["version8_groups"]["n_full_throttle"] for r in rows), default=None),
                      version8_max=max((r["version8_groups"]["n_full_throttle"] for r in rows), default=None))
    return out


# --------------------------------------------------------------------------------------------- the spent seeds 10-14

def num(x):
    """A recorded number ('inf', '-inf', 'nan' strings as step2lib writes them)."""
    if x is None:
        return None
    return float(x)


def switch_calls_of(entry, margin=K.MEANDIFF_MARGIN, rule=K.ARM_RULE):
    """The four switches' calls from one certificate entry (the recorded one or the pooled one): BCA's certificate,
    the mean difference > margin, the plain 90% gap > the conformal margin, the rank test p < rank_p; with their
    statistics and BCA's height (policy_phase.step2_arms' rules, restated without step 2's module)."""
    c, g, rt = entry["certificate"], entry["groups"], entry["rank_test"]
    h = num(c.get("height", "nan"))
    mean_gap = num(g["mean_full_throttle"]) - num(g["mean_rest"])
    plain_gap = num(g["q90_full_throttle"]) - num(g["q90_rest"])
    p = num(rt.get("p"))
    return dict(bca=bool(c["on"]), bca_height=h, bca_refused=bool(c["on"] and not math.isfinite(h)),
                bca_q05=num(c.get("q05")), finite_share_tail=c.get("finite_share_tail"),
                tail_certified=c.get("tail_certified"),
                meandiff=bool(math.isfinite(mean_gap) and mean_gap > margin), mean_gap=mean_gap,
                conformal=bool(math.isfinite(plain_gap) and plain_gap > float(rule["conformal_margin"])),
                plain_gap=plain_gap, twice_plain_gap=2.0 * plain_gap,
                rank=bool(p is not None and math.isfinite(p) and p < float(rule["rank_p"])), rank_p=p,
                n_full_throttle=int(g["n_full_throttle"]), n_rest=int(g["n_rest"]))


def kill_rule(pooled_calls, recorded_calls):
    """Section 8.1, check 5: BCA's and the mean-difference switch's pooled calls equal version 8's recorded calls."""
    same = dict(bca=bool(pooled_calls["bca"] == recorded_calls["bca"]),
                meandiff=bool(pooled_calls["meandiff"] == recorded_calls["meandiff"]))
    return dict(same, passed=bool(all(same.values())),
                also=dict(conformal=bool(pooled_calls["conformal"] == recorded_calls["conformal"]),
                          rank=bool(pooled_calls["rank"] == recorded_calls["rank"])))


def _read_v8(rel):
    """A version 8 record under K.V8_OUT, verified against its sidecar (read only)."""
    import ledger as LG
    p = Path(K.V8_OUT) / rel
    body = LG.read_json_verified(p)
    return body, LG.file_sha256(p), str(p)


def spent_seed_data(seed, world, record):
    """Version 8's seed regenerated as its data phase generated it and checked against its data record: the log
    (dataset.generate with the record's recorder, the gate's sizes, the record's stall slope; every array's SHA-256
    equal to the record's), the ball (the record's gap), both removals (kept-row and bank-row hashes equal).
    settingdata.FreshSeed restated for a spent seed (FreshSeed refuses seeds outside this version's)."""
    import settingdata as SD
    from experiments.shift import dataset as DS
    from experiments.shift import shift as SH
    if not record.get("passed_gate_checks"):
        raise N.Refused(f"refused: version 8's seed {seed} ({world}) failed the gate's checks")
    spec = RC.RecorderSpec(**record["recorder"])
    sizes = SD.fresh_settings()[1]
    slope = float(record["stall_slope"])
    lg = DS.generate(int(seed), world, spec, tuple(sizes), slope)
    hashes = lg.array_hashes()
    if hashes != record["log_sha256"]:
        bad = sorted(k for k in record["log_sha256"] if hashes.get(k) != record["log_sha256"][k])
        raise N.Refused(f"refused: the regenerated log of seed {seed} ({world}) differs from version 8's record in "
                        f"{bad}")
    gap = SD._gap_of(record["gap"])
    sd = SD.SeedView(seed, world, record, lg, gap, spec, lg.physics)
    for k in ("gap", "random"):
        s = sd.removals[k].summary(lg)
        want = record["removals"][k]
        if s["kept_rows_sha256"] != want["kept_rows_sha256"] or s["bank_rows_sha256"] != want["bank_rows_sha256"]:
            raise N.Refused(f"refused: the {k} removal of seed {seed} ({world}) differs from version 8's record")
    return sd, dict(log_sizes=list(sizes), stall_slope=slope, recorder=spec.as_dict(), log_arrays_equal=True,
                    removals_equal=True)


def spent_switch_facts(seed, world, step1):
    """One spent seed-world's facts (module docstring). Nothing trained."""
    import learner_phases as LP
    import settingdata as SD
    from experiments.shift import learners as L
    t0 = time.perf_counter()
    data_rec, data_sha, data_path = _read_v8(V8_DATA_REL.format(world=world, seed=seed))
    sw_rec, sw_sha, sw_path = _read_v8(V8_SWITCH_REL.format(world=world, seed=seed))
    c2_rec, c2_sha, c2_path = _read_v8(V8_CRITIC2_REL.format(world=world, seed=seed))
    sd, data_check = spent_seed_data(seed, world, data_rec)
    learn = SD.learn_settings(type("Run", (), dict(stub=False))())
    pg = SD.prepared("main", sd, learn)
    prep_want = data_rec.get("prepared_sha256") or {}
    prep_equal = {k: bool(pg.sha256[k] == prep_want.get(k)) for k in SD.PREPARED_KEYS}
    if not all(prep_equal.values()):
        raise N.Refused(f"refused: the prepared rows of seed {seed} ({world}) differ from version 8's record: "
                        f"{prep_equal}")
    gr = LP.gap_rule_of(pg)
    if not gr["defined"]:
        raise N.Refused(f"refused: the gap rule is not defined on seed {seed} ({world}); version 8 recorded it defined")
    surv = np.asarray(pg.rows[gr["tail"]], np.int64)
    if step1.array_sha256(surv) != c2_rec.get("survivors_sha256"):
        raise N.Refused(f"refused: the survivors of seed {seed} ({world}) differ from version 8's second critic's")
    critic2 = SD.load_tree_checked(Path(c2_path).with_name("critic_second.msgpack"), L.critic_template(),
                                   c2_rec["file_sha256"])
    if L.tree_sha256(critic2) != c2_rec.get("critic_sha256"):
        raise N.Refused(f"refused: version 8's second critic of seed {seed} ({world}) does not hash as recorded")
    tail, rest, detail = LP.switch_scores_pooled(critic2, sd, pg, gr, learn.gamma)
    entry = LP.certificate_entry(tail, rest, seed, world, gr)
    pooled = switch_calls_of(entry)
    recorded = switch_calls_of(sw_rec["entry"])
    counts = N.pooled_count_record(sd.log, sd.gap, SD.removal("main", sd), step1)
    if (counts["groups"]["n_full_throttle"], counts["groups"]["n_rest"]) != (len(tail), len(rest)):
        raise N.Refused(f"refused: the pooled count and the scored groups differ on seed {seed} ({world})")
    sorted_tail = detail["tail_scores_sorted"]
    c = entry["certificate"]
    r_rest, q90_rest = num(c.get("R_rest")), num(entry["groups"]["q90_rest"])
    ranks = dict(n=len(sorted_tail),
                 bca_threshold=LP.height_rank(sorted_tail, num(c.get("R_tail"))),
                 plain_q90=LP.height_rank(sorted_tail, num(entry["groups"]["q90_full_throttle"])),
                 twice_plain=LP.height_rank(sorted_tail, None if q90_rest is None else q90_rest + pooled["twice_plain_gap"]),
                 twice_bca=LP.height_rank(sorted_tail, None if r_rest is None or pooled["bca_height"] is None
                                          else r_rest + 2.0 * pooled["bca_height"]))
    return dict(kind=SPENT_KIND, rep=int(seed), seed=int(seed), world=world, defined=True,
                sources=dict(data_record=dict(path=data_path, sha256=data_sha),
                             switch_record=dict(path=sw_path, sha256=sw_sha),
                             critic_second_record=dict(path=c2_path, sha256=c2_sha, file_sha256=c2_rec["file_sha256"],
                                                       critic_sha256=c2_rec["critic_sha256"])),
                data_check=dict(data_check, prepared_equal=prep_equal, survivors_equal=True),
                gap_rule=dict(t_lo=float(gr["t_lo"]), t_up=float(gr["t_up"]), n_survivors=int(gr["n_tail"])),
                groups=counts["groups"], certification=counts["certification"],
                version8_groups=counts["version8_groups"],
                pooled=dict(entry=entry, calls=pooled, ranks=ranks,
                            scores_tail_sha256=detail["scores_tail_sha256"],
                            scores_rest_sha256=detail["scores_rest_sha256"]),
                recorded=dict(entry=sw_rec["entry"], calls=recorded, groups_rule=sw_rec.get("groups_rule", "step1")),
                kill_rule=kill_rule(pooled, recorded), seconds=time.perf_counter() - t0)


def spent_summary(rows):
    """The kill rule over every spent seed-world, and the calls side by side."""
    passed = [bool((r.get("kill_rule") or {}).get("passed")) for r in rows]
    return dict(seed_worlds=len(rows), kill_rule_passed_count=sum(passed),
                kill_rule_passed=bool(rows) and all(passed),
                bca_on_pooled=sum(1 for r in rows if r["pooled"]["calls"]["bca"]),
                bca_refused_pooled=sum(1 for r in rows if r["pooled"]["calls"]["bca_refused"]),
                bca_refused_recorded=sum(1 for r in rows if r["recorded"]["calls"]["bca_refused"]),
                meandiff_made_bcas_call_pooled=sum(1 for r in rows if r["pooled"]["calls"]["meandiff"]
                                                   == r["pooled"]["calls"]["bca"]),
                heights_pooled={f"{r['world']}/seed{r['seed']}": r["pooled"]["calls"]["bca_height"] for r in rows},
                heights_recorded={f"{r['world']}/seed{r['seed']}": r["recorded"]["calls"]["bca_height"] for r in rows})


# --------------------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="the counts' seeds (0-4)")
    p.add_argument("--spent-switch", nargs="*", type=int, default=[],
                   help="the spent seeds whose pooled switch is recomputed from version 8's recorded second critics")
    p.add_argument("--worlds", nargs="+", default=["stall", "no_stall"], choices=["stall", "no_stall"])
    a = p.parse_args(argv)
    seeds = N.check_seeds(a.seeds)
    spent = N.check_spent_seeds(a.spent_switch)
    out = N.guard_out(a.out)
    from experiments.shift import acceptance as AC
    t0 = time.perf_counter()
    step1 = N.load_step1lib()
    results = []
    for world in a.worlds:
        for rep in seeds:
            t1 = time.perf_counter()
            sd = AC.load_seed(RUNS, None, "base", world, rep)
            f = seed_world_counts(sd.log, sd.gap, world, sd.removals["gap"], step1)
            f["seconds"] = time.perf_counter() - t1
            results.append(f)
            g = f.get("groups") or {}
            print(f"{world} seed {rep}: pooled full-throttle group {g.get('n_full_throttle')} "
                  f"(version 8's {((f.get('version8_groups') or {}).get('n_full_throttle'))}), rest {g.get('n_rest')}, "
                  f"{f['seconds']:.0f} s", flush=True)
    spent_rows = []
    for world in a.worlds:
        for rep in spent:
            f = spent_switch_facts(rep, world, step1)
            spent_rows.append(f)
            pc, rc = f["pooled"]["calls"], f["recorded"]["calls"]
            print(f"{world} seed {rep} (spent): pooled group {f['groups']['n_full_throttle']} (version 8's "
                  f"{rc['n_full_throttle']}); BCA {rc['bca']} -> {pc['bca']}, height {rc['bca_height']} -> "
                  f"{pc['bca_height']:.3f}; mean difference {rc['meandiff']} -> {pc['meandiff']}; kill rule "
                  f"{'passed' if f['kill_rule']['passed'] else 'FAILED'}; {f['seconds']:.0f} s", flush=True)
    payload = dict(kind=KIND, status=N.STATUS, label=K.LABEL, seeds=seeds, spent_seeds=spent, worlds=a.worlds,
                   results=results, summary=summary(results), spent_results=spent_rows,
                   spent_summary=spent_summary(spent_rows) if spent_rows else None, min_rows=K.CERT_MIN_ROWS,
                   seconds=time.perf_counter() - t0,
                   note="outcome-free counts on seeds 0-4 (commands and split labels only); on the spent seeds 10-14 "
                        "BCA's pooled switch from version 8's recorded second critics; nothing trained, no reward of "
                        "any trained arm read; version 8's outputs read, never written")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf8")
    print(f"wrote {out}")
    return payload


if __name__ == "__main__":
    main()
