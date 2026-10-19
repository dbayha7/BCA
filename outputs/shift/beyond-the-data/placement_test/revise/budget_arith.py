# Run-count and compute arithmetic for the revised DESIGN.md (section 9). Made-up arithmetic only; speeds from
# bench/conc_*.out (this container's CPUs) and HOSTS PRESPEC.md:601-604 (laptop card), as quoted in DESIGN.md.
# Run: PYTHONDONTWRITEBYTECODE=1 smokevenv/bin/python -B budget_arith.py > budget_arith.out

CPU_MIN = dict(plain=23.5, bca=27.5, light=10.0, actor_only=6.0, oracle=33.0)  # CPU-minutes per run
# card minutes per run: bare TD3+BC 0.58 ms/step -> ~1.0 min; every C-family run at >= 1.26 ms/step -> ~2.1 min;
# +0.5 min compilation per run; BCA-class runs add 18 refresh tables x ~4 s = 1.2 min; the x64 oracle step ~2x
CARD_MIN = dict(host=1.0 + 0.5, cplain=2.1 + 0.5, bca=2.1 + 0.5 + 1.2, light=0.5, actor_only=1.0, oracle=4.2 + 0.5)
MATCHED = 3 + 1 + 1.5  # S: 3-rung ladder + interpolated run + expected 1.5 more rounds at 1% tolerance (cap 5)


def main_runs(stage, scope="full"):
    """Runs per seed on the main generator: (kind for CPU, kind for card, count, label)."""
    r = [
        ("plain", "host", 4, "hosts S, N, MW+, MW-"),
        ("plain", "host", 2, "A's gated hinge (k = 1) S, N"),
        ("light", "light", 2, "clones S, N"),
        ("oracle", "oracle", 2, "C-ORACLE co-trained (x64 step) S, N"),
        ("bca", "bca", 2, "C-BCA S, N"),
        ("bca", "bca", 8, "k = 1 references C-SC, C-RATIO, C-F0, C-SHUF in S, N"),
        ("plain", "cplain", 3 * (MATCHED + 1), "matched C-HINGE, C-EVEN, C-SUP (S ladder + N lock)"),
        ("bca", "bca", 2 * (MATCHED + 1), "matched C-F0_m, C-SC_m (S ladder + N lock)"),
        ("actor_only", "actor_only", 10, "actor-only C-I2 (3 arms) and two actor-only oracles, S, N"),
    ]
    if scope == "full":
        r.append(("bca", "bca", 2 * (MATCHED + 1), "matched C-RATIO_m, C-SHUF_m (S ladder + N lock)"))
    if stage == "pilot":
        r.append(("plain", "cplain", 2 * (MATCHED + 1), "C-RAND1, C-RAND2 matched (pilot only)"))
    else:
        r.append(("plain", "cplain", 4, "C-RAND1, C-RAND2 at the pilot's locked coefficient"))
    return r


def w_runs():
    return [
        ("plain", "host", 2, "W hosts S, N twin"),
        ("bca", "bca", 2, "W C-BCA weighted S, N"),
        ("oracle", "oracle", 2, "W C-ORACLE S, N"),
        ("bca", "bca", 4 * (MATCHED + 1), "W matched unweighted C-BCA, C-F0_m, C-RATIO_m, C-SC_m"),
        ("plain", "cplain", 3 * (MATCHED + 1), "W matched C-SUP, C-HINGE, C-EVEN"),
    ]


def total(runs):
    n = sum(c for _, _, c, _ in runs)
    cpu = sum(CPU_MIN[k] * c for k, _, c, _ in runs) / 60
    card = sum(CARD_MIN[k] * c for _, k, c, _ in runs) / 60
    return n, cpu, card


for stage in ("pilot", "confirmatory"):
    for scope in ("full", "trimmed"):
        n, cpu, card = total(main_runs(stage, scope))
        print(f"main {stage} {scope}: {n:.1f} runs per seed (actor-only 10 of them at 20,000 steps); "
              f"{cpu:.1f} CPU-h and {card:.2f} card-h per seed; per 5 seeds {5 * cpu:.0f} CPU-h "
              f"(+20%: {6 * cpu:.0f}) or {5 * card:.1f} card-h (+20%: {6 * card:.1f})")
n, cpu, card = total(w_runs())
print(f"module W: {n:.1f} runs per seed; {cpu:.1f} CPU-h and {card:.2f} card-h per seed; per 5 seeds {5 * cpu:.0f} "
      f"CPU-h (+20%: {6 * cpu:.0f}) or {5 * card:.1f} card-h (+20%: {6 * card:.1f})")
for k, _, c, lab in main_runs("pilot"):
    print(f"   main pilot: {c:5.1f}  {lab}")
for k, _, c, lab in w_runs():
    print(f"   W:          {c:5.1f}  {lab}")

# Evaluation per seed (CPU-hours), main generator
ev = dict(
    replicate_blocks=1.0,          # ~500,000 recorded-law episodes per seed-world, 4 worlds, vectorized simulate
    a_replicates=0.7,              # 1,000 replicates x 4 worlds x ~12 rules x ~35 ms, shuffles, sweeps, size table
    ladder_decisions=1.4,          # 12 sigmas x 1,000 replicates x ~12 rules x ~35 ms (confirmatory too)
    staged_b=0.7,                  # ~22 staged decisions x ~0.03 CPU-h with replicate probe sets
    online_scoring=0.4,            # ~70 deployed policies x 4 situations x 2 banks x ~2 s, plus monitors
    c_v1_refresh_bands=0.5,        # 7 refreshes x 2 network states x 50,000 draws x 2 worlds (hand-off)
    c_v3_replicate_banks=3.2,      # 200 banks x 7 refreshes x 2 worlds x 2 bands (BCA, RATIO) x ~4 s per table
    truths_and_attribution=0.5,    # Q^pi pairs, attribution, B-V2 residuals
)
e = sum(ev.values())
print(f"evaluation per seed (main): {e:.1f} CPU-h; per 5 seeds {5 * e:.0f} CPU-h; items {ev}")
print("module W evaluation per seed: about 3 CPU-h (B gates and staged decisions, C-V1/C-V3 on W's bands)")
