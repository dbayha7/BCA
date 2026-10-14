"""Stage 2 v7 evidence: one table from the probe_*.json files run_probes.sh wrote, and a check of each against the
claims in V7_SPEC C1 (written to probe_summary.json in this folder; nothing else is written).

Required (C1 acceptance rule; a failure here means nothing may be pinned and the spec is revised):
  every pinned run, every kernel family: the gate's load_prereg + guard accept (all 64 anchors exact, software
  check on), the 64 anchors compared one by one all equal, and load_seed matches all 10 base gate logs; the
  substitute was asked only for the file's tuples; no caller reached the original functions; no LAPACK call.
  default kernels: the file equals the local values bit for bit (the file makes no difference), and the unpinned
  default run also accepts and matches all 10 logs.
Information (the spec's quoted numbers, compared and reported, never required):
  unpinned anchors differing per family (earlier: Haswell 44, Sandybridge 43, Nehalem 58, Katmai 42), unpinned logs
  failing (earlier: all 10 under Haswell), max |dA| (earlier: 1.7e-18 under Haswell), max |dK| (earlier: at most
  1.04e-14 across the four families).

Usage: python3 summarize_probes.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = ("default", "Haswell", "Sandybridge", "Nehalem", "Katmai")
EARLIER_ANCHORS = {"default": 0, "Haswell": 44, "Sandybridge": 43, "Nehalem": 58, "Katmai": 42}
TUPLES = {"experiments.shift.thruster_chain._matrices": {"(0.3, 0.27)"},
          "experiments.shift.recorder._gain": {"(0.3, 0.27, 0.3, 0.98)"}}

rows, problems, info = [], [], []
for fam in FAMILIES:
    for mode in ("pinned", "unpinned"):
        p = HERE / f"probe_{fam}_{mode}.json"
        if not p.exists():
            problems.append(f"{p.name} missing")
            continue
        d = json.loads(p.read_text(encoding="utf8"))
        env, s = d["environment"], d["local_vs_file_summary"]
        oc = d["original_caches"]
        row = dict(
            run=p.name, family=fam, mode=mode,
            OPENBLAS_CORETYPE=env["OPENBLAS_CORETYPE"], NPY_DISABLE_CPU_FEATURES=env["NPY_DISABLE_CPU_FEATURES"],
            XLA_FLAGS=env["XLA_FLAGS"], gate_software_check=env["gate_software_check"],
            kernels={k: v["kernel"] for k, v in env["openblas"].items()},
            numpy_avx512_dispatch_active=[t for t in env["numpy_dispatch_targets_active"] if t.startswith("AVX512")],
            processors_usable=env["processors_usable"],
            gate_accepts=d["gate_accepts"], gate_error=d.get("gate_error"),
            anchors_differing=d["anchors"]["n_differing"], anchors_n=d["anchors"]["n"],
            anchors_max_abs=d["anchors"]["max_abs"], anchors_max_ulp=d["anchors"]["max_ulp"],
            logs_matched=d["load_seed_summary"]["matched"], logs_run=d["load_seed_summary"]["run"],
            max_abs_dA=s["max_abs_dA"], max_abs_dK=s["max_abs_dK"], B_bit_identical=s["B_bit_identical"],
            A_entries_differing=sum(v["differing"] for k, v in d["local_vs_file"].items() if k.startswith("A ")),
            K_entries_differing=sum(v["differing"] for k, v in d["local_vs_file"].items() if k.startswith("K ")),
            within_v7_tolerances=s["within_v7_tolerances"],
            substitute_tuples={k: sorted(v) for k, v in d["substitute_calls"].items()},
            tripwire_calls=d["tripwire_calls_after_local_recompute"],
            originals_untouched_after_substitution=(oc["matrices_before_gate"] == oc["matrices_end"]
                                                    and oc["gain_before_gate"] == oc["gain_end"]),
            software_state_equals_gate_record=d["software_state_equals_gate_record"],
            seconds_total=round(d["seconds_total"], 1),
            seconds_load_seed=round(sum(v["seconds"] for v in d["load_seed"].values()), 1))
        rows.append(row)
        if mode == "pinned":
            if not (row["gate_accepts"] and row["anchors_differing"] == 0 and row["anchors_n"] == 64):
                problems.append(f"{fam} pinned: gate_accepts={row['gate_accepts']} anchors differing="
                                f"{row['anchors_differing']} of {row['anchors_n']} ({row['gate_error']})")
            if not (row["logs_matched"] == row["logs_run"] == 10):
                problems.append(f"{fam} pinned: load_seed matched {row['logs_matched']} of {row['logs_run']}")
            if any(set(v) - TUPLES.get(k, set()) for k, v in row["substitute_tuples"].items()):
                problems.append(f"{fam} pinned: substitute asked for {row['substitute_tuples']}")
            if not row["originals_untouched_after_substitution"] or any(row["tripwire_calls"].values()):
                problems.append(f"{fam} pinned: a caller reached the original functions or LAPACK")
            if not row["software_state_equals_gate_record"]:
                problems.append(f"{fam} pinned: software state differs from the gate's record")
            if not row["within_v7_tolerances"]:
                problems.append(f"{fam}: A or K beyond v7's tolerances (1e-14, 1e-12) or B differs")
        if fam == "default":
            if row["kernels"] != {"numpy": "SkylakeX", "scipy": "SkylakeX"}:
                problems.append(f"default {mode}: kernels {row['kernels']}")
            if not (row["max_abs_dA"] == 0 and row["max_abs_dK"] == 0 and row["B_bit_identical"]):
                problems.append(f"default {mode}: the file differs from this machine's own values")
            if not (row["gate_accepts"] and row["logs_matched"] == 10):
                problems.append(f"default {mode}: gate or logs failed")
        else:
            if row["kernels"] != {"numpy": fam, "scipy": fam}:
                problems.append(f"{fam} {mode}: kernels reported {row['kernels']} (emulation not in force)")
            if row["numpy_avx512_dispatch_active"]:
                problems.append(f"{fam} {mode}: numpy AVX-512 dispatch still active {row['numpy_avx512_dispatch_active']}")
        if mode == "unpinned":
            if row["anchors_differing"] != EARLIER_ANCHORS[fam]:
                info.append(f"{fam} unpinned: {row['anchors_differing']} anchors differ (earlier probe: "
                            f"{EARLIER_ANCHORS[fam]})")
            if fam == "Haswell" and row["logs_matched"] != 0:
                info.append(f"Haswell unpinned: {row['logs_matched']} of 10 logs matched (spec: all 10 fail)")
families = [r for r in rows if r["family"] != "default"]
maxK = max((r["max_abs_dK"] for r in families), default=None)
hasA = [r["max_abs_dA"] for r in families if r["family"] == "Haswell"]
# The spec's "at most 1.04e-14" rounds the earlier probes' largest value; compare with that value exactly
# (crosscheck_constants.json, by_kernel_family, from scratchpad/v7_hw_probe/core_*.json).
earlier = json.loads((HERE / "crosscheck_constants.json").read_text(encoding="utf8"))["by_kernel_family"]
earlier_maxK = max(v["max_abs_dK"] for k, v in earlier.items() if k in FAMILIES)
if maxK is not None and maxK != earlier_maxK:
    info.append(f"max |dK| across the four families is {maxK!r} (earlier probes: {earlier_maxK!r}; spec: 1.04e-14)")
if hasA and abs(hasA[0] - 1.734723475976807e-18) > 1e-30:
    info.append(f"Haswell max |dA| is {hasA[0]!r} (spec: 1.7e-18)")

out = dict(kind="stage2-v7-evidence-probe-summary", rows=rows, required_failures=problems,
           information_differences=info,
           verdict=("AGREES with V7_SPEC C1: with the pinned constants the unchanged gate accepted all 64 anchors "
                    "exactly and load_seed matched all 10 base gate logs under every kernel family, and with the "
                    "default kernels the file makes no difference" if not problems else
                    "DISAGREES with V7_SPEC C1: see required_failures; nothing may be pinned"))
(HERE / "probe_summary.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf8")
cols = ("family", "mode", "kernels", "gate_accepts", "anchors_differing", "logs_matched", "max_abs_dA", "max_abs_dK",
        "A_entries_differing", "K_entries_differing", "seconds_total")
for r in rows:
    print({c: r[c] for c in cols})
print("required failures:", problems)
print("information:", info)
print(out["verdict"])
