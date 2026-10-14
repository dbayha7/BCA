"""Collects the information-only XLA evidence in this folder into summary.json (+ sidecar) and prints the table that
SUMMARY.md reproduces. Reads only files in this folder; writes only summary.json and its sidecar, once.
NOT A v7 RESULT (see SUMMARY.md).

Usage: python summarize_xla.py [--write]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xla_emul_common as X  # noqa: E402

K = X.K


def load(p):
    return K.read_json_verified(p) if Path(str(p) + ".sha256").exists() else json.loads(Path(p).read_text())


def fmt(v):
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "0" if v == 0.0 else f"{v:.2g}"
    return str(v)


def env_of(rec):
    e = rec.get("environment") or {}
    return dict(mode=e.get("mode"), cpu=e.get("cpu_model"), affinity=e.get("affinity"), openblas=e.get("openblas"),
                numpy_avx512=e.get("numpy_avx512_dispatch_active"), xla_flags=(e.get("variables") or {}).get("XLA_FLAGS"),
                confirmed=e.get("mode_confirmed"), glibc=e.get("glibc"), jax=e.get("jax"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows, out = [], dict(kind="stage2-v7-evidence-xla-summary", information_only=X.INFO, repro=[], feasibility=[],
                         forecast=[])
    for p in sorted((HERE / "repro").glob("seed*_*.json")):
        r = load(p)
        if "arms" not in r:
            out["repro"].append(dict(file=p.name, refused=r.get("refused")))
            rows.append((f"P1a stall (xla_repro) seed {r['seed']}", r["mode"], "refused: " + str(r.get("refused")), "", ""))
            continue
        c = (r.get("constants") or {}).get("differences") or {}
        item = dict(file=p.name, seed=r["seed"], mode=r["mode"], arms=r["arms"], all_equal=r["all_equal"],
                    seconds=r["seconds"], environment=env_of(r), constants_local_vs_file=c,
                    substitute_calls=(r.get("constants") or {}).get("substitute_calls"),
                    tripwire_calls=((r.get("constants") or {}).get("tripwire") or {}).get("calls"))
        out["repro"].append(item)
        rows.append((f"P1a stall (xla_repro) seed {r['seed']}", r["mode"],
                     "; ".join(f"{n} {'equal' if v['equal'] else 'differs'} ({fmt(v['max_abs_diff'])})"
                               for n, v in r["arms"].items()),
                     f"{r['seconds']['total']:.0f}", str(item["environment"]["affinity"])))
    for where, folder, pattern in (("review run (processors not recorded)", HERE / "feas_review_seed2" / "cloud_review_run"
                                    / "out", "result_*_seed2.json"),
                                   ("repeat on processor 0", HERE / "feas_review_seed2" / "cpu0_repeat",
                                    "result_*_seed2.json")):
        for p in sorted(folder.glob(pattern)):
            r = load(p)
            mode = r.get("mode")
            if "refused" in r:
                out["feasibility"].append(dict(where=where, file=p.name, refused=r["refused"]))
                rows.append((f"feasibility {mode} seed 2", where, "refused: " + r["refused"], "", ""))
                continue
            if mode == "host":
                acts = {"actor": r["actor"]}
                extra = f"; critic {'equal' if r.get('critic_identical') else 'differs'}"
            else:
                acts = {k: v for k, v in r.items() if isinstance(v, dict) and "identical" in v}
                extra = ""
            out["feasibility"].append(dict(where=where, file=p.name, mode=mode, actors=acts,
                                           seconds=r.get("total_seconds"), affinity=r.get("affinity"),
                                           environment=env_of(r) if "environment" in r else None))
            rows.append((f"feasibility {'P2 host' if mode == 'host' else 'P1a ' + mode} seed 2", where,
                         "; ".join(f"{n} {'equal' if v['identical'] else 'differs'} ({fmt(v['max_abs_diff'])})"
                                   for n, v in acts.items()) + extra,
                         f"{r.get('total_seconds', 0):.0f}", str(r.get("affinity", "not recorded"))))
    for mode in X.MODES:
        d = HERE / "forecast_seed2" / mode
        for p in sorted(d.glob("run_*.json")):
            run = load(p)
            if "refused" in run:
                out["forecast"].append(dict(mode=mode, file=p.name, refused=run["refused"]))
                rows.append((f"forecast {run.get('probe')}", mode, "refused: " + run["refused"], "", ""))
                continue
            rec = load(d / run["record"]["file"])
            item = dict(mode=mode, probe=run["probe"], record=run["record"], seconds=run["seconds"],
                        environment=env_of(run))
            if rec["kind"] == "stage2-v7-probe-p1":
                w = rec["world"]
                key = "gapped" if w == "stall" else "no_stall_gapped"
                gate_sha = rec["p1a"]["actors"][key]["gate_sha256"]
                p1a = {n: dict(equal=not v["differs"], max_abs_diff=v["max_abs_diff"], failed=v["laptop_failed"])
                       for n, v in rec["p1a"]["actors"].items()}
                p1b = dict(equal=rec["p1b"]["reproduces_gate"], max_abs_diff_vs_gate=rec["p1b"]["max_abs_diff_vs_gate"],
                           max_abs_diff_vs_p1a=rec["p1b"]["max_abs_diff_vs_p1a"], finite=rec["p1b"]["finite"])
                p1c = dict(equal=rec["p1c"]["actor_sha256"] == gate_sha,
                           max_abs_diff_vs_gate=rec["p1c"]["max_abs_diff_vs_gate"],
                           max_abs_diff_vs_p1a=rec["p1c"]["max_abs_diff_vs_p1a"], failed=rec["p1c"]["failed"])
                f = rec["forecast"]
                item.update(world=w, p1a=p1a, p1b=p1b, p1c=p1c, forecast=f, probe_seconds=rec["seconds"])
                rows.append((f"P1 seed 2 {w} (P1a, P1b, P1c)", mode,
                             "P1a: " + ", ".join(f"{n} {'equal' if v['equal'] else 'differs'} ({fmt(v['max_abs_diff'])})"
                                                 for n, v in p1a.items())
                             + f"; P1b {'equal' if p1b['equal'] else 'differs'} ({fmt(p1b['max_abs_diff_vs_gate'])})"
                             + f"; P1c {'equal' if p1c['equal'] else 'differs'} ({fmt(p1c['max_abs_diff_vs_gate'])})"
                             + f"; F_ref {fmt(f['F_ref'])} (against {f['F_ref_against']}), F_fresh {fmt(f['F_fresh'])}"
                             + f"; above 1e-5: {', '.join(f['above_tolerance']) or 'none'}",
                             f"{run['seconds']['probe']:.0f}", str(item["environment"]["affinity"])))
            else:
                act, cri = rec["actor"], rec["critic"]
                item.update(actor=act, critic=cri, td3bc_gapped=rec["td3bc_gapped"])
                rows.append(("P2 seed 2 host (plain TD3+BC)", mode,
                             f"actor {'equal' if not act['differs'] else 'differs'} ({fmt(act['max_abs_diff'])}); "
                             f"critic {'equal' if cri['identical'] else 'differs'} ({fmt(cri['max_abs_diff'])}); "
                             f"check 2's td3bc_gapped equals the gate's host actor: "
                             f"{fmt(rec['td3bc_gapped']['equals_gate_host_actor'])}",
                             f"{run['seconds']['probe']:.0f}", str(item["environment"]["affinity"])))
            out["forecast"].append(item)
    out["dump_check"] = []
    for p in sorted((HERE / "dump_check").glob("*.json")):
        r = load(p)
        if "refused" in r:
            out["dump_check"].append(dict(file=p.name, refused=r["refused"]))
            rows.append(("machine code of the actor-only program", r.get("mode"), "refused: " + r["refused"], "", ""))
            continue
        dd = r["dump"]
        out["dump_check"].append(dict(file=p.name, mode=r["mode"], object_files=dd["object_files"],
                                      instructions=dd["instructions"], zmm_instructions=dd["zmm_instructions"],
                                      ymm_instructions=dd["ymm_instructions"], largest_module=dd["largest_module"],
                                      arms_after_one_chunk_sha256=r["arms_after_one_chunk_sha256"]))
        rows.append(("machine code of the actor-only program (seed 2, one 4,000-step chunk)", r["mode"],
                     f"{dd['zmm_instructions']} of {dd['instructions']} instructions use 512-bit registers "
                     f"(the training loop alone: {dd['largest_module']['counts']['zmm_instructions']} of "
                     f"{dd['largest_module']['counts']['instructions']})",
                     f"{r['seconds']:.0f}", str(env_of(r)["affinity"])))
    shas = {d["mode"]: d.get("arms_after_one_chunk_sha256") for d in out["dump_check"] if "mode" in d}
    if len(shas) == 2:
        out["dump_check_same_bits_in_both_modes"] = shas["default"] == shas["emulated"]
    print("| Run | Where / mode | Result (largest parameter gap from the gate's saved actor) | Seconds | Processor |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    if a.write:
        K.write_json_once(HERE / "summary.json", out)


if __name__ == "__main__":
    main()
