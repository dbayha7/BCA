import json
p = json.load(open(r"C:\Users\David\dev\BCA\runs\wbcp_signal\lq\results.json"))
seen = set()
for s in p["settings"]:
    k = (s["behavior"], s["pi_offset"])
    if k in seen: continue
    seen.add(k); d = s["diagnostics"]
    print(k, "J_pi %.3f J_opt %.3f J_beta %.3f gap %.3f" % (d["J_pi"]["mean"], d["J_opt"]["mean"], d["J_beta"]["mean"], d["J_opt"]["mean"]-d["J_pi"]["mean"]))
# host J change and constant-none by case/behaviour/offset (q1)
for r in p["results"]:
    if r["signal"]!="q1": continue
    m=r["metrics"]
    if r["case"] in ("clean","independent_errors") or (r["kappa"]==2.0 and r["case"] in("q1_optimistic","shared_bias")):
        print(r["behavior"], r["case"], r["kappa"], r["pi_offset"], "none %.4f const-none %.4f oracle-const %.4f algK %.2f" % (m["J_change_none"]["mean"], m["J_change_constant"]["mean"]-m["J_change_none"]["mean"], m["J_change_oracle"]["mean"]-m["J_change_constant"]["mean"], m["alignment_K"]["mean"]))
