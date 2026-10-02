# Read-only tabulation of step-3 results.json (q1 signal) for the step-4 synthesis. No new runs.
import json, numpy as np
d = json.load(open("/mnt/c/Users/David/dev/BCA/runs/wbcp_signal/lq/results.json"))
sett = {(s["behavior"], s["case"], s["kappa"], s["pi_offset"]): s["diagnostics"] for s in d["settings"]}
print("beh case kappa off alignK gap Jnone rho_host_med const-none oracle-const bca-const")
rows = []
for r in d["results"]:
    if r["signal"] != "q1":
        continue
    m = r["metrics"]
    dg = sett[(r["behavior"], r["case"], r["kappa"], r["pi_offset"])]
    gap = dg["J_opt"]["mean"] - dg["J_pi"]["mean"]
    v = lambda k: np.array([x if x is not None else np.nan for x in m[k]["values"]], float)
    rho = np.nanmedian(v("bc_term_grad_norm_none") / v("q_term_grad_norm"))
    cn = np.nanmean(v("J_change_constant") - v("J_change_none"))
    oc = np.nanmean(v("J_change_oracle") - v("J_change_constant"))
    bc = np.nanmean(v("J_change_bca") - v("J_change_constant"))
    a = m["alignment_K"]["mean"]
    jn = m["J_change_none"]["mean"]
    rows.append((r["behavior"], r["case"], r["kappa"], r["pi_offset"], a, gap, jn, rho, cn, oc, bc))
    print(r["behavior"][:4], r["case"][:12].ljust(12), r["kappa"], r["pi_offset"],
          "na" if a is None else "%.3f" % a, "%.3f" % gap, "%.4f" % jn, "%.3f" % rho, "%+.4f" % cn,
          "%+.4f" % oc, "%+.5f" % bc)
a = np.array([x[4] if x[4] is not None else np.nan for x in rows])
print("cells with alignment_K < 0.5:", sum(1 for x in rows if x[4] is not None and x[4] < 0.5))
print("cells with alignment_K < 0.9:", sum(1 for x in rows if x[4] is not None and x[4] < 0.9))
for x in rows:
    if x[4] is not None and x[4] < 0.9:
        print("  ", x[0], x[1], x[2], x[3], "%.3f" % x[4])
rh = np.array([x[7] for x in rows])
print("rho_host range %.3f-%.3f median %.3f" % (rh.min(), rh.max(), np.median(rh)))
g = np.array([x[5] for x in rows])
print("gap range %.3f-%.3f" % (g.min(), g.max()))
