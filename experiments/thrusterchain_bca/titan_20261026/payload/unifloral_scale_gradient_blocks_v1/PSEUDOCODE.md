# Frozen diagnostic pseudocode

```text
verify pinned sources, exact trained export and preceding passing result
load actor, twin critic, BCA reference, fitting observations and recorded actions
freeze the previously reported contrast factor f = 31.372587203979492

for b in eight consecutive blocks of 32 fitting rows:
    a = actor_theta(s)
    M_b = mean(abs(Q1(s,a)))
    c_b = stop_gradient(2.5 / (M_b + 1e-7))
    BC_b = mean_rows(mean_action_coordinates((a-recorded_action)^2))
    differentiate mean Q1, mean U, and BC separately with respect to theta
    measure action gradients Q1 and U separately
    verify actual existing candidate g_1 - g_0 = c_b * grad_theta mean U
    verify g_0 = -c_b * grad_theta mean Q1 + grad_theta BC
    verify reference gradients are zero and report live-action gradients
    compare block zero with the existing pinned passing receipt
    for each fixed kappa:
        report -c_b*gQ, c_b*kappa*gU, gBC, sums, norms and angles
        separately report hypothetical c_b*kappa*f*gU
        keep undefined ratios/cosines unavailable, not zero
    retain raw component gradient trees in memory for aggregation

M_all = row-weighted mean of M_b
gQ_all, gU_all, gBC_all = row-weighted means of raw component trees
c_all = stop_gradient(2.5 / (M_all + 1e-7))
derive whole-batch g_kappa = -c_all*gQ_all + c_all*kappa*gU_all + gBC_all
report whole-batch and per-block summaries separately
verify frozen state and source hashes unchanged and forbidden-call count zero
```

No optimizer, simulator, fit/refresh routine or coefficient selection is called. The factor only changes hypothetical penalty strength. Component-gradient aggregation retains vector cancellation; scalar norm/ratio averages cannot replace it. The same local numerical thresholds are retained from the preceding stage.
