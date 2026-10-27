# Bounded references

The reference solves a constrained quadratic program, not an unconstrained Riccati gain followed by clipping. The cost is the canonical finite objective, sum over t=0,...,H-1 of gamma^t (||s_t||² + 0.3 ||a_t||²), gamma=0.98. There is no extra terminal cost. Therefore the final action is zero.

The no-stall dynamics are linear in the command. In the canonical stall world, any command beyond the knee is dominated: applying its effective thrust as an in-knee command gives the same successor and lower command cost. This holds because the canonical stall curve stays inside the knee's thrust range. The implementation explicitly rejects slopes outside that proven regime. The resulting box [-knee,knee] gives a convex quadratic program for the continuous command relaxation. L-BFGS-B must satisfy the recorded projected-gradient KKT tolerance; an uncertified solve raises an error.

Actual Gymnasium commands are rounded to float32. The reference reports that conversion error and does not claim the exact optimum over the float32 lattice. Shorter planning horizons are receding-horizon references, not full-episode optimal policies. The 100-step planner covers the full remaining horizon, using the same terminal cost convention.

The augmented 13-coordinate plant is not fully controllable: its controllability
rank is 10/13, while the ten-coordinate oscillator subsystem has rank 10/10.
Current and the two wave coordinates are autonomous, unactuated forcing; a
planner may counter their effect but cannot stop or damp them. These ranks refer
to the linear effective-thrust model and do not certify arbitrary reachability
with bounded commands. See the [environment checks](environment.md#controllability-and-exogenous-forcing).

The separate terminal four-arm diagnostic also has a structural limit: every one
of its 6,912 archived arm-level choices selects zero, the known terminal optimum,
with zero regret. It validates mechanics, coverage and refusal behavior, and
cannot establish an uncertainty-guided decision benefit. It is distinct from
the nonterminal planning references described here; its target is not changed
to improve any method's comparison. See [terminal diagnostic limits](coverage_and_calibration.md#four-arms-and-repeated-smoke).

Two information budgets share exactly this planner:

- **Privileged true model:** canonical A and B are provided to the solver.
- **Train-only identified model:** ordinary least squares fits A and B from the first 13 float32 observation coordinates and executed actions in training trajectories only. It does not use exact diagnostic info states. Public state coordinates, reward, discount, actuator map, bounds, and horizon are assumed known. Full feature rank 17 is required. This is a structure-aware system-identification baseline, not a model-free learner. Numerical rank, singular values, conditioning, training residuals and separate offline validation residuals are reported. Synthetic omitted-region and model-misspecification checks distinguish model-class extrapolation from density support.

All matrices and fitted training group IDs are hashed in receipts. Solver workspaces and per-action certificates may change during evaluation; learned matrices, reward model, normalization, and policy selection may not. Random and behavior cloning provide simple reference points. No BCA advantage is assumed.

Run: python -B -m unittest discover -s tests -p test_planning.py -v.

## Saved identification receipt checks

Revision v3 requires the runner's train-only baseline to include its identification
record. The record must name the declared train-only fitting protocol and its
public structural assumptions. Its matrix hash must agree with the actual saved
A/B arrays, its dataset hash with the validated dataset, and its nonempty integer
group IDs with exactly the Train split. The declared full-rank unregularized fit
requires integer rank 17, finite ordered singular values, condition number and
training RMSE consistent with the stored training observations, executed actions
and matrices. Numerical comparisons use relative tolerance 1e-10 and absolute
tolerance 1e-12; identities and role membership are exact.

The two copies of the receipt must agree and meet these contracts. Agreement
alone is insufficient. Requiring a previously missing record strengthens the
acceptance contract; rejecting a nested matrix hash that contradicts the saved
matrix repairs an internal-consistency gap. These checks neither authenticate
that the recorded fitting process executed nor prevent jointly fabricated
artifacts. A valid constructor still does not certify training history. The
original saved model and its training fit passed independent checks and are not
changed by this validation revision.

