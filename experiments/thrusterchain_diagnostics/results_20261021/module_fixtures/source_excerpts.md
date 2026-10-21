# Actual source excerpts

Source files are read from the recorded repository; line numbers and SHA-256 are recorded.

## experiments/shift_iw/world.py:128-135

SHA-256: `7bd80da0b4430b305490ae8dd2dde90f9a08cec7a3a633d30ae5d3cc92088a48`

```python
 128: def ratio(c, tilt):
 129:     """Exact w(c) = p_test(c) / p_train(c) of a tilt ('none', 'mild', 'strong'), closed form:
 130:     (M(-2) / M(lam)) exp((lam + 2) c). It is 1 everywhere under 'none'."""
 131:     lam = law_lambda(tilt)
 132:     c = np.asarray(c, np.float64)
 133:     if lam == LAMBDA_TRAIN:
 134:         return np.ones_like(c)
 135:     return (normaliser(LAMBDA_TRAIN) / normaliser(lam)) * np.exp((lam - LAMBDA_TRAIN) * c)
```

## experiments/shift_iw/classifier.py:42-43

SHA-256: `3e107564d08e7aec64a6b41952656a26493ac32abb1e4e1127654897b554fe41`

```python
  42:     def log_weight(self, obs):
  43:         return self.log_odds(obs) + math.log(self.n0 / self.n1)
```

## calibration/wbcp.py:57-117

SHA-256: `fdd36b42a1d2c440cfa5e4df4b59c421c8fca477f8b3ffc5e7d9f095e889c023`

```python
  57: def calibrate(scores, rng, weights=None, test_mass=None, *, alpha=0.1, beta=0.95, draws=1000):
  58:     """Run Algorithm 1 and return the deployed threshold with its posterior.
  59: 
  60:     `rng` is a numpy Generator; no randomness is drawn implicitly. Omitting
  61:     `weights` is the exchangeable case (all weights and the test mass equal 1).
  62:     """
  63:     scores = np.asarray(scores, dtype=np.float64)
  64:     if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
  65:         raise ValueError("scores must be a nonempty finite vector")
  66:     if weights is None:
  67:         if test_mass is not None:
  68:             raise ValueError("test_mass requires explicit calibration weights")
  69:         weights, test_mass = np.ones_like(scores), 1.0
  70:     else:
  71:         weights = np.asarray(weights, dtype=np.float64)
  72:         if weights.shape != scores.shape or not np.all(np.isfinite(weights) & (weights >= 0)):
  73:             raise ValueError("weights must be finite, nonnegative and aligned with scores")
  74:         if not weights.max() > 0:
  75:             raise ValueError("calibration weights need positive total mass")
  76:         if test_mass is None or not (math.isfinite(test_mass) and test_mass > 0):
  77:             raise ValueError("weighted calibration requires a finite positive test_mass")
  78:     if not (0 < alpha < 1 and 0 < beta < 1):
  79:         raise ValueError("alpha and beta must lie strictly between zero and one")
  80:     if not isinstance(rng, np.random.Generator) or not isinstance(draws, (int, np.integer)) or draws < 1:
  81:         raise ValueError("rng must be a numpy Generator and draws a positive integer")
  82: 
  83:     # Only weight ratios matter; rescaling to max 1 keeps every running sum finite.
  84:     scale = weights.max()
  85:     weights, test_mass = weights / scale, test_mass / scale
  86:     order = np.argsort(scores, kind="stable")
  87:     sorted_scores, sorted_weights = scores[order], weights[order]
  88: 
  89:     # Eq. (1): smallest score whose weighted empirical miscoverage is at most alpha.
  90:     # The float index is settled in exact rationals, since R_w = alpha counts.
  91:     cumulative = np.cumsum(sorted_weights)
  92:     k = int(np.argmax(cumulative >= (1.0 - alpha) * cumulative[-1]))
  93:     exact = [Fraction(w) for w in sorted_weights]
  94:     target = (1 - Fraction(str(alpha))) * sum(exact)
  95:     covered = sum(exact[:k + 1])
  96:     while k > 0 and covered - exact[k] >= target:
  97:         covered -= exact[k]
  98:         k -= 1
  99:     while covered < target:
 100:         k += 1
 101:         covered += exact[k]
 102:     lambda_hat = float(sorted_scores[k])
 103: 
 104:     n = scores.size
 105:     chunk = max(1, _CHUNK_ELEMENTS // (n + 1))
 106:     posterior = np.concatenate([
 107:         crossings(sorted_scores, sorted_weights, test_mass,
 108:                   rng.standard_exponential((min(chunk, draws - start), n + 1)), alpha)
 109:         for start in range(0, draws, chunk)
 110:     ])
 111:     posterior.sort()
 112: 
 113:     # Eq. (7): Pr(L+ <= alpha) at lambda is the fraction of draws crossing at or
 114:     # below lambda, so the smallest beta-credible grid point is an order statistic.
 115:     lambda_hpd = float(posterior[math.ceil(Fraction(str(beta)) * int(draws)) - 1])
 116:     n_eff = float(weights.sum() ** 2 / np.dot(weights, weights))  # weights rescaled above
 117:     return Calibration(max(lambda_hat, lambda_hpd), lambda_hat, lambda_hpd, posterior, n_eff)
```

## experiments/shift_iw/world.py:160-164

SHA-256: `7bd80da0b4430b305490ae8dd2dde90f9a08cec7a3a633d30ae5d3cc92088a48`

```python
 160: def exact_ratio(c, actions, gap, tilt, Z):
 161:     """The exact density ratio of a test step to a kept training row: Z w(c) for a command outside the ball, +inf
 162:     for a command inside it."""
 163:     w = Z * ratio(c, tilt)
 164:     return np.where(gap.contains(actions), np.inf, w)
```

## experiments/shift_iw/world.py:148-151

SHA-256: `7bd80da0b4430b305490ae8dd2dde90f9a08cec7a3a633d30ae5d3cc92088a48`

```python
 148: def full_test_mass(tilt):
 149:     """No removal: the test mass E_test[w] = E_train[w^2] = M(-2) M(2 lam + 2) / M(lam)^2 (1, 1.381, 2.841)."""
 150:     lam = law_lambda(tilt)
 151:     return normaliser(LAMBDA_TRAIN) * normaliser(2 * lam - LAMBDA_TRAIN) / normaliser(lam) ** 2
```

## experiments/shift_hosts/iw.py:94-98

SHA-256: `e076537f8d8b217c1c751269901d57e7e9f82b88d566624930c8bf35c4c26784`

```python
  94: def calibrate(scores, stream, weights=None, test_mass=None, *, settings=SETTINGS):
  95:     """wbcp.calibrate with a fresh generator from `stream` (coverage.py:59-62); weights None is the unweighted
  96:     (exchangeable) case. Miscoverage, credibility and draws from bca_core's settings (0.1, 0.95, 1,000)."""
  97:     return wbcp.calibrate(np.asarray(scores, np.float64), generator(stream), weights, test_mass,
  98:                           alpha=settings.miscoverage, beta=settings.credibility, draws=settings.draws)
```

## experiments/shift_hosts/iw.py:101-104

SHA-256: `e076537f8d8b217c1c751269901d57e7e9f82b88d566624930c8bf35c4c26784`

```python
 101: def kish(weights):
 102:     """Kish effective size (sum w)^2 / sum w^2 (coverage.py:94-97)."""
 103:     w = np.asarray(weights, np.float64)
 104:     return float(w.sum() ** 2 / np.dot(w, w))
```

## experiments/shift_iw/classifier.py:45-46

SHA-256: `3e107564d08e7aec64a6b41952656a26493ac32abb1e4e1127654897b554fe41`

```python
  45:     def weight(self, obs):
  46:         return np.exp(self.log_weight(obs))
```

## calibration/reference.py:152-194

SHA-256: `f26273784c458f058c9ebd9b4ebe1e3e7e20c1fc05abfc3297aead724e3b6b1b`

```python
 152: def freeze_reference(cal_params, residual_scale, predictions, residuals, key, config):
 153:     """Score held-out residuals with the frozen scale and select the WBCP threshold.
 154: 
 155:     Scores are |y - q| / (max(eta, 1e-6) u) with unit weights, the exchangeable case
 156:     (BQ-CP with the test atom). Runs eagerly on the host at a refresh. Returns the
 157:     reference, whether its inputs were valid, and scalar diagnostics; an invalid
 158:     refresh returns valid=False and leaves the caller to keep its previous state.
 159:     """
 160:     residual_scale = jnp.asarray(residual_scale)
 161:     predictions, residuals = jnp.asarray(predictions), jnp.asarray(residuals)
 162:     if residual_scale.ndim != 0 or predictions.ndim != 1 or predictions.shape != residuals.shape:
 163:         raise ValueError("expected aligned prediction/residual vectors and a scalar residual unit")
 164:     scale = np.asarray(positive_scale(predictions, residual_scale), np.float64)
 165:     residuals = np.asarray(residuals, np.float64)
 166:     valid = bool(
 167:         tree_valid(cal_params)
 168:         and np.isfinite(float(residual_scale)) and float(residual_scale) > 0
 169:         and predictions.size > 0
 170:         and np.all(np.isfinite(scale) & (scale > 0))
 171:         and np.all(np.isfinite(residuals))
 172:     )
 173:     if not valid:
 174:         return initial_reference(cal_params), False, dict(certified=False)
 175:     scores = np.abs(residuals) / scale
 176:     result = wbcp.calibrate(
 177:         scores, np.random.default_rng(np.asarray(key, np.uint32).ravel()),
 178:         alpha=config.alpha, beta=config.credibility, draws=config.draws,
 179:     )
 180:     with np.errstate(over="ignore"):
 181:         stored = np.float32(result.threshold)
 182:     if result.certified and not np.isfinite(stored):  # a finite threshold must stay finite when stored
 183:         return initial_reference(cal_params), False, dict(certified=False)
 184:     as32 = lambda value: jnp.asarray(value, jnp.float32)
 185:     reference = FrozenReference(
 186:         cal_params, residual_scale.astype(jnp.float32), as32(result.threshold),
 187:         as32(result.lambda_hat), as32(result.lambda_hpd), as32(result.n_eff), jnp.asarray(True),
 188:     )
 189:     diagnostics = dict(
 190:         certified=result.certified, threshold=result.threshold, lambda_hat=result.lambda_hat,
 191:         lambda_hpd=result.lambda_hpd, sigma_post=result.sigma_post, n_eff=result.n_eff,
 192:         clamp_binds=result.lambda_hat > result.lambda_hpd, scores=int(scores.size),
 193:     )
 194:     return reference, True, diagnostics
```

## calibration/reference.py:148-149

SHA-256: `f26273784c458f058c9ebd9b4ebe1e3e7e20c1fc05abfc3297aead724e3b6b1b`

```python
 148: def positive_scale(predictions, residual_scale):
 149:     return jnp.maximum(predictions, 1e-06) * residual_scale
```

## calibration/wbcp.py:43-54

SHA-256: `fdd36b42a1d2c440cfa5e4df4b59c421c8fca477f8b3ffc5e7d9f095e889c023`

```python
  43: def crossings(sorted_scores, sorted_weights, test_mass, exponentials, alpha):
  44:     """Algorithm 1, lines 3-4: each draw's alpha-crossing on the score grid.
  45: 
  46:     Row m of `exponentials` holds E_1..E_n, aligned with the sorted scores, then
  47:     E_{n+1} for the test atom. L+(lambda) <= alpha exactly when the posterior mass
  48:     of scores <= lambda reaches 1-alpha of the total, the test atom included; a
  49:     draw whose test atom alone exceeds alpha never crosses (+inf).
  50:     """
  51:     mass = exponentials[:, :-1] * sorted_weights
  52:     total = mass.sum(axis=1) + exponentials[:, -1] * test_mass
  53:     covered = np.cumsum(mass, axis=1) >= (1.0 - alpha) * total[:, None]
  54:     return np.where(covered.any(axis=1), sorted_scores[covered.argmax(axis=1)], np.inf)
```

## experiments/shift_hosts/iw.py:157-170

SHA-256: `e076537f8d8b217c1c751269901d57e7e9f82b88d566624930c8bf35c4c26784`

```python
 157: def threshold_table(scores, stream, weights, grid, *, settings=SETTINGS):
 158:     """R(t) = calibrate(scores, stream, weights, t).threshold at every grid mass t, each call with a fresh generator
 159:     from the same stream, so every t sees the same posterior draws (coverage.py:173-186). Checks that R never falls
 160:     along the grid (monotone) and takes the running maximum if it ever does (running_max_applied). Returns a Table."""
 161:     grid = np.asarray(grid, np.float64)
 162:     if grid.ndim != 1 or not grid.size or not np.all(np.isfinite(grid) & (grid > 0)) or np.any(grid[1:] <= grid[:-1]):
 163:         raise ValueError("the grid must be an ascending vector of finite positive test masses")
 164:     weights = np.asarray(weights, np.float64)
 165:     cals = [calibrate(scores, stream, weights, float(t), settings=settings) for t in grid]
 166:     raw = np.array([c.threshold for c in cals], np.float64)
 167:     monotone = bool(np.all(raw[1:] >= raw[:-1]))
 168:     used = np.maximum.accumulate(raw)
 169:     return Table(grid, raw, used, float(cals[0].lambda_hat), float(cals[0].n_eff), monotone,
 170:                  bool(np.any(used != raw)))
```

## experiments/shift_hosts/iw.py:180-189

SHA-256: `e076537f8d8b217c1c751269901d57e7e9f82b88d566624930c8bf35c4c26784`

```python
 180: def lookup(table, masses):
 181:     """Per row: the threshold at the smallest grid mass at least the row's own test mass (conservative, since R does
 182:     not fall; coverage.py:196-205). Refuses a mass beyond the grid. Returns (R per row float64, grid index per row)."""
 183:     m = np.asarray(masses, np.float64)
 184:     if not np.all(np.isfinite(m) & (m > 0)):
 185:         raise ValueError("test masses must be finite and positive")
 186:     idx = np.searchsorted(table.grid, m, side="left")
 187:     if np.any(idx >= len(table.grid)):
 188:         raise ValueError("a test mass lies beyond the grid; build the grid on every mass that is looked up")
 189:     return table.thresholds[idx], idx
```

## calibration/dose.py:65-95

SHA-256: `6d2fab35cb772952103adeeb6e9ca5f0a17040c0bc05217e5704c278ad617470`

```python
  65: def frozen_level_dose(reference, predictions, blend):
  66:     """Width = WBCP threshold x frozen scale at the recorded rows; not ready means width 0.
  67: 
  68:     A threshold of +inf (nothing certifiable) gives infinite widths, which
  69:     level_critic_dose treats as unsupported rows with the native multiplier 1.
  70:     """
  71:     predictions = jnp.asarray(predictions)
  72:     if predictions.ndim != 1 or reference.threshold.ndim != 0 or reference.residual_scale.ndim != 0:
  73:         raise ValueError("level dose requires vector predictions, one threshold and a scalar unit")
  74:     threshold, unit = reference.threshold, reference.residual_scale
  75:     scale = jnp.maximum(predictions, 1e-06) * unit
  76:     width = threshold * scale
  77:     valid = (
  78:         tree_finite(reference.cal_params)
  79:         & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
  80:         & jnp.isfinite(unit)
  81:         & (unit > 0)
  82:         & jnp.all(jnp.isfinite(scale) & (scale > 0))
  83:         & ~jnp.isnan(threshold)
  84:         & (threshold >= 0)
  85:         & (
  86:             jnp.isposinf(threshold)
  87:             | jnp.all(jnp.isfinite(width) & ((threshold == 0) | (width > 0)))
  88:         )
  89:     )
  90:     width = jnp.where(reference.ready, width, jnp.zeros_like(width))
  91:     valid = jnp.where(reference.ready, valid, True)
  92:     unit = jnp.where(reference.ready, unit, 1.0)
  93:     return level_critic_dose(
  94:         width, jnp.ones_like(width, dtype=bool), unit, blend, valid
  95:     )
```

## calibration/dose.py:23-62

SHA-256: `6d2fab35cb772952103adeeb6e9ca5f0a17040c0bc05217e5704c278ad617470`

```python
  23: def level_critic_dose(widths, usable, residual_scale, blend, inputs_valid=True):
  24:     widths, usable = (jnp.asarray(widths), jnp.asarray(usable))
  25:     unit, blend, upstream = map(jnp.asarray, (residual_scale, blend, inputs_valid))
  26:     if widths.ndim != 1 or usable.shape != widths.shape:
  27:         raise ValueError("widths and usable must be aligned vectors")
  28:     if unit.ndim or blend.ndim or upstream.ndim:
  29:         raise ValueError("residual_scale, blend and inputs_valid must be scalars")
  30:     if usable.dtype != jnp.bool_ or upstream.dtype != jnp.bool_:
  31:         raise TypeError("usable and inputs_valid must be boolean")
  32:     for value in (widths, unit, blend):
  33:         if not (
  34:             jnp.issubdtype(value.dtype, jnp.floating)
  35:             or jnp.issubdtype(value.dtype, jnp.integer)
  36:         ):
  37:             raise TypeError(
  38:                 "widths, residual_scale and blend must be real numeric values"
  39:             )
  40:     dtype = jnp.result_type(widths, unit, blend, jnp.float32)
  41:     widths, unit, blend = (
  42:         jax.lax.stop_gradient(v.astype(dtype)) for v in (widths, unit, blend)
  43:     )
  44:     valid_unit = jnp.isfinite(unit) & (unit > 0)
  45:     valid_blend = jnp.isfinite(blend) & (blend >= 0) & (blend <= 1)
  46:     valid = (
  47:         upstream
  48:         & jnp.all(~jnp.isnan(widths) & (widths >= 0))
  49:         & valid_unit
  50:         & valid_blend
  51:     )
  52:     support = valid & usable & jnp.isfinite(widths)
  53:     safe_u, safe_c = (jnp.where(support, widths, 0.0), jnp.where(valid_unit, unit, 1.0))
  54:     normalizer = jnp.maximum(safe_u, safe_c)
  55:     rescale = jnp.where(normalizer > jnp.finfo(dtype).max / 4.0, 0.25, 1.0)
  56:     denominator_scale = normalizer * rescale
  57:     u_norm = safe_u * rescale / denominator_scale
  58:     c_norm = safe_c * rescale / denominator_scale
  59:     ratio = u_norm / (u_norm + c_norm)
  60:     dose = 1.0 + jnp.where(valid_blend, blend, 0.0) * ratio
  61:     result = CQLLevelDose(widths, dose, support, valid, jnp.any(support))
  62:     return jax.tree_util.tree_map(jax.lax.stop_gradient, result)
```

## experiments/shift_hosts/iw.py:279-312

SHA-256: `e076537f8d8b217c1c751269901d57e7e9f82b88d566624930c8bf35c4c26784`

```python
 279: def per_row_dose(reference, eta, R_rows, blend, most_caution):
 280:     """BCA's frozen-width dose with each row's own threshold (learners_iw.py:655-683): width_i = R_i x max(eta_i, 1e-6)
 281:     x u with frozen_level_dose's validity checks taken row by row, then dose_i = 1 + blend U_i / (U_i + u) by
 282:     calibration.dose.level_critic_dose (BCA's own arithmetic) for a finite R_i. Before the first refresh (reference not
 283:     ready) every width is 0 and every dose exactly 1. A refused row (reference ready, R_i = +inf) gets, by the carried
 284:     refusal rule most_caution (a traced boolean, so standard and IW BCA compile one program):
 285:       False   dose 1, standard BCA's rule (an infinite width is unsupported; calibration/dose.py:65-95)
 286:       True    dose 1 + blend = 1.5, the most caution: the weighting test's own rule for TD3+BC (lines 681-682), and the
 287:               new rule for CQL and ReBRAC (choice 16)
 288:     With every R_i equal to the reference's threshold this is frozen_level_dose (and the ports' bc_readout, which adds
 289:     reference_valid to inputs_valid) bit for bit. eta: the scale network's prediction with the frozen reference's
 290:     parameters at the rows (learners_iw.py:664 computes it inside). Traceable. Returns (CQLLevelDose, refused rows)."""
 291:     ref = reference
 292:     predictions = jnp.asarray(eta)                       # declared change (1): eta is passed in (line 664)
 293:     R = jnp.asarray(R_rows, jnp.float32)
 294:     if predictions.ndim != 1 or R.shape != predictions.shape:
 295:         raise ValueError("per-row doses need one threshold per row")
 296:     caution = _flag(most_caution)                        # declared change (2): the carried refusal rule
 297:     unit = ref.residual_scale
 298:     scale = jnp.maximum(predictions, 1e-06) * unit
 299:     width = R * scale
 300:     valid = (tree_finite(ref.cal_params)
 301:              & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
 302:              & jnp.isfinite(unit) & (unit > 0)
 303:              & jnp.all(jnp.isfinite(scale) & (scale > 0))
 304:              & jnp.all(~jnp.isnan(R) & (R >= 0))
 305:              & jnp.all(jnp.isposinf(R) | (jnp.isfinite(width) & ((R == 0) | (width > 0)))))
 306:     width = jnp.where(ref.ready, width, jnp.zeros_like(width))
 307:     valid = jnp.where(ref.ready, valid, True)
 308:     unit = jnp.where(ref.ready, unit, 1.0)
 309:     base = level_critic_dose(width, jnp.ones_like(width, dtype=bool), unit, blend, valid)
 310:     refused = ref.ready & jnp.isposinf(R)
 311:     dose = jnp.where(refused & caution, jnp.asarray(1.0 + blend, base.dose.dtype), base.dose)   # (2): `& caution`
 312:     return base._replace(dose=dose, inputs_valid=base.inputs_valid & reference_valid(ref)), refused
```

## experiments/shift_hosts/iw.py:315-364

SHA-256: `e076537f8d8b217c1c751269901d57e7e9f82b88d566624930c8bf35c4c26784`

```python
 315: def per_row_iql_weights(reference, eta, R_rows, adv, beta, cap, most_caution):
 316:     """IQL's BCA actor weights with each row's own threshold (new, plan sections 4.2-4.3). The band's width and support
 317:     are the IQL port's own (calibration/iql_reference.py:157-181, posterior_width and weights_at_reference) with R_i in
 318:     place of the one threshold, and its validity checks taken row by row as in per_row_dose:
 319:       width_i = R_i x max(eta_i, 1e-6) x u; a row is supported when its width is finite and the inputs are valid.
 320:     Weights (beta = Unifloral's args.beta, cap = args.exp_adv_clip):
 321:       before the first refresh       Unifloral's own weight jnp.exp(adv * beta).clip(max=cap) (iql.py:245) on every
 322:                                      row, bit for bit, so a band that is not ready leaves Unifloral's step unchanged
 323:       supported, advantage > 0       BCA's post-cap shrinkage 1 + A / (A + U) x (min(e^(beta A), cap) - 1)
 324:                                      (calibration/advantage.py:32-70, postcap_level_actor_weights; decision gain 1)
 325:       supported, advantage <= 0      Unifloral's own weight (equal to the port's capped weight there)
 326:       refused (ready, R_i = +inf)    most_caution False: weight 0 and unsupported, standard BCA's rule (the port's
 327:                                      unsupported row; when no row is supported, has_support is False and the step skips
 328:                                      its actor update, calibration/iql_state.py:45-79). most_caution True: weight 1
 329:                                      where A > 0 (plain cloning, the limit of the shrinkage as U grows without bound)
 330:                                      and Unifloral's own weight elsewhere, and the row is supported (a new rule, choice
 331:                                      16)
 332:     With every R_i equal to the reference's threshold and most_caution False this equals the port's
 333:     weights_at_reference bit for bit. used_advantage is the port's, except on refused rows under most_caution (0 where
 334:     A > 0, A elsewhere). Traceable. Returns (calibration.advantage.LevelActorWeights, refused rows)."""
 335:     ref = reference
 336:     predictions = jnp.asarray(eta)
 337:     R = jnp.asarray(R_rows, jnp.float32)
 338:     adv = jnp.asarray(adv)
 339:     if predictions.ndim != 1 or R.shape != predictions.shape or adv.shape != predictions.shape:
 340:         raise ValueError("per-row actor weights need one threshold and one advantage per row")
 341:     caution = _flag(most_caution)
 342:     unit = ref.residual_scale
 343:     scale = positive_scale(predictions, unit)            # iql_reference.posterior_width, lines 159-171, row by row
 344:     valid = (reference_valid(ref)
 345:              & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
 346:              & jnp.all(jnp.isfinite(scale) & (scale > 0))
 347:              & jnp.all(~jnp.isnan(R) & (R >= 0)))
 348:     width = R * scale
 349:     valid = valid & jnp.all(jnp.isposinf(R) | (jnp.isfinite(width) & ((R == 0) | (width > 0))))
 350:     usable = jnp.isfinite(width) & valid
 351:     width = jnp.where(ref.ready, width, 0.0)              # iql_reference.weights_at_reference, lines 174-181
 352:     usable = jnp.where(ref.ready, usable, True)
 353:     valid = jnp.where(ref.ready, valid, True)
 354:     base = postcap_level_actor_weights(adv, width, usable, beta, cap, valid)
 355:     plain = jnp.exp(adv * beta).clip(max=cap)             # Unifloral's own weight, iql.py:245
 356:     positive = adv > 0.0
 357:     refused = ref.ready & jnp.isposinf(R)
 358:     most = refused & caution & base.inputs_valid          # refused rows under IW's rule
 359:     certified = jnp.where(positive, base.weights, jnp.where(base.support_mask, plain, 0.0))
 360:     weights = jnp.where(ref.ready, jnp.where(most, jnp.where(positive, 1.0, plain), certified), plain)
 361:     support = base.support_mask | most
 362:     used = jnp.where(most, jnp.where(positive, 0.0, adv), base.used_advantage)
 363:     out = LevelActorWeights(weights.astype(base.weights.dtype), used, support, base.inputs_valid, jnp.any(support))
 364:     return jax.tree_util.tree_map(jax.lax.stop_gradient, out), refused
```
