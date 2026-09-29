# Preserved first synthetic attempt

This directory preserves the original validation and two independent-reader failures. The conformal reference operated on its supplied weights, but the synthetic runner constructed the nominal easy ratio 0.25 as 0.24999999999999994 through floating subtraction. Exact quantile-boundary effects were detected, so these are not the final exact-ratio results. See `numerical_correction.json`, immutable `source_snapshot/`, and the separately verified `../2026-09-29-v2/` results. No RL model, update, or simulator step occurred.
