"""Pre-outcome candidate arithmetic in the captured native action coordinates.

This module has no simulator/learner dispatch. Applying the D4RL transform twice
would change some float32 coordinates, so sent and applied actions stay separate.
The simulator must receive `sent`, while frozen warnings describe `applied`.
"""
import numpy as np
from experiments.ood.collect import require,sha

SLOTS=['host_reference','bca','nearest_recorded','uniform']+[
    f'perturb_{rho}_{sign:+d}' for rho in (.05,.15,.30) for sign in (-1,1)]


def action_bank(host,bca,nearest,seed,bounds):
    host,bca,nearest=map(np.asarray,(host,bca,nearest))
    require(host.ndim==1 and host.shape==bca.shape==nearest.shape
            and host.dtype==bca.dtype==nearest.dtype==np.float32
            and all(np.isfinite(x).all() and (abs(x)<=1).all() for x in (host,bca,nearest)),
            'Require accepted float32 unit-bound actions; no repair.')
    lo,hi=bounds['native_low'],bounds['native_high']
    require(lo.dtype==hi.dtype==np.float32 and lo.shape==hi.shape==host.shape
            and np.array_equal(lo,-np.ones_like(host)) and np.array_equal(hi,np.ones_like(host)),
            'Only captured unit native bounds are accepted.')
    require(type(seed) is int and 0<=seed<2**32,'Invalid candidate stream.')
    rng=np.random.default_rng(seed)
    uniform=rng.uniform(-1,1,host.shape).astype(np.float32)
    direction=rng.normal(size=host.shape)
    require(np.isfinite(direction).all() and np.square(direction).mean()>0,'Invalid predeclared direction.')
    direction/=np.sqrt(np.square(direction).mean())
    # Preserve ideal declared RMS offsets, and separately save the float32 input
    # rounding used by the accepted actor/simulator contract.
    proposed=np.stack([host,bca,nearest,uniform]+[
        host.astype(np.float64)+sign*rho*direction for rho in (.05,.15,.30) for sign in (-1,1)])
    clipped=np.clip(proposed,-1,1)
    sent=clipped.astype(np.float32)
    applied=np.clip(lo+(sent+1)*.5*(hi-lo),lo,hi)
    alias=[next((j for j in range(i) if np.array_equal(applied[i],applied[j])),i) for i in range(10)]
    return dict(slot_names=SLOTS.copy(),seed=seed,direction=direction,proposed=proposed,
        clipped=clipped,sent=sent,applied=applied,alias=alias,
        preclip_rms=np.sqrt(np.square(proposed-proposed[0]).mean(axis=1)),
        postclip_rms=np.sqrt(np.square(clipped-clipped[0]).mean(axis=1)),
        applied_rms=np.sqrt(np.square(applied.astype(np.float64)-applied[0]).mean(axis=1)),
        clipping_fraction=(proposed!=clipped).mean(axis=1),
        cast_rounding_max_abs=np.abs(clipped-sent.astype(np.float64)).max(axis=1),
        wrapper_rounding_max_abs=np.abs(applied.astype(np.float64)-sent).max(axis=1))


def score_bank(bank,scores,support,threshold,random_seed):
    require(set(scores)=={'width','scale','dose','usable','radius','bayesian_radius','conformal_radius','unit'},
            'Missing frozen BCA components.')
    for name in ('width','scale','dose','usable'):
        require(np.asarray(scores[name]).shape==(10,),'Wrong score shape.')
    require(all(np.isfinite(np.asarray(v)).all() for v in scores.values())
            and (scores['width']>=0).all() and (scores['scale']>0).all()
            and np.asarray(scores['usable']).all(),'Invalid or unavailable BCA scores; stop.')
    require(float(scores['radius'])==max(float(scores['bayesian_radius']),float(scores['conformal_radius'])),
            'Both Bayesian/conformal radii must remain.')
    require(np.asarray(support).shape==(10,) and np.isfinite(support).all() and (support>=0).all(),
            'Invalid support distance.')
    require(threshold is None or np.isfinite(threshold) and threshold>=0,'Invalid support threshold.')
    require(type(random_seed) is int and 0<=random_seed<2**32,'Invalid saved random-score seed.')
    random=np.random.default_rng(random_seed).random(10)
    for i,j in enumerate(bank['alias']):
        if i!=j:
            require(support[i]==support[j] and all(scores[k][i]==scores[k][j]
                    for k in ('width','scale','dose','usable')),'Conflicting scores on identical applied actions.')
            random[i]=random[j]
    return dict(**scores,support=support,support_threshold=threshold,
                support_distant=None if threshold is None else support>threshold,
                random=random,random_seed=random_seed,constant=np.ones(10),
                scored_coordinates='applied native coordinates, unit normalized bounds',
                action_bank_sha256=sha(bank),native_width=None)
