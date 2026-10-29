"""Fixed logged-probe gradient decomposition; no optimizer or simulator dispatch.

Three 256-row batches preserve the actor loss's batch-specific stopped lambda.
Parameter gradients include the actor Jacobian. Per-row action gradients do not
include that Jacobian or the batch mean factor and are labeled separately.
"""
import hashlib
import numpy as np

ROWS = 768
BATCH_SIZE = 256
DISTANCE_FACTOR = 19.29619572569095
COMPONENTS = ('q', 'bc', 'actual_penalty')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vector_summary(component_gradients):
    g = np.asarray(component_gradients, np.float64)
    require(g.ndim == 2 and g.shape[0] == 3 and np.all(np.isfinite(g)),
            'Three finite parameter-gradient vectors required')
    norms = np.sqrt(np.sum(g*g, axis=1))
    cosines = {}
    for i, left in enumerate(COMPONENTS):
        for j, right in enumerate(COMPONENTS):
            cosines[left+'/'+right] = None if norms[i] == 0 or norms[j] == 0 else float(
                np.clip(np.dot(g[i],g[j])/(norms[i]*norms[j]), -1, 1))
    return {'l2_norm':dict(zip(COMPONENTS,map(float,norms))), 'cosine':cosines,
            'combined_l2_norm':float(np.linalg.norm(g.sum(axis=0)))}


def inward_summary(actions, high_load, loss_gradients):
    """Sign of negative-gradient motion toward zero, only outside the hole."""
    a = np.asarray(actions, np.float64)
    high = np.asarray(high_load, bool)
    require(a.ndim == 2 and a.shape[1] == 4 and high.shape == (len(a),), 'Action/stratum mismatch')
    outside = high & (np.abs(a[:,0]) > .25)
    result = {'high_load_rows':int(high.sum()), 'high_load_actor_outside_band_rows':int(outside.sum()),
              'meaning':'Negative loss-gradient action direction only, not an executed optimizer movement'}
    for name, gradient in loss_gradients.items():
        g=np.asarray(gradient,np.float64)
        require(g.shape==a.shape and np.all(np.isfinite(g)), 'Action gradient shape/finite mismatch')
        signed=g[outside,0]*np.sign(a[outside,0])
        result[name]={'inward_rows':int((signed>0).sum()),'outward_rows':int((signed<0).sum()),
                      'zero_rows':int((signed==0).sum()),
                      'mean_signed_inward_loss_gradient':float(signed.mean()) if len(signed) else None}
    return result


def _parameter_vectors(jax, jacobian):
    leaves=jax.tree_util.tree_leaves(jacobian)
    return np.concatenate([np.asarray(x,np.float64).reshape(3,-1) for x in leaves],axis=1)


def audit(adapter, state, source_row_ids, *, anchor_params=None, progress=None):
    """Evaluate a frozen snapshot on exactly 768 condition-retained logged rows.

    The caller must freeze and authenticate source IDs and count one invocation
    at warmup10k or branch15k/20k. No evaluation labels, new RNG draws or fitting
    are used. A callback receives attempted/completed diagnostic dispatch counts.
    """
    import runtime
    context=adapter.context
    jax,jnp=context.native.jax,context.native.jnp
    probe=context.probe
    ids=np.asarray(source_row_ids,np.int64)
    require(ids.shape==(ROWS,) and len(np.unique(ids))==ROWS, 'Exactly768 unique retained probe IDs required')
    require(probe.obs.shape==(ROWS,14) and probe.action.shape==(ROWS,4), 'Frozen probe dimensions differ')
    require(context.args.batch_size==BATCH_SIZE, 'Gradient audit requires native batch size256')
    require(np.all(np.isin(ids,np.asarray(context.data.source_row_ids,np.int64))), 'Probe identity outside retained replay')
    row_digest=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
    require(row_digest==context.data.receipt['gradient_probe_source_row_ids_sha256'], 'Probe source ID digest differs')
    require(adapter.arm in ('native','bca','distance'), 'Unknown actor objective')
    if adapter.arm=='distance':
        require(anchor_params is not None, 'Distance audit needs authenticated frozen fork actor parameters')
    ref=jax.tree_util.tree_map(jax.lax.stop_gradient,state.aux.posterior)
    require(bool(np.asarray(ref.ready)) and bool(np.asarray(context.pipeline.reference_valid(ref))), 'Unready gradient reference')
    require(runtime.tree_digest(ref)==runtime.tree_digest(state.microstep.reference), 'Actor reference differs from audited posterior')
    before=runtime.tree_digest(state)
    agent=state.microstep.agent_state
    actor=agent.actor
    ledger={'probe_action_rows_attempted':0,'probe_action_rows_completed':0,
            'parameter_gradient_batches_attempted':0,'parameter_gradient_batches_completed':0,
            'parameter_gradient_probe_row_uses_attempted':0,'parameter_gradient_probe_row_uses_completed':0,
            'q_action_gradient_rows_attempted':0,'q_action_gradient_rows_completed':0,
            'prospective_bca_action_gradient_rows_attempted':0,'prospective_bca_action_gradient_rows_completed':0,
            'distance_anchor_action_rows_attempted':0,'distance_anchor_action_rows_completed':0,
            'optimizer_updates':0,'calibration_updates':0,'simulator_transitions':0}
    def notify():
        if progress is not None: progress(dict(ledger))
    def components(params,batch,reference):
        loss,metrics=adapter.functions.actor_loss(params,agent,batch,reference)
        parts=jnp.stack((-metrics['q_scale_coefficient']*metrics['q_mean'],
                         metrics['bc_loss'],metrics['actor_penalty_term']))
        return parts,(loss,metrics)
    component_jacobian=jax.jit(jax.jacrev(components,argnums=0,has_aux=True))
    def one_q(obs,action):
        return agent.dual_q.apply_fn(agent.dual_q.params,obs,action)[0]
    q_gradient=jax.jit(jax.vmap(jax.grad(one_q,argnums=1)))
    def one_width(obs,action):
        width,usable,valid=context.pipeline.live_width(context.models,ref,obs[None,:],action[None,:])
        return jnp.where(usable & valid,width[0],jnp.nan)
    width_gradient=jax.jit(jax.vmap(jax.grad(one_width,argnums=1)))
    act=jax.jit(lambda params,obs:actor.apply_fn(params,obs).sample(seed=None))
    batches=[]; vectors=[]; action_rows=[]; q_rows=[]; bc_rows=[]; bca_rows=[]; actual_rows=[]
    try:
        for start in range(0,ROWS,BATCH_SIZE):
            batch=jax.tree_util.tree_map(lambda x:x[start:start+BATCH_SIZE],probe)
            ledger['parameter_gradient_batches_attempted']+=1
            ledger['parameter_gradient_probe_row_uses_attempted']+=BATCH_SIZE; notify()
            jac,(loss,metrics)=runtime.block(jax,component_jacobian(actor.params,batch,ref))
            vector=_parameter_vectors(jax,jac)
            require(np.all(np.isfinite(vector)), 'Nonfinite component parameter gradient')
            ledger['parameter_gradient_batches_completed']+=1
            ledger['parameter_gradient_probe_row_uses_completed']+=BATCH_SIZE
            require(abs(float(np.asarray(loss))-float(np.asarray(-metrics['q_scale_coefficient']*metrics['q_mean']+
                    metrics['bc_loss']+metrics['actor_penalty_term'])))<=2e-6*(1+abs(float(np.asarray(loss)))),
                    'Pinned actor loss decomposition differs')
            coefficient=float(np.asarray(metrics['q_scale_coefficient']))
            require(np.isfinite(coefficient) and coefficient>0, 'Invalid raw-Q normalization coefficient')
            ledger['probe_action_rows_attempted']+=BATCH_SIZE; notify()
            actions=np.asarray(runtime.block(jax,act(actor.params,batch.obs)),np.float32)
            ledger['probe_action_rows_completed']+=BATCH_SIZE
            ledger['q_action_gradient_rows_attempted']+=BATCH_SIZE; notify()
            qg=np.asarray(runtime.block(jax,q_gradient(batch.obs,jnp.asarray(actions))),np.float64)
            ledger['q_action_gradient_rows_completed']+=BATCH_SIZE
            ledger['prospective_bca_action_gradient_rows_attempted']+=BATCH_SIZE; notify()
            ug=np.asarray(runtime.block(jax,width_gradient(batch.obs,jnp.asarray(actions))),np.float64)
            ledger['prospective_bca_action_gradient_rows_completed']+=BATCH_SIZE
            q_loss=-coefficient*qg
            bc_loss=2*(actions.astype(np.float64)-np.asarray(batch.action,np.float64))/4
            prospective=coefficient*3*ug
            if adapter.arm=='native': actual=np.zeros_like(prospective)
            elif adapter.arm=='bca': actual=(adapter.kappa/3)*prospective
            else:
                ledger['distance_anchor_action_rows_attempted']+=BATCH_SIZE; notify()
                anchors=np.asarray(runtime.block(jax,act(anchor_params,batch.obs)),np.float64)
                ledger['distance_anchor_action_rows_completed']+=BATCH_SIZE
                actual=coefficient*adapter.kappa*(2*DISTANCE_FACTOR/4)*(actions.astype(np.float64)-anchors)
            require(all(np.all(np.isfinite(x)) for x in (actions,q_loss,bc_loss,prospective,actual)), 'Nonfinite diagnostic actions/gradients')
            vectors.append(vector)
            batches.append({'start':start,'rows':BATCH_SIZE,'lambda_from_unpenalized_Q1':coefficient,
                'q_mean':float(np.asarray(metrics['q_mean'])),'bc_loss':float(np.asarray(metrics['bc_loss'])),
                'actual_penalty_loss':float(np.asarray(metrics['actor_penalty_term'])),
                'loss':float(np.asarray(loss)),**vector_summary(vector)})
            action_rows.append(actions);q_rows.append(q_loss);bc_rows.append(bc_loss);bca_rows.append(prospective);actual_rows.append(actual)
            notify()
        actions=np.concatenate(action_rows)
        gradients=dict(q=np.concatenate(q_rows),bc=np.concatenate(bc_rows),
                       prospective_bca3=np.concatenate(bca_rows),actual_penalty=np.concatenate(actual_rows))
        observations=np.asarray(probe.obs,np.float32).astype(np.float64)
        high=observations[:,10]+observations[:,11]>=.4
        require(before==runtime.tree_digest(state), 'Gradient audit changed complete training state')
        result={'status':'completed_fixed_logged_gradient_audit','arm':adapter.arm,'actual_kappa':adapter.kappa,
            'source_row_ids_sha256':row_digest,'source_row_ids':ids.tolist(),
            'complete_state_digest_before_after':before,'reference_digest':runtime.tree_digest(ref),
            'probe_rows':ROWS,'batches':batches,'mean_parameter_gradient':vector_summary(np.mean(vectors,axis=0)),
            'inward_action_direction':inward_summary(actions,high,gradients),
            'logged_probe_high_load_mean_a0':float(np.asarray(probe.action)[high,0].mean()) if high.any() else None,
            'scope':'Training-only diagnostic; prospective_bca3 is shadow information in native/distance arms, not their applied penalty',
            'boundary_timing':'Frozen checkpoint objective for the next actor update; at calibration boundaries the preceding actor update used the previous reference',
            'dispatch_accounting':'768 distinct probes; four logical batched forward/autodiff dispatches per row plus one frozen-anchor forward for distance. Autodiff internal primitive executions/FLOPs are not inferred.',
            'action_gradient_convention':'Per-row loss gradients include stopped batch lambda but not1/256 batch mean factor or actor Jacobian',
            'parameter_gradient_convention':'Actual pinned batch-mean actor parameter gradients; mean across three fixed batches',
            'per_row':{'actor_actions':actions.tolist(),'high_load':high.tolist(),
                       'action_loss_gradients':{k:v.tolist() for k,v in gradients.items()}},
            'ledger':ledger,'execution_counts_known':True}
        return result
    except BaseException as exc:
        notify()
        # Caller must preserve this ledger even when its worker terminates.
        exc.gradient_audit_ledger=dict(ledger)
        raise
