"""Pure paired contrasts. No files, policies, simulator, or own-best benchmark."""
import math


def paired_contrasts(host_anchor, bca_anchor, host_action, bca_action):
    values = (host_anchor, bca_anchor, host_action, bca_action)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        raise ValueError('Four finite real recorded returns required.')
    host_loss = host_anchor - host_action
    bca_loss = bca_anchor - bca_action
    result = dict(host_anchor_return=host_anchor, bca_anchor_return=bca_anchor,
                  host_action_return=host_action, bca_action_return=bca_action,
                  absolute_advantage=bca_action-host_action,
                  anchor_advantage=bca_anchor-host_anchor,
                  host_degradation=host_loss, bca_degradation=bca_loss,
                  degradation_advantage=host_loss-bca_loss)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Contrast overflow.')
    return result


def sequential_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Nonempty finite observations required.')
    total = 0.0
    for value in values:
        total += value
    result = total / len(values)
    if not math.isfinite(result):
        raise ValueError('Mean overflow.')
    return result


def resource_envelope(pairs=20, states=256, slots=14, horizon=250):
    """Fixed v2 maximum, anchored to the separately accepted original snapshot."""
    if (pairs, states, slots, horizon) != (20, 256, 14, 250):
        raise ValueError('Protocol dimensions changed; a new declaration is required.')
    scopes = dict(outcomes=pairs*states*slots*2*horizon,
                  repeats=pairs*states*slots*2, collection=pairs*128*300,
                  engineering=4*10_000)
    additional = sum(scopes.values())
    combined = 1_298_353 + additional
    if combined > 39_998_400:
        raise ValueError('Original cumulative global cap exceeded.')
    return dict(scopes=scopes, old_environment=1_298_353,
                new_environment=additional, combined_environment=combined,
                combined_physics=combined*4, unallocated_environment=39_998_400-combined,
                old_ledger_changes=False, executable_ledger=False)
