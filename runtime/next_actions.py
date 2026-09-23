"""Data."""

import numpy as onp


def qlearning_dataset_with_next_actions(env, dataset=None, terminate_on_end=False):
    if dataset is None:
        dataset = env.get_dataset()
    N = dataset["rewards"].shape[0]
    obs_, next_obs_, action_, next_action_, reward_, done_ = ([], [], [], [], [], [])
    use_timeouts = "timeouts" in dataset
    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(onp.float32)
        new_obs = dataset["observations"][i + 1].astype(onp.float32)
        action = dataset["actions"][i].astype(onp.float32)
        new_action = dataset["actions"][i + 1].astype(onp.float32)
        reward = dataset["rewards"][i].astype(onp.float32)
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if not terminate_on_end and final_timestep:
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1
    return {
        "observations": onp.array(obs_),
        "actions": onp.array(action_),
        "next_observations": onp.array(next_obs_),
        "next_actions": onp.array(next_action_),
        "rewards": onp.array(reward_),
        "terminals": onp.array(done_),
    }
