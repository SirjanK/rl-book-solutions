from multi_armed_bandits.experiment_config import ExperimentConfig
from multi_armed_bandits.make_bandit_algorithm import make_bandit_algorithm
from typing import Dict
import numpy as np


def run_experiment(expt_config: ExperimentConfig) -> Dict[str, np.ndarray]:
    """
    Run experiment given the expt_config and return a dictionary of:
      1. rewards: numpy array for each time step of reward values
      2. is_optimal: numpy array on whether the optimal action was selected
    """

    # validation
    if not expt_config.is_stationary:
        assert expt_config.random_walk_std is not None
    
    # initialize reward means
    if expt_config.is_stationary:
        reward_means = np.random.standard_normal(size=(expt_config.k,))  # set true reward means
    else:
        reward_means = np.zeros(shape=(expt_config.k,), dtype=np.float32)  # initial reward means are zeros
    
    reward_means_max = np.max(reward_means)

    # intitialize algorithm
    algo = make_bandit_algorithm(expt_config, expt_config.bandit_algorithm_config)
    
    # result containers
    rewards = np.zeros(shape=(expt_config.num_time_steps,), dtype=np.float32)
    is_optimals = np.zeros(shape=(expt_config.num_time_steps,), dtype=np.bool)
    for time_step in range(expt_config.num_time_steps):
        # get action
        action = algo.act()

        # compute reward
        reward = np.random.normal(loc=reward_means[action], scale=1)

        # update algorithm
        algo.update(reward)

        # set results
        rewards[time_step] = reward
        is_optimals[time_step] = (reward_means[action] == reward_means_max)

        # for nonstationary, update true reward means and the max tracker
        if not expt_config.is_stationary:
            reward_means += np.random.normal(loc=0, scale=expt_config.random_walk_std, size=(expt_config.k,))
            reward_means_max = np.max(reward_means)

    return {
        "reward": rewards,
        "is_optimal": is_optimals,
    }


def run_experiment_for_average_reward(expt_config: ExperimentConfig) -> float:
    """
    Run experiment and yield the average reward for the last specified timesteps (in expt_config).
    """

    assert expt_config.aggregation_time_steps is not None

    expt_results = run_experiment(expt_config)
    rewards = expt_results["reward"]

    return np.mean(rewards[-expt_config.aggregation_time_steps:])
