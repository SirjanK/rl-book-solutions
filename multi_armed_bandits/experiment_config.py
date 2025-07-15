from dataclasses import dataclass
from multi_armed_bandits.bandit_algorithm_config import BanditAlgorithmConfig
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    Top level experiment config for a single multi arm bandit experiment run.
    """

    # name of the experiment; useful for distinguishing type semantically
    name: str

    # whether the experiment is stationary or nonstationary
    is_stationary: bool

    # random walk standard deviation for nonstationary case - only applicable if is_stationary is false
    random_walk_std: Optional[float]

    # number of actions
    k: int

    # number of time steps
    num_time_steps: int

    # last number of time steps to compute average return
    aggregation_time_steps: Optional[int]

    # bandit algorithm config
    bandit_algorithm_config: BanditAlgorithmConfig
    
