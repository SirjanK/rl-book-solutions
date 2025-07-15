from enum import Enum
from dataclasses import dataclass


@dataclass
class BanditAlgorithmConfig:
    """
    Top level algorithm config for a single multi arm bandit experiment run.
    """

    pass


@dataclass
class SampleAverageActionValueEpsilonGreedyConfig(BanditAlgorithmConfig):
    """
    Configuration for the sample avg action value epsilon greedy algorithm.
    """

    # Pr(random action)
    epsilon: float


@dataclass
class ExponentialRecencyWeightedAverageActionValueEpsilonGreedyConfig(BanditAlgorithmConfig):
    """
    Configuration for the ERWA action value epsilon greedy algorithm.
    """

    # Pr(random action)
    epsilon: float

    # step size
    step_size: float

    # initial value of action value
    initial_action_value: float


@dataclass
class UpperConfidenceBoundGreedyConfig(BanditAlgorithmConfig):
    """
    Configuration for the upper confidence bound greedy algorithm.
    """

    # weight on the uncertainty term for sampling
    confidence: float


@dataclass
class PerformanceGradientConfig(BanditAlgorithmConfig):
    """
    Configuration for the performance gradient algorithm.
    """

    # step size
    step_size: float
