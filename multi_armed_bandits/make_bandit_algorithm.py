from multi_armed_bandits.bandit_algorithm import BanditAlgorithm
from multi_armed_bandits.bandit_algorithm_config import (
    BanditAlgorithmConfig, 
    SampleAverageActionValueEpsilonGreedyConfig,
    ExponentialRecencyWeightedAverageActionValueEpsilonGreedyConfig,
    UpperConfidenceBoundGreedyConfig,
    PerformanceGradientConfig,
)
from multi_armed_bandits.experiment_config import ExperimentConfig
from multi_armed_bandits.action_value_epsilon_greedy_algos import SampleAverageAlgo, ExponentialRecencyWeightedAverageAlgo
from multi_armed_bandits.upper_confidence_bound_algo import UpperConfidenceBoundAlgo
from multi_armed_bandits.performance_gradient_algorithm import PerformanceGradientAlgorithm


def make_bandit_algorithm(expt_config: ExperimentConfig, algorithm_config: BanditAlgorithmConfig) -> BanditAlgorithm:
    """
    Factory function for initializing a bandit algorithm.
    """

    match algorithm_config:
        case SampleAverageActionValueEpsilonGreedyConfig(epsilon=epsilon):
            return SampleAverageAlgo(k=expt_config.k, eps=epsilon)
        case ExponentialRecencyWeightedAverageActionValueEpsilonGreedyConfig(
            epsilon=epsilon, 
            initial_action_value=initial_action_value, 
            step_size=step_size):
            return ExponentialRecencyWeightedAverageAlgo(
                k=expt_config.k, 
                eps=epsilon, 
                init_value=initial_action_value, 
                alpha=step_size
            )
        case UpperConfidenceBoundGreedyConfig(confidence=confidence):
            return UpperConfidenceBoundAlgo(k=expt_config.k, confidence=confidence)
        case PerformanceGradientConfig(step_size=step_size):
            return PerformanceGradientAlgorithm(k=expt_config.k, step_size=step_size)
        case _:
            raise ValueError(f"Unsupported algorithm config type {type(algorithm_config)=}")
