import numpy as np
from abc import ABC, abstractmethod
from multi_armed_bandits.bandit_algorithm import BanditAlgorithm


class ActionValueEpsilonGreedyAlgo(BanditAlgorithm, ABC):
    """
    Action value bandit algorithms that keep track of estimated action values using some configurable step size.
    """

    def __init__(self, k: int, eps: float, init_value: float) -> None:
        super().__init__(k=k)

        self.eps = eps

        self.counts = np.zeros(shape=(self.k,), dtype=np.int32)
        self.action_values = np.ones(shape=(self.k,), dtype=np.float32) * init_value

    def compute_action(self) -> int:
        if np.random.rand() < self.eps:
            # sample random action
            return np.random.randint(low=0, high=self.k)
        
        # get indices with max action value
        max_action_value_indices = np.where(self.action_values == np.max(self.action_values))[0]
        # take one at random from this
        return np.random.choice(max_action_value_indices)

    def update(self, reward: float) -> None:
        # using previous action and reward, update tables
        self.counts[self._prev_action] += 1

        step_size = self.compute_step_size()

        prev_action_value = self.action_values[self._prev_action]
        self.action_values[self._prev_action] = prev_action_value + step_size * (reward - prev_action_value)
    
    @abstractmethod
    def compute_step_size(self) -> float:
        """
        Compute step size for update.
        """

        pass


class SampleAverageAlgo(ActionValueEpsilonGreedyAlgo):
    def __init__(self, k: int, eps: float) -> None:
        super().__init__(k=k, eps=eps, init_value=0.0)

    def compute_step_size(self) -> float:
        return 1 / self.counts[self._prev_action]


class ExponentialRecencyWeightedAverageAlgo(ActionValueEpsilonGreedyAlgo):
    def __init__(self, k: int, eps: float, init_value: float, alpha: float) -> None:
        super().__init__(k=k, eps=eps, init_value=init_value)

        self.alpha = alpha
    
    def compute_step_size(self) -> float:
        return self.alpha
