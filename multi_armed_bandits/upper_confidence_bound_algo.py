import numpy as np
from multi_armed_bandits.bandit_algorithm import BanditAlgorithm
from overrides import override


class UpperConfidenceBoundAlgo(BanditAlgorithm):
    """
    Upper confidence bound algorithm that samples action based on the estimate action value plus an
    uncertainty term.
    """

    def __init__(self, k: int, confidence: float) -> None:
        super().__init__(k=k)

        self.confidence = confidence

        self.counts = np.zeros(shape=(self.k,), dtype=np.int32)  # counts for each action
        self.action_values = np.zeros(shape=(self.k,), dtype=np.float32)  # action values for each action

        self.time_step = 0  # time step (each time act() gets called)

        # optimization to avoid checking for counts[a] == 0 every time once we've incremented each once
        self.actions_not_chosen = set(range(self.k))
    
    @override
    def act(self) -> int:
        self.time_step += 1
        return super().act()

    def compute_action(self) -> int:
        if len(self.actions_not_chosen) > 0:
            # get a random one from here
            action = np.random.choice(list(self.actions_not_chosen))

            self.actions_not_chosen.remove(action)

            return action
        
        # get indices with max action value + uncertainty
        value_with_uncertainties = self.action_values + self.confidence * np.sqrt(np.log(self.time_step) / self.counts)
        max_action_value_indices = np.where(value_with_uncertainties == np.max(value_with_uncertainties))[0]
        # take one at random from this
        return np.random.choice(max_action_value_indices)

    def update(self, reward: float) -> None:
        # using previous action and reward, update tables
        self.counts[self._prev_action] += 1

        step_size = 1 / self.counts[self._prev_action]

        prev_action_value = self.action_values[self._prev_action]
        self.action_values[self._prev_action] = prev_action_value + step_size * (reward - prev_action_value)
        