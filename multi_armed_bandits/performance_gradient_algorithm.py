from multi_armed_bandits.bandit_algorithm import BanditAlgorithm
import numpy as np


class PerformanceGradientAlgorithm(BanditAlgorithm):
    def __init__(self, k: int, step_size: float) -> None:
        super().__init__(k)

        self.step_size = step_size

        self.policy_logits = np.zeros(shape=(k,), dtype=np.float32)
        self.reward_baseline = 0
        self.count = 0
    
    def compute_action(self) -> int:
        # sample from distribution of policy logits
        policy = self._get_policy()

        return np.random.choice(np.arange(start=0, stop=self.k, step=1, dtype=np.int32), p=policy)

    def update(self, reward: float) -> None:
        # update baseline
        self.reward_baseline = reward / (self.count + 1) + self.reward_baseline * (self.count / (self.count + 1))
        self.count += 1

        # update logits
        update_scale_factor = self.step_size * (reward - self.reward_baseline)
        policy = self._get_policy()
        mask = np.ones(shape=self.policy_logits.shape, dtype=np.bool)
        mask[self._prev_action] = False

        self.policy_logits[mask] -= update_scale_factor * policy[mask]
        self.policy_logits[self._prev_action] += update_scale_factor * (1 - policy[self._prev_action])

    def _get_policy(self) -> np.ndarray:
        # softmax on policy logits for each one
        exp_logit = np.exp(self.policy_logits)
        total = np.sum(exp_logit)

        return exp_logit / total
