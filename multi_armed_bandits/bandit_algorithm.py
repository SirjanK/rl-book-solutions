from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    """
    Generic bandit algorithm; initialize for a start of a run
    """

    def __init__(self, k: int) -> None:
        self._prev_action = None
        self.k = k

    def act(self) -> int:
        """
        Compute and store next action; return integer of action label
        """

        self._prev_action = self.compute_action()
        return self._prev_action

    @abstractmethod
    def compute_action(self) -> int:
        """
        Compute next action; return integer of action label
        """

        pass

    @abstractmethod
    def update(self, reward: float) -> None:
        """
        Update trackers based on the and reward incurred
        """

        pass
