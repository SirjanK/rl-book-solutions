"""
Functions for solving barebones Jack's car rental problem using policy iteration.

Policies here are 2D np arrays pi where pi[i, j] represents the action to take at state (i, j)

Value functions similarly are 2D np arrays
"""

import numpy as np
from typing import List, Tuple


# Constants describing the parameters of the problem
RENT_MULTIPLIER = 10  # multiplier on number of cars we're able to rent out
MOVE_COST = 2  # cost for moving cars
LAMBDA_LOC1_REQUESTS = 3  # lambda for location 1 for requesting rentals
LAMBDA_LOC2_REQUESTS = 4
LAMBDA_LOC1_RETURNS = 3  # lambda for location 1 returns
LAMBDA_LOC2_RETURNS = 2
MAX_CAR_AT_LOC = 20  # max allowable cars at a location
MAX_CAR_MOVED = 5  # max number of cars we're allowed to move in one night
GAMMA = 0.9  # discount factor

TOL = 1e-3  # tolerance factor for differing value functions


def initalize_policy() -> np.ndarray:
    """
    Initialize the policy with zeroes (no car movement)
    """

    return np.zeros(shape=(MAX_CAR_AT_LOC + 1, MAX_CAR_AT_LOC + 1), dtype=np.int32)


def policy_evaluation(policy: np.ndarray, tol: float=TOL) -> np.ndarray:
    """
    Carry out policy evaluation given the current policy

    :param policy: 2D np array representing current policy
    :return: 2D np array representing value function for this policy
    """

    # TODO impl
    return np.zeros(shape=(MAX_CAR_AT_LOC + 1, MAX_CAR_AT_LOC + 1), dtype=np.float32)


def policy_improvement(starting_policy: np.ndarray, value_fn: np.ndarray) -> np.ndarray:
    """
    Carry out policy improvement on the current value function.

    :param starting_policy: policy to start iteration from
    :param value_fn: value function to use during policy improvement
    :return: converged policy (fixed point)
    """

    # TODO impl
    return starting_policy


def policy_iteration() -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Carry out policy iteration for this problem. Return list of policies we get as we iterate along
    with the final value function.
    """

    # TODO impl
    pass
