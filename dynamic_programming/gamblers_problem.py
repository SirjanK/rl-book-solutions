import numpy as np
from typing import List, Tuple


def get_max_action(target: int, p_h: int, state: int, value_fn: np.ndarray) -> Tuple[int, float]:
    """
    Given state, value_fn, get the optimal action along with the expected return given this action.
    It's arbitrary to select which action in cases of ties. In that case, we select the one closest to state

    :return: 1. optimal action, 2. expected return given this action
    """

    candidate_actions = []
    max_action_value = -float("inf")
    # in this setting, we're not allowed to stay still, so actions start with stake of 1
    for action in range(1, min(state, target - state) + 1):
        value = p_h * value_fn[state + action] + (1 - p_h) * value_fn[state - action]
        if value > max_action_value:
            candidate_actions = [action]
            max_action_value = value
        elif value == max_action_value:
            candidate_actions.append(action)
    
    max_action = min(candidate_actions, key=lambda action: abs(state - action))
    
    return max_action, max_action_value


def solve_gamblers_problem_value_iteration(target: int, p_h: float, tol: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Solve the Gambler's Problem via value iteration.
    Value functions are 1D numpy arrays with indices being the state {0, 1, ..., target}, V(0) = 0, V(target) = 1 fixed.
    Policies are 1D integer numpy arrays; pi[s] fall in between {0, 1, ..., min{s, target - s}}

    :param target: amount that constitutes a win
    :param p_h: probability of heads
    :param tol: convergence parameter
    :param max_iter: exit after hitting this many iterations
    :return final policy, list of value functions through the iterations of value iteration
    """

    assert target > 0
    assert 0 < p_h < 1

    # PART I: core value iteration
    # initialize to 0 except for target
    value_fn = np.zeros(shape=(target + 1,))
    value_fn[target] = 1

    # iterate
    value_fns = []
    for iter_idx in range(1, max_iter + 1):
        delta = 0
        for state in range(1, target):
            old_val = value_fn[state]

            _, max_action_value = get_max_action(target, p_h, state, value_fn)

            value_fn[state] = max_action_value
            delta = max(delta, abs(max_action_value - old_val))
        
        value_fns.append(value_fn.copy())
        
        if iter_idx % 10 == 0:
            print(f"At {iter_idx=}; {delta=}")
        
        if delta < tol:
            print(f"Ending value iteration due to convergence; {iter_idx=}, {delta=}")
            break
    
    if iter_idx == max_iter:
        raise Exception(f"Exiting due to no convergence. Check max_iter or the algorithm.")

    # PART II: construction of an optimal policy
    policy = np.zeros(shape=(target + 1,), dtype=np.int32)
    for state in range(1, target):
        max_action, _ = get_max_action(target, p_h, state, value_fn)
        policy[state] = max_action
    
    return policy, value_fns
