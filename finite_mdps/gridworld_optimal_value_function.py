"""
Solve the state optimal Bellman equations for the Gridworld problem in Chapter 3.

5 x 5 grid, NWSE actions, hitting boundaries incur reward of -1, otherwise 0.
Special twists: 
* at (0, 1), any action leads to +10 reward and leads to (4, 1).
* at (0, 3), any action leads to +5 reward, leads to (2, 3).

Generate 5 x 5 grid of v*(s).

We will solve this using value iteration. To prove to myself value iteration does converge, I spelled it out
in `Value_Iteration_Convergence_Proof.pdf`.
"""


import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


# static constants
HEIGHT = 5
WIDTH = 5

BOUNDARY_REWARD = -1
NEUTRAL_REWARD = 0

# discount factor
GAMMA = 0.9

# delta between successive states that defines completion
# if |V_{k+1}(s) - V_k(s)| < VALUE_DELTA for all states, we terminate the algorithm
VALUE_DELTA = 1e-6

# maximum iteration number
MAX_ITER = 10000

# special points and their rewards
@dataclass
class SpecialRewardSpec:
    src_point: Tuple[int, int]
    dest_point: Tuple[int, int]
    reward: float


SPECIAL_REWARD_SPECS: List[SpecialRewardSpec] = [
    SpecialRewardSpec(
        src_point=(0, 1),
        dest_point=(4, 1),
        reward=10,
    ),
    SpecialRewardSpec(
        src_point=(0, 3),
        dest_point=(2, 3),
        reward=5
    )
]


def in_bounds(i: int, j: int) -> bool:
    # return if (i, j) is in bounds
    return 0 <= i < HEIGHT and 0 <= j < WIDTH


def value_iteration_on_grid() -> np.ndarray:
    special_points = {spec.src_point: spec for spec in SPECIAL_REWARD_SPECS} 

    # arbitrary initial point
    state_values = np.zeros(shape=(HEIGHT, WIDTH), dtype=np.float32)

    for iter_idx in range(MAX_ITER):
        # value iteration by applying update functions
        terminate = True
        for i in range(HEIGHT):
            for j in range(WIDTH):
                prev_value = state_values[i, j]
                if (i, j) in special_points:
                    # handle special point case
                    spec = special_points[(i, j)]
                    i_prime, j_prime = spec.dest_point
                    state_values[i, j] = spec.reward + GAMMA * state_values[i_prime, j_prime]
                else:
                    update_points = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                    max_value = -float("inf")
                    for i_prime, j_prime in update_points:
                        if not in_bounds(i_prime, j_prime):
                            max_value = max(max_value, -1 + GAMMA * state_values[i, j])
                        else:
                            max_value = max(max_value, GAMMA * state_values[i_prime, j_prime])
                    state_values[i, j] = max_value

                if abs(state_values[i, j] - prev_value) >= VALUE_DELTA:
                    terminate = False
        
        if terminate:
            break

        if iter_idx % 1000 == 0:
            print(f"completed {iter_idx=}")
    
    print(f"completed total of {iter_idx=}")
    return state_values


if __name__ == "__main__":
    optimum = value_iteration_on_grid()
    with np.printoptions(precision=4, suppress=True, floatmode='fixed'):
        print(optimum)
