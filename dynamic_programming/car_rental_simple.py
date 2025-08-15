import numpy as np
from typing import List, Tuple, Iterator
from truncated_poisson import TruncatedPoisson
import itertools


class CarRentalSimpleSolver:
    def __init__(self, 
                 rent_multiplier: float, 
                 move_cost: float, 
                 lambda_loc1_requests: float,
                 lambda_loc2_requests: float,
                 lambda_loc1_returns: float,
                 lambda_loc2_returns: float,
                 max_car_at_loc: float,
                 max_car_moved: float,
                 gamma: float,
                 tol: float) -> None:
        """
        Initialize the solver with given solve parameters.
        Policies here are 2D np arrays pi where pi[i, j] represents the action to take at state (i, j).
        Value functions are also 2D arrays V where V[i, j] representing value at state (i, j).

        :param rent_multipler: mutliplier on number of cars we're able to rent out
        :param move_cost: cost for moving cars
        :param lambda_loc1_requests: lambda for location 1 for requesting rentals
        :param lambda_loc2_requests: lambda for location 2 for requesting rentals
        :param lambda_loc1_returns: lambda for location 1 returns
        :param lambda_loc2_returns: lambda for location 2 returns
        :param max_car_at_loc: max allowable cars at a location
        :param max_car_moved: max number of cars we're allowed to move in one night
        :param gamma: discount factor
        :param tol: tolerance for value functions to determine convergence
        """

        self._rent_multplier = rent_multiplier
        self._move_cost = move_cost
        self._max_car_at_loc = max_car_at_loc
        self._max_car_moved = max_car_moved
        self._gamma = gamma
        self._tol = tol

        # initialize truncated poisson instances that we use later in computation
        self._poisson_loc1_requests = TruncatedPoisson(
            lambda_val=lambda_loc1_requests,
            max_trunc_val=max_car_at_loc,
        )
        self._poisson_loc2_requests = TruncatedPoisson(
            lambda_val=lambda_loc2_requests,
            max_trunc_val=max_car_at_loc,
        )
        self._poisson_loc1_returns = TruncatedPoisson(
            lambda_val=lambda_loc1_returns,
            max_trunc_val=max_car_at_loc,
        )
        self._poisson_loc2_returns = TruncatedPoisson(
            lambda_val=lambda_loc2_returns,
            max_trunc_val=max_car_at_loc,
        )

        # tracker for the instance on whether we have a converged policy
        self._is_converged = False

        # initialize policy and value function
        self._policy = self._initalize_policy()
        self._value_fn = self._initialize_value_fn() 
    
    @property
    def is_converged(self) -> bool:
        """
        Flag to indicate whether this solver has converged
        """

        return self._is_converged
    
    def solve(self, num_iter: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Run policy iteration to solve until convergence. Return list of policies and the final value function.
        The user can call is_converged() to determine whether we have a converged optimal policy or if we exited
        because we hit `num_iter` iterations.

        :param num_iter: maximum number of iterations to run policy iteration for. If it converges before, we'll exit.
        :return:
            1. List of policies through the iteration
            2. Final value function
        """

        assert num_iter > 0

        if self.is_converged:
            print(f"Already converged, nothing left to solve")
            return [], self._value_fn.copy()

        policies = [self._policy.copy()]  # list of policies, init with current policy
        for iter_idx in range(1, num_iter + 1):
            # 1. duplicate the prev value function 
            prev_value_fn = self._value_fn.copy()

            # 2. run policy evaluation
            self._policy_evaluation()

            # 3. run policy improvement
            self._policy_improvement()
            # append copy
            policies.append(self._policy.copy())

            # 4. compare two value functions - if converged, set the flag and exit
            delta = np.max(np.abs(prev_value_fn - self._value_fn))
            if delta < self._tol:
                self._is_converged = True
                print(f"Terminating due to convergence at {iter_idx=}")
                break

            if iter_idx % 10 == 0:
                print(f"At iteration {iter_idx=}")
        
        print(f"Concluded at {iter_idx=}")
        return policies, self._value_fn.copy()
    
    def _compute_expected_return_given_state_action(self, state: Tuple[int, int], action: int) -> float:
        """
        Compute the expected return given S_t, A_t
        """

        state_car1, state_car2 = state

        # expected visits to locations that are successful (we actually have cars left)
        expected_successful_visits1 = self._poisson_loc1_requests.expectation(trunc_val=state_car1)
        expected_successful_visits2 = self._poisson_loc2_requests.expectation(trunc_val=state_car2)

        # reward accrued due to visits
        reward_accrued_due_to_visits = self._rent_multplier * (expected_successful_visits1 + expected_successful_visits2)
        # actual expected reward is accrued minus action penalty
        expected_reward = reward_accrued_due_to_visits - self._move_cost * np.abs(action)

        # iterate through the next possible states
        # first add affects of cars moved
        state_after_move1 = state_car1 + action
        state_after_move2 = state_car2 - action

        weighted_next_state_contributions = 0
        # returns to each location is independent of the other
        # they are distributed according to the truncated poisson
        # with trunc val max_car_at_loc - state_after_move
        max_return1 = self._max_car_at_loc - state_after_move1
        max_return2 = self._max_car_at_loc - state_after_move2
        for returns_loc1, returns_loc2 in itertools.product(
            range(max_return1 + 1),
            range(max_return2 + 1),
        ):
            pmf_loc1 = self._poisson_loc1_returns.pmf(trunc_val=max_return1, num=returns_loc1)
            pmf_loc2 = self._poisson_loc2_returns.pmf(trunc_val=max_return2, num=returns_loc2)
            # returns are independent
            pmf = pmf_loc1 * pmf_loc2

            weighted_next_state_contributions += pmf * self._value_fn[state_after_move1 + returns_loc1, state_after_move2 + returns_loc2]
        
        return expected_reward + self._gamma * weighted_next_state_contributions
    
    def _policy_evaluation(self) -> None:
        """
        Policy evaluation iterations - update value_fn given current policy.
        Guaranteed to terminate given this is a finite MDP.
        """

        delta = float("inf")  # delta between value functions
        # iterate until convergence
        while delta >= self._tol:
            delta = 0
            # update all states
            for state in self._state_iter():
                new_value = self._compute_expected_return_given_state_action(
                    state=state, 
                    action=self._policy[*state],
                )
                old_value = self._value_fn[*state]
                self._value_fn[*state] = new_value

                delta = max(delta, abs(old_value - new_value))

    def _policy_improvement(self) -> None:
        """
        Policy improvement iterations - update policy given current value function.
        Guaranteed to terminate given this is a finite MDP.
        """

        # True once we've reached a fixed point in the policy
        stable = False

        while not stable:
            stable = True
            for state in self._state_iter():
                state_car1, state_car2 = state
                old_action = self._policy[state_car1, state_car2]

                # get the action that yields max expected return
                # actions are restricted to a certain range
                max_action = old_action 
                max_action_value = -float("inf")
                for action in range(max(-state_car1, -self._max_car_moved), min(state_car2, self._max_car_moved) + 1):
                    expected_return = self._compute_expected_return_given_state_action(state=state, action=action)

                    if expected_return > max_action_value:
                        max_action_value = expected_return
                        max_action = action
                
                stable = stable and (max_action == old_action)

                self._policy[state_car1, state_car2] = action
    
    def _state_iter(self) -> Iterator[Tuple[int, int]]:
        """
        Utility function to iterate through all states

        :return: iterator of all states (n1, n2) of number of cars at location
        """
        return itertools.product(range(self._max_car_at_loc + 1), range(self._max_car_at_loc + 1))

    def _initalize_policy(self) -> np.ndarray:
        """
        Initialize the policy with zeroes (no car movement)
        """

        return np.zeros(shape=(self._max_car_at_loc + 1, self._max_car_at_loc + 1), dtype=np.int32)

    def _initialize_value_fn(self) -> np.ndarray:
        """
        Initialize value function randomly
        """

        return 100 * np.random.uniform(size=(self._max_car_at_loc + 1, self._max_car_at_loc + 1))
