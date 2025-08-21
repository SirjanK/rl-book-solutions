import numpy as np
from typing import List, Tuple, Iterator
from truncated_poisson import TruncatedPoisson
import itertools
from typing import Optional


class CarRentalSolver:
    def __init__(self, 
                 rent_multiplier: float, 
                 move_cost: float, 
                 lambda_loc1_requests: float,
                 lambda_loc2_requests: float,
                 lambda_loc1_returns: float,
                 lambda_loc2_returns: float,
                 max_car_at_loc: int,
                 max_car_moved: int,
                 gamma: float,
                 tol: float,
                 move_benefit: bool = False,
                 overflow_limit: Optional[int] = None,
                 overflow_cost: Optional[float] = None) -> None:
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
        :param move_benefit: bool flag indicating if we get one first loc -> second loc movement for free
        :param overflow_limit: limit for a parking lot at a given location. Any more than this is counted as an "overflow"
        :param overflow_cost: cost incurred if we an overflow
        """

        self._rent_multplier = rent_multiplier
        self._move_cost = move_cost
        self._max_car_at_loc = max_car_at_loc
        self._max_car_moved = max_car_moved
        self._gamma = gamma
        self._tol = tol
        self._move_benefit = move_benefit
        assert not ((overflow_limit is None) ^ (overflow_cost is None))
        self._overflow_limit = overflow_limit
        self._overflow_cost = overflow_cost

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

        policies = [self._policy.copy()]  # list of policies, init with current policy
        if self.is_converged:
            print(f"Already converged, nothing left to solve")
            return policies, self._value_fn.copy()

        for iter_idx in range(1, num_iter + 1):
            # 1. duplicate the prev value function 
            prev_value_fn = self._value_fn.copy()

            # 2. run policy evaluation
            self._policy_evaluation()

            # 3. run policy improvement
            converged = self._policy_improvement()
            # append copy
            policies.append(self._policy.copy())

            # 4. compare two value functions - if converged, set the flag and exit
            if converged or np.max(np.abs(prev_value_fn - self._value_fn)) < self._tol:
                self._is_converged = True
                print(f"Terminating due to convergence at {iter_idx=}")
                break

            print(f"At iteration {iter_idx=}")
        
        print(f"Concluded at {iter_idx=}")
        return policies, self._value_fn.copy()
    
    def _compute_expected_return_given_state_action(self, state: Tuple[int, int], action: int) -> float:
        """
        Compute the expected return given S_t, A_t
        """

        state_car1, state_car2 = state

        # state after moves
        state_after_move1, state_after_move2 = state_car1 + action, state_car2 - action

        # compute expected successful requests
        expected_requests1 = self._poisson_loc1_requests.expectation(trunc_val=state_after_move1)
        expected_requests2 = self._poisson_loc2_requests.expectation(trunc_val=state_after_move2)
        # expected reward for a single time step r(s, a)
        moves_with_cost = abs(action)
        # if we move from first to second and we have move benefit enabled, subtract one off
        if action < 0 and self._move_benefit:
            moves_with_cost -= 1
        expected_reward = self._rent_multplier * (expected_requests1 + expected_requests2) - self._move_cost * moves_with_cost
        # if overflow limit and cost are enabled, incur penalties
        if self._overflow_limit is not None:
            if state_after_move1 > self._overflow_limit:
                expected_reward -= self._overflow_cost
            if state_after_move2 > self._overflow_limit:
                expected_reward -= self._overflow_cost

        # expected next state contribution using p(s'|s, a)
        expected_next_state_value = 0
        for request1, request2 in itertools.product(range(state_after_move1 + 1), range(state_after_move2 + 1)):
            # all pmfs are independent
            pmf_requests = self._poisson_loc1_requests.pmf(trunc_val=state_after_move1, num=request1) \
                * self._poisson_loc2_requests.pmf(trunc_val=state_after_move2, num=request2)

            # iterate through the return random variables
            expected_given_requests = 0
            max_return1 = self._max_car_at_loc - state_after_move1 + request1
            max_return2 = self._max_car_at_loc - state_after_move2 + request2
            for return1, return2 in itertools.product(
                range(max_return1 + 1), range(max_return2 + 1),
            ):
                pmf_returns = self._poisson_loc1_returns.pmf(trunc_val=max_return1, num=return1) \
                    * self._poisson_loc2_returns.pmf(trunc_val=max_return2, num=return2)
                next_state1 = state_after_move1 - request1 + return1
                next_state2 = state_after_move2 - request2 + return2
                expected_given_requests += pmf_returns * self._value_fn[next_state1, next_state2]
            expected_next_state_value += pmf_requests * expected_given_requests
        
        return expected_reward + self._gamma * expected_next_state_value
    
    def _policy_evaluation(self) -> None:
        """
        Policy evaluation iterations - update value_fn given current policy.
        Guaranteed to terminate given this is a finite MDP.
        """

        delta = float("inf")  # delta between value functions
        iter_no = 0
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
            iter_no += 1
            print(f"{iter_no=}; {delta=}")

    def _policy_improvement(self) -> bool:
        """
        Policy improvement iterations - update policy given current value function.
        Guaranteed to terminate given this is a finite MDP.

        :return: bool flag whether we've hit convergence
        """

        # True once we've reached a fixed point in the policy
        stable = True
        for state in self._state_iter():
            state_car1, state_car2 = state
            old_action = self._policy[state_car1, state_car2]

            # get the action that yields max expected return
            # actions are restricted to a certain range
            max_action = old_action 
            max_action_value = -float("inf")
            for action in range(
                max(-state_car1, -self._max_car_moved, state_car2 - self._max_car_at_loc), 
                min(state_car2, self._max_car_moved, self._max_car_at_loc - state_car1) + 1):
                expected_return = self._compute_expected_return_given_state_action(state=state, action=action)

                if expected_return > max_action_value:
                    max_action_value = expected_return
                    max_action = action
            
            stable = stable and (max_action == old_action)

            self._policy[state_car1, state_car2] = max_action
        
        return stable
    
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
