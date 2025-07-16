"""
Solve the state optimal Bellman equations for the Gridworld problem in Chapter 3.

5 x 5 grid, NWSE actions, hitting boundaries incur reward of -1, otherwise 0.
Special twists: 
* at (0, 1), any action leads to +10 reward and leads to (4, 1).
* at (0, 3), any action leads to +5 reward, leads to (2, 3).

Generate 5 x 5 grid of v*(s).

We will solve this using value iteration. To prove to myself value iteration does converge,

"""


