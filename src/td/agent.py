from collections.__init__ import defaultdict

import numpy as np

from src.app.direction import Direction


class Agent:

    def __init__(self, gamma):
        self.gamma = gamma
        self.V = defaultdict(lambda: 0)
        self.N = defaultdict(lambda: 0)
        self.choices = np.array([0, 1, 2, 3])

    def get_random_action(self) -> Direction:
        return Direction(np.random.choice(self.choices))

    def learn(self, next_state, reward, state):
        state = hash(state.tostring())  # overwrite with hashcode
        next_state = hash(next_state.tostring())  # overwrite with hashcode

        alpha = 1 / (self.N[state] + 1)
        self.V[state] = self.V[state] + alpha * (reward + self.gamma * self.V[next_state] - self.V[state])
        self.N[state] += 1
