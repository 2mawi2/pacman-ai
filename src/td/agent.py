from collections.__init__ import defaultdict

import numpy as np

from src.app.direction import Direction


class Agent:

    def __init__(self, gamma, alpha):
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(lambda: 0)

    def get_random_action(self) -> Direction:
        return Direction(np.random.choice([0, 1, 2, 3]))

    def learn(self, next_state, reward, state):
        state = hash(state.tostring())  # overwrite with hashcode
        next_state = hash(next_state.tostring())  # overwrite with hashcode
        self.V[state] = self.V[state] + self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])
