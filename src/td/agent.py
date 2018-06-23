from collections.__init__ import defaultdict

import numpy as np

from src.app.action import Action


class Agent:

    def __init__(self, gamma):
        self.gamma = gamma
        self.V = defaultdict(lambda: 0)
        self.N = defaultdict(lambda: 0)

    def get_random_action(self, valid_actions) -> Action:
        return Action(np.random.choice(valid_actions))

    def learn(self, next_state, reward, state):
        state = hash(state.tostring())  # overwrite with hashcode
        next_state = hash(next_state.tostring())  # overwrite with hashcode

        alpha = 1 / (self.N[state] + 1)
        self.V[state] = self.V[state] + alpha * (reward + self.gamma * self.V[next_state] - self.V[state])
        self.N[state] += 1
