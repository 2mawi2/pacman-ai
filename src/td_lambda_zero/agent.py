from collections.__init__ import defaultdict

import numpy as np

from src.app.direction import Direction


class Agent:
    def __init__(self, discount_factor, alpha, epsilon):
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.n_actions = 4
        self.Q = defaultdict(lambda: np.zeros(4))

    def get_action_probs(self, state):
        probabilities = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        best_action = np.argmax(self.Q[state])
        probabilities[best_action] += (1.0 - self.epsilon)
        return probabilities

    def get_action(self, state) -> Direction:
        action_probs = self.get_action_probs(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return Direction(action)

    def learn(self, next_state, reward, state, action):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action.value]
        self.Q[state][action.value] += self.alpha * td_delta
