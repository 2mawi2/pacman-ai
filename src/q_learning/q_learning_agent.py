from collections.__init__ import defaultdict

import numpy as np

from src.app.action import Action


class Agent:
    def __init__(self, gamma, alpha, epsilon, epsilon_decay, alpha_decay):
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = 4
        self.Q = defaultdict(lambda: np.zeros(4))

    def get_action_probs(self, state):
        probabilities = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        best_action = np.argmax(self.Q[state])
        probabilities[best_action] += (1.0 - self.epsilon)
        return probabilities

    def get_action(self, state) -> Action:
        action_probs = self.get_action_probs(state)
        action = np.random.multinomial(len(action_probs), action_probs).argmax()
        return Action(action)

    def learn(self, next_state, reward, state, action):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action.value]
        self.Q[state][action.value] = self.Q[state][action.value] + self.alpha * td_delta

        self.epsilon *= (1 - self.epsilon_decay)
        self.alpha *= (1 - self.alpha_decay)
