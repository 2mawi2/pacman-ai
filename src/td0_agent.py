import itertools

import numpy as np
from numpy import random
import operator
from collections import defaultdict

from src.direction import Direction

q_init = {
    Direction.UP: 0.0,
    Direction.DOWN: 0.0,
    Direction.RIGHT: 0.0,
    Direction.LEFT: 0.0,
}


class TD0Agent:
    def __init__(self, alpha: float, gamma: float, epsilon: float, epsilon_decay: float):
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.V_t = {}
        self.Q_t: {int, {Direction, float}} = {}

    def Q(self, state: int) -> dict:
        return self.Q_t.get(state, q_init)

    def max_Q(self, state: int):
        qs: dict = self.Q(state)

        max_value = max(qs.items(), key=operator.itemgetter(1))
        all_maximas = [key for key, value in qs.items() if value == max_value[1]]
        if len(all_maximas) > 0:
            result = self.get_random_action(all_maximas), max_value[1]
            return result
        else:
            return max(qs.items(), key=operator.itemgetter(1))

    def get_random_action(self, actions=None) -> Direction:
        if actions is None:
            actions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        return random.choice(actions)

    def get_action(self, state: int) -> Direction:
        if np.random.rand() <= self.epsilon:
            return self.get_random_action()
        else:
            return self.max_Q(state)[0]

    def learn(self, state: int, next_state: int, reward: int, action: Direction):
        if state not in self.Q_t:
            self.Q_t[state] = q_init
            # delta = self.alpha * self.gamma * self.max_Q(next_state)[1]
            # q = reward + delta
        # q = (1 - self.alpha) * self.Q(state)[action] + self.alpha * (reward + self.gamma * self.max_Q(next_state)[1])
        # delta = (1 - self.alpha) * (reward + self.gamma * self.max_Q(next_state)[1])
        # print(f"q : {q}")

        td_target = reward + self.max_Q(next_state)[1]
        self.Q_t[state][action] = self.alpha * (td_target - self.Q_t[state][action])
        self.alpha *= self.gamma

        # self.V_t[state] = self.V(state, next_state, reward)
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = (self.epsilon * self.epsilon_decay)

    def V(self, state: int, next_state: int, reward: int):
        if state not in self.V_t:
            self.V_t[state] = 0
        if next_state not in self.V_t:
            self.V_t[next_state] = 0
        return self.V_t[state] + self.alpha * (reward + self.gamma * self.V_t[next_state] - self.V_t[state])
