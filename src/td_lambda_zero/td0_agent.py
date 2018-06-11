import numpy as np
from numpy import random
import operator

from src.app.direction import Direction

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
        self.V_t = np.array([])

    def get_random_action(self, actions=None) -> Direction:
        if actions is None:
            actions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        return random.choice(actions)

    def get_action(self, state: int) -> Direction:
        return self.get_random_action()

    def learn(self, state: int, next_state: int, reward: int, t: int, action: Direction):
        self.V_t[state] = self.V(state, next_state, reward, t, action)

    def decay_epsilon(self):
        self.epsilon = (self.epsilon * self.epsilon_decay)

    def V(self, state: int, next_state: int, reward: int, t: int, action: Direction):
        if state not in self.V_t:
            self.V_t[state] = 0
        if next_state not in self.V_t:
            self.V_t[next_state] = 0
        alpha = 1 / t
        return self.V_t[state, action] + alpha * (reward + self.gamma * self.V_t[next_state, action] - self.V_t[state, action])

