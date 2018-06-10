import numpy as np
from numpy import random
import operator

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
        if np.all([i == 0.0 for i in self.Q(state).values()]):
            return self.get_random_action(), 0.0  # if no q values set -> take random action
        else:
            return max(self.Q(state).items(), key=operator.itemgetter(1))

    def get_random_action(self) -> Direction:
        return random.choice([Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP])

    def get_action(self, state: int) -> Direction:
        if np.random.rand() <= self.epsilon:
            return self.get_random_action()
        else:
            return self.max_Q(state)[0]

    def learn(self, state: int, next_state: int, reward: int, action: Direction):
        delta = reward + self.max_Q(next_state)[1] - self.Q(state)[action]

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
