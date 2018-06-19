from collections.__init__ import defaultdict

import numpy as np

from src.app.direction import Direction


class Agent:

    def __init__(self, gamma, alpha, epsilon):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(lambda: 0)
        self.choices = np.array([0, 1, 2, 3])

    def get_random_action(self) -> Direction:
        return Direction(np.random.choice(self.choices))

    def learn(self, next_state, reward, state, done):
        state = hash(state.tostring())  # overwrite with hashcode
        next_state = hash(next_state.tostring())  # overwrite with hashcode

        if done:
            self.V[state] = self.V[state] + self.alpha * (reward - self.V[state])
        else:
            self.V[state] = self.V[state] + self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])

    def get_greedy_state(self, game, all_states_list):
        if self.epsilon < np.random.rand():  # get random state
            return self._get_next_random_state(game)
        else:  # get state form V greedy
            valid_states = game.get_valid_states(all_states_list)
            if len(valid_states) == 0:
                return self._get_next_random_state(game)
            probs = [self.V[hash(s.tostring())] for s in valid_states]
            return max(zip(valid_states, probs), key=lambda i: i[1])[0]

    def _get_next_random_state(self, game):
        action = self.get_random_action()
        reward, done, _ = game.move2(action)
        return game.get_state_field()
