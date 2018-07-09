from collections.__init__ import defaultdict

import numpy as np
from src.app.fieldtype import FieldType


class Agent:

    def __init__(self, gamma, alpha):
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(lambda: 0)

    def learn(self, next_state, reward, state) -> None:
        state = hash(state.tostring())
        next_state = hash(next_state.tostring())
        td_target = reward + self.gamma * self.V[next_state]
        td_delta = (td_target - self.V[state])
        self.V[state] = self.V[state] + self.alpha * td_delta

    def get_greedy_state(self, game):
        valid_states = game.get_valid_states()
        vs = self._get_vs(valid_states, game)
        return valid_states[vs.index(max(vs))]

    def _get_vs(self, valid_states, game):
        vs = []
        for s in valid_states:
            y, x = np.where(s == "p")
            field_type = game.get_field_type(x, y)
            if field_type == FieldType.GHOST:
                vs.append(-100)
            elif field_type == FieldType.STAR:
                vs.append(10)
            else:
                vs.append(self.V[hash(s.tostring())])
        return vs
