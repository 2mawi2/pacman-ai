from collections.__init__ import defaultdict

import numpy as np
from src.app.action import Action
from src.app.game import Game


class Agent:

    def __init__(self, gamma, alpha, epsilon, epsilon_decay):
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(lambda: 0)
        self.N = defaultdict(lambda: 0)
        self.choices = np.array([0, 1, 2, 3])
        self.state_map = defaultdict(lambda: set())

    def get_random_action(self, game: Game) -> Action:
        actions: [Action] = game.get_valid_actions()
        return Action(np.random.choice([i.value for i in actions]))

    def learn(self, next_state, reward, state) -> None:
        state = hash(state.tostring())
        next_state = hash(next_state.tostring())
        self.alpha = 1 / (self.N[state] + 1)
        td_target = reward + self.gamma * self.V[next_state]
        td_delta = (td_target - self.V[state])
        self.V[state] = self.V[state] + self.alpha * td_delta
        self.N[state] += 1

    def get_action_probs(self, Vs):
        probabilities = np.ones(4, dtype=float) * self.epsilon / 4
        best_action = np.argmax(Vs)
        probabilities[best_action] += (1.0 - self.epsilon)
        return probabilities

    def get_greedy_state_and_move(self, game: Game, all_states_list: dict, already_visited: dict):
        self.epsilon *= self.epsilon_decay
        if (1 - self.epsilon) < np.random.rand():  # get random state
            return self._get_random_state(game)
        else:
            valid_states = game.get_valid_states()
            Vs = [self.V[hash(s.tostring())] for s in valid_states]
            best_state = max(zip(valid_states, Vs), key=lambda i: i[1])[0]

            best_state_hash = hash(best_state.tostring())
            # if already_visited[best_state_hash] > 0:
            #   return self._get_random_state(game)

            reward, done = game.move_to_state(best_state)
            return best_state, reward, done, False

    def _get_random_state(self, game: Game) -> (object, int, bool):
        action = self.get_random_action(game)
        state_before_random_move = game.get_field_state()
        reward, done, _ = game.move2(action)
        state = game.get_field_state()
        self.state_map[hash(state_before_random_move.tostring())].add(hash(state.tostring()))
        return game.get_field_state(), reward, done, True
