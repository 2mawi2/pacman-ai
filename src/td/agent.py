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
        state = hash(state.tostring())  # overwrite with hashcode
        next_state = hash(next_state.tostring())  # overwrite with hashcode
        alpha = 1 / (self.N[state] + 1)
        self.V[state] = self.V[state] + alpha * (reward + self.gamma * self.V[next_state] - self.V[state])
        self.N[state] += 1

    def get_valid_states(self, all_states: dict, current_state: int) -> [np.multiarray]:
        state_hashes = self.state_map[current_state]
        return [all_states[i] for i in state_hashes]

    def get_greedy_state_and_move(self, game: Game, all_states_list: dict) -> (np.multiarray, int, bool):
        self.epsilon *= self.epsilon_decay
        if (1 - self.epsilon) < np.random.rand():  # get random state
            return self._get_random_state(game)
        else:
            current_state = game.get_state()
            valid_states = self.get_valid_states(all_states_list, current_state)

            if len(valid_states) < 1:
                return self._get_random_state(game)

            Vs = [self.V[hash(s.tostring())] for s in valid_states]
            best_state = max(zip(valid_states, Vs), key=lambda i: i[1])[0]

            reward, done = game.move_to_state(best_state)
            return best_state, reward, done

    def _get_random_state(self, game: Game) -> (np.multiarray, int, bool):
        action = self.get_random_action(game)
        state_before_random_move = game.get_field_state()
        reward, done, _ = game.move2(action)
        state = game.get_field_state()
        self.state_map[hash(state_before_random_move.tostring())].add(hash(state.tostring()))
        return game.get_field_state(), reward, done
