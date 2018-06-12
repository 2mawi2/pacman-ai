import itertools
import plotly.graph_objs as go
import plotly
import numpy as np
import pandas as pd
import sys
from collections import defaultdict

import plotly

from src.app.direction import Direction
from src.app.game import Game


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def parse_action(action: int) -> Direction:
    switcher = {
        0: Direction.RIGHT,
        1: Direction.LEFT,
        2: Direction.UP,
        3: Direction.DOWN,
    }
    return switcher.get(action, -1)


def q_learning(num_episodes, discount_factor=0.99, alpha=0.5, epsilon=0.2):
    Q = defaultdict(lambda: np.zeros(4))

    policy = make_epsilon_greedy_policy(Q, epsilon, 4)

    max_reward = 0
    ideal_path = []

    x = []
    y = []
    ma = []
    avg_reward = 0
    max_reward = 0

    for i_episode in range(num_episodes):
        path = []
        game = Game()
        state = game.get_state()

        total_reward = 0

        for t in itertools.count():
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            dir = parse_action(action)
            field_state, _ = game.move(dir)
            path.append(dir)
            reward, done = game.get_reward(field_state)
            total_reward += reward
            next_state = game.get_state()

            if total_reward > max_reward:
                max_reward = total_reward
                ideal_path = path

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]

            Q[state][action] += alpha * td_delta

            if done:
                if i_episode > num_episodes - 100:
                    policy = make_epsilon_greedy_policy(Q, 0, 4)
                avg_reward += total_reward
                x.append(i_episode)
                y.append(total_reward)
                ma.append(avg_reward / (i_episode + 1))
                print(f"episode: {i_episode} finished with reward: {total_reward}")
                break

            state = next_state
    return Q, ideal_path, max_reward, x, y, ma


Q, P, m, x, y, ma = q_learning(5000)

# print found solution

print(f"Best solution found with maximum reward of: {m}")
print("first agent")
game = Game()
for i in P:
    game.move(i)
    game.update_ui()
pass

plotly.tools.set_credentials_file(username='2mawi2', api_key='AylpnmyLc4ghzSemcCwM')
xy_data = go.Scatter(x=x, y=y, mode='markers', marker=dict(size=4), name='AAPL')
# vvv clip first and last points of convolution
mov_avg = go.Scatter(x=x[5:-4], y=ma[5:-4],
                     line=dict(width=2, color='red'), name='Moving average')
data = [xy_data, mov_avg]
try:
    plotly.plotly.iplot(data, filename='results')
except:
    pass
