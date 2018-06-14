import plotly.graph_objs as go

import plotly

from src.app.direction import Direction
from src.app.game import Game
from src.td_lambda_zero.agent import Agent


class Statistics:
    ideal_path: [Direction] = []
    x: [int] = []
    y: [int] = []
    mean_average: [float] = []
    avg_reward: [float] = 0
    max_reward: [int] = 0


statistics = Statistics()


def td_learning(num_episodes, gamma=0.99, alpha=0.5, epsilon=0.1, epsilon_decay=0.001, alpha_decay=0.0):
    agent = Agent(gamma, alpha, epsilon, epsilon_decay, alpha_decay)

    for i_episode in range(num_episodes):
        path = []
        game = Game()
        state = game.get_state()

        total_reward = 0

        done = False
        while not done:
            action = agent.get_action(state)
            reward, done, next_state = game.move2(action)
            # game.update_ui()

            agent.learn(next_state, reward, state, action)

            path.append(action)
            total_reward += reward

            if total_reward > statistics.max_reward:
                statistics.max_reward = total_reward
                statistics.ideal_path = path

            if done:
                if i_episode > num_episodes - 100:
                    agent.epsilon = 0
                    statistics.avg_reward += total_reward
                statistics.x.append(i_episode)
                statistics.y.append(total_reward)
                statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))
                print(f"episode: {i_episode} finished with reward: {total_reward}")

            state = next_state


def print_best_solution():
    print(f"Best solution found with maximum reward of: {statistics.max_reward}")
    game = Game()
    for i in statistics.ideal_path:
        game.move(i)
        game.update_ui()


def plot_data():
    plotly.tools.set_credentials_file(username='2mawi2', api_key='AylpnmyLc4ghzSemcCwM')
    xy_data = go.Scatter(x=statistics.x, y=statistics.y, mode='markers', marker=dict(size=4), name='AAPL')
    mov_avg = go.Scatter(x=statistics.x[5:-4], y=statistics.mean_average[5:-4],
                         line=dict(width=2, color='red'), name='Moving average')
    data = [xy_data, mov_avg]
    try:
        plotly.plotly.iplot(data, filename='results')
    except:
        pass


def iterate_lambda_epsilon():
    global all_results, statistics
    alphas = [0.5 for _ in range(5)]  # list(reversed([0.5, 0.6, 0.7, 0.8, 0.9]))  #
    epsilon = [0.5 for _ in range(5)]
    all_results = []
    for _ in range(10):
        results = []
        for a, e in zip(alphas, epsilon):
            td_learning(
                num_episodes=1000,
                gamma=0.9,
                alpha=a,
                epsilon=e,
                epsilon_decay=0
            )
            results.append((a, e, statistics.max_reward))
            statistics = Statistics()

        all_results.append(results)

    print(all_results)


if __name__ == '__main__':
    # iterate_lambda_epsilon()
    td_learning(
        num_episodes=3000,
        gamma=0.9,
        alpha=1,
        epsilon=0.05,
        epsilon_decay=0,
        alpha_decay=0
    )
    print_best_solution()
    plot_data()

# [[(0.1, 0.1, 37), (0.25, 0.25, 39), (0.5, 0.5, 43), (0.75, 0.75, 41), (1, 1, 30)],
# [(0.1, 0.1, 37), (0.25, 0.25, 37), (0.5, 0.5, 41), (0.75, 0.75, 43), (1, 1, 30)],
# [(0.1, 0.1, 37), (0.25, 0.25, 41), (0.5, 0.5, 41), (0.75, 0.75, 43), (1, 1, 30)],
# [(0.1, 0.1, 37), (0.25, 0.25, 37), (0.5, 0.5, 43), (0.75, 0.75, 43), (1, 1, 30)],
# [(0.1, 0.1, 36), (0.25, 0.25, 39), (0.5, 0.5, 43), (0.75, 0.75, 39), (1, 1, 30)],
# [(0.1, 0.1, 37), (0.25, 0.25, 43), (0.5, 0.5, 43), (0.75, 0.75, 41), (1, 1, 30)],
# [(0.1, 0.1, 36), (0.25, 0.25, 41), (0.5, 0.5, 43), (0.75, 0.75, 43), (1, 1, 41)],
# [(0.1, 0.1, 37), (0.25, 0.25, 41), (0.5, 0.5, 41), (0.75, 0.75, 41), (1, 1, 30)],
# [(0.1,# 0.1, 39), (0.25, 0.25, 43), (0.5, 0.5, 41), (0.75, 0.75, 41), (1, 1, 30)],
# [(0.1, 0.1, 37), (0.25, 0.25, 41), (0.5, 0.5, 43), (0.75, 0.75, 43), (1, 1, 30)]]

# [[(0.1, 1, 41), (0.25, 0.75, 41), (0.5, 0.5, 43), (0.75, 0.25, 39), (1, 0.1, 41)],
# [(0.1, 1, 30), (0.25, 0.75, 41), (0.5, 0.5, 43), (0.75, 0.25, 39), (1, 0.1, 43)],
# [(0.1, 1, 30), (0.25, 0.75, 43), (0.5, 0.5, 41), (0.75, 0.25, 41), (1, 0.1, 39)],
# [(0.1, 1, 30), (0.25, 0.75, 30), (0.5, 0.5, 43), (0.75, 0.25, 41), (1, 0.1, 41)],
# [(0.1, 1, 41), (0.25, 0.75, 41), (0.5, 0.5, 41), (0.75, 0.25, 41), (1, 0.1, 43)],
# [(0.1, 1, 30), (0.25, 0.75, 41), (0.5, 0.5, 41), (0.75, 0.25, 41), (1, 0.1, 43)],
# [(0.1, 1, 37), (0.25, 0.75, 41), (0.5, 0.5, 41), (0.75, 0.25, 41), (1, 0.1, 42)],
# [(0.1, 1, 30), (0.25, 0.75, 41), (0.5, 0.5, 39), (0.75, 0.25, 43), (1, 0.1, 41)],
# [(0.1, 1, 30), (0.25, 0.75, 30), (0.5, 0.5, 43), (0.75, 0.25, 41), (1, 0.1, 41)],
# [(0.1, 1, 28), (0.25, 0.75, 30), (0.5, 0.5, 43), (0.75, #0.25, 41), (1, 0.1, 43)]]

# [[(0.5, 1, 41), (0.5, 0.75, 43), (0.5, 0.5, 41), (0.5, 0.25, 41), (0.5, 0.1, 41)],
# [(0.5, 1, 30), (0.5, 0.75, 43), (0.5, 0.5, 41), (0.5, 0.25, 41), (0.5, 0.1, 41)],
# [(0.5, 1, 30), (0.5, 0.75, 30), (0.5, 0.5, 43), (0.5, 0.25, 39), (0.5, 0.1, 41)],
# [(0.5, 1, 41), (0.5, 0.75, 41), (0.5, 0.5, 43), (0.5, 0.25, 43), (0.5, 0.1, 41)],
# [(0.5, 1, 30), (0.5, 0.75, 30), (0.5, 0.5, 41), (0.5, 0.25, 41), (0.5, 0.1, 41)],
# [(0.5, 1, 30), (0.5, 0.75, 43), (0.5, 0.5, 41), (0.5, 0.25, 43), (0.5, 0.1, 41)],
# [(0.5, 1, 30), (0.5, 0.75, 30), (0.5, 0.5, 39), (0.5, 0.25, 41), (0.5, 0.1, 41)],
# [(0.5, 1, 30), (0.5, 0.75, 41), (0.5, 0.5, 43), (0.5, 0.25, 43), (0.5, 0.1, 41)],
# [(0.5, 1, 30), (0.5, 0.75, 43), (0.5, 0.5, 43), (0.5, 0.25, 41), (0.5, 0.1, 43)],
# [(0.5, 1, 30), (0.5, 0.75, 43), (0.5, 0.5, 43), (0.5, 0.25, 40), (0.5, 0.1, 41)]]

# [[(1, 0.5, 43), (0.75, 0.5, 43), (0.5, 0.5, 39), (0.25, 0.5, 43), (0.1, 0.5, 39)],
# [(1, 0.5, 43), (0.75, 0.5, 43), (0.5, 0.5, 43), (0.25, 0.5, 41), (0.1, 0.5, 30)],
# [(1, 0.5, 41), (0.75, 0.5, 43), (0.5, 0.5, 41), (0.25, 0.5, 43), (0.1, 0.5, 41)],
# [(1, 0.5, 43), (0.75, 0.5, 41), (0.5, 0.5, 41), (0.25, 0.5, 39), (0.1, 0.5, 41)],
# [(1, 0.5, 41), (0.75, 0.5, 43), (0.5, 0.5, 43), (0.25, 0.5, 31), (0.1, 0.5, 30)],
# [(1, 0.5, 43), (0.75, 0.5, 43), (0.5, 0.5, 43), (0.25, 0.5, 41), (0.1, 0.5, 41)],
# [(1, 0.5, 41), (0.75, 0.5, 41), (0.5, 0.5, 43), (0.25, 0.5, 43), (0.1, 0.5, 32)],
# [(1, 0.5, 41), (0.75, 0.5, 43), (0.5, 0.5, 41), (0.25, 0.5, 43), (0.1, 0.5, 43)],
# [(1, 0.5, 41), (0.75, 0.5, 43), (0.5, 0.5, 43), (0.25, 0.5, 43), (0.1, 0.5, 29)],
# [(1, 0.5, 41), (0.75, 0.5, 43), (0.5, 0.5, 41), (0.25, 0.5, 43), (0.1, 0.5, 43)]]

# [[(0.9, 0.5, 43), (0.8, 0.5, 41), (0.7, 0.5, 43), (0.6, 0.5, 41), (0.5, 0.5, 43)],
# [(0.9, 0.5, 41), (0.8, 0.5, 43), (0.7, 0.5, 43), (0.6, 0.5, 43), (0.5, 0.5, 43)],
# [(0.9, 0.5, 43), (0.8, 0.5, 41), (0.7, 0.5, 39), (0.6, 0.5, 43), (0.5, 0.5, 43)],
# [(0.9, 0.5, 41), (0.8, 0.5, 41), (0.7, 0.5, 41), (0.6, 0.5, 41), (0.5, 0.5, 43)],
# [(0.9, 0.5, 41), (0.8, 0.5, 43), (0.7, 0.5, 43), (0.6, 0.5, 43), (0.5, 0.5, 39)],
# [(0.9, 0.5, 43), (0.8, 0.5, 41), (0.7, 0.5, 43), (0.6, 0.5, 43), (0.5, 0.5, 41)],
# [(0.9, 0.5, 43), (0.8, 0.5, 43), (0.7, 0.5, 41), (0.6, 0.5, 43), (0.5, 0.5, 41)],
# [(0.9, 0.5, 43), (0.8, 0.5, 41), (0.7, 0.5, 41), (0.6, 0.5, 39), (0.5, 0.5, 43)],
# [(0.9, 0.5, 43), (0.8, 0.5, 43), (0.7, 0.5, 41), (0.6, 0.5, 41), (0.5, 0.5, 41)],
# #[(0.9, 0.5, 43), (0.8, 0.5, 41), (0.7, 0.5, 43), (0.6, 0.5, 41), (0.5, 0.5, 41)]]
