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


def q_learning(num_episodes, discount_factor=0.99, alpha=0.5, epsilon=0.2):
    agent = Agent(discount_factor, alpha, epsilon)

    for i_episode in range(num_episodes):
        path = []
        game = Game()
        state = game.get_state()

        total_reward = 0

        done = False
        while not done:
            action = agent.get_action(state)
            field_state, _ = game.move(action)
            path.append(action)
            reward, done = game.get_reward(field_state)
            total_reward += reward
            next_state = game.get_state()

            if total_reward > statistics.max_reward:
                statistics.max_reward = total_reward
                statistics.ideal_path = path

            agent.learn(next_state, reward, state, action)

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


if __name__ == '__main__':
    q_learning(5000)
    print_best_solution()
    plot_data()
