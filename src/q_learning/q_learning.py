import plotly.graph_objs as go

import plotly
from src.app.action import Action
from src.app.game import Game
from src.app.statistics import Statistics
from src.q_learning.q_learning_agent import Agent

statistics = Statistics()


def q_learning(num_episodes, gamma=0.99, alpha=0.5, epsilon=0.1, epsilon_decay=0.001):
    agent = Agent(gamma, alpha, epsilon, epsilon_decay)

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

            td_delta = agent.learn(next_state, reward, state, action)

            path.append(action)
            total_reward += reward

            if done:
                agent.epsilon = agent.epsilon * 0.01 ** (1 / num_episodes)
                if i_episode > num_episodes - 100:
                    agent.epsilon = 0
                print(f"episode: {i_episode} finished with reward: {total_reward} epsilon: {agent.epsilon}")

            collect_stats(done, i_episode, path, total_reward, td_delta)

            state = next_state


def collect_stats(done, i_episode, path, total_reward, td_delta):
    if done:
        if total_reward > statistics.max_reward:
            statistics.max_reward = total_reward
            statistics.ideal_path = path
        statistics.avg_reward += total_reward
        statistics.x.append(i_episode)
        statistics.y.append(total_reward)
        statistics.td_delta.append(td_delta)
        statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))


def print_best_solution():
    print(f"Best solution found with maximum reward of: {statistics.max_reward}")
    game = Game()
    for i in statistics.ideal_path:
        game.move(i)
        game.update_ui()


def plot_data():
    plotly.tools.set_credentials_file(username='2mawi2', api_key='AylpnmyLc4ghzSemcCwM')
    xy_data = go.Scatter(x=statistics.x, y=statistics.y, mode='markers', marker=dict(size=4), name='reward')
    mov_avg = go.Scatter(x=statistics.x[5:-4], y=statistics.mean_average[5:-4],
                         line=dict(width=2, color='red'), name='Moving average')
    data = [xy_data, mov_avg]
    try:
        plotly.plotly.iplot(data, filename='results')
    except:
        pass


if __name__ == '__main__':
    q_learning(
        num_episodes=5_000,
        gamma=1,
        alpha=1,
        epsilon=1,
        epsilon_decay=0,
    )
    print_best_solution()
    #plot_data()
