from collections import defaultdict

import plotly.graph_objs as go
import plotly
from src.app.game import Game
from src.td.agent import Agent
from src.app.statistics import Statistics

statistics = Statistics()

all_collected_states = {}


def td_learning(num_episodes, gamma=0.99, alpha=1.0, epsilon=0.5, epsilon_decay=1.0):
    agent = Agent(gamma, alpha, epsilon, epsilon_decay)

    for i_episode in range(num_episodes):
        game = Game()
        state = game.get_field_state()
        all_collected_states[hash(state.tostring())] = state
        total_reward = 0
        done = False

        visited_states = defaultdict(lambda: 0)
        visited_states[hash(state.tostring())] += 1
        while not done:
            next_state, reward, done = agent.get_greedy_state_and_move(game, visited_states)

            next_state_hash = hash(next_state.tostring())

            # game.update_ui()

            visited_states[next_state_hash] += 1

            if next_state_hash not in all_collected_states:
                all_collected_states[next_state_hash] = next_state

            total_reward += reward
            agent.learn(next_state, reward, state)
            state = next_state

            collect_stats(done, i_episode, total_reward, agent.alpha, agent.epsilon)

            if i_episode > num_episodes - 2:  # print last episode
                game.update_ui()

            if done:
                # agent.epsilon = agent.epsilon = agent.epsilon * 0.01 ** (1 / num_episodes)
                if i_episode > num_episodes - 100:
                    agent.epsilon = 0
                print(f"episode: {i_episode} finished with reward: {total_reward} with epsilon: {agent.epsilon}")
    return agent


def collect_stats(done, i_episode, total_reward, alpha, epsilon):
    if done:
        if total_reward > statistics.max_reward:
            statistics.max_reward = total_reward
        statistics.avg_reward += total_reward
        statistics.x.append(i_episode)
        statistics.y.append(total_reward)
        statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))
        statistics.alpha.append(alpha)
        statistics.epsilon.append(epsilon)


def plot_data():
    plotly.tools.set_credentials_file(username='2mawi2', api_key='AylpnmyLc4ghzSemcCwM')
    xy_data = go.Scatter(x=statistics.x, y=statistics.y, mode='markers',
                         marker=dict(size=8), name='reward', yaxis="y1")
    mov_avg = go.Scatter(x=statistics.x[5:-8], y=statistics.mean_average[5:-8],
                         line=dict(width=2, color='red'), name='Moving reward average', yaxis="y1")
    data = [xy_data, mov_avg]

    plotly.plotly.iplot(data, filename='results')


if __name__ == '__main__':
    agent = td_learning(
        num_episodes=500,
        gamma=0.83,
        epsilon=0,
        alpha=1
    )
    plot_data()
