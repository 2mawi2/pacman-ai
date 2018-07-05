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

            visited_states[next_state_hash] += 1

            if next_state_hash not in all_collected_states:
                all_collected_states[next_state_hash] = next_state

            total_reward += reward
            agent.learn(next_state, reward, state)
            state = next_state

            collect_stats(done, i_episode, num_episodes, total_reward, agent.alpha, agent.epsilon)

            if i_episode > num_episodes - 2:  # print last episode
                game.update_ui()

            if done:
                agent.epsilon = agent.epsilon - (1 / num_episodes)
                if i_episode > num_episodes - 100:
                    agent.epsilon = 0
                print(f"episode: {i_episode} finished with reward: {total_reward}")
    return agent


def collect_stats(done, i_episode, total_reward, td_delta, alpha, epsilon):
    if done:
        if total_reward > statistics.max_reward:
            statistics.max_reward = total_reward
        statistics.avg_reward += total_reward
        statistics.x.append(i_episode)
        statistics.y.append(total_reward)
        statistics.td_delta.append(td_delta)
        statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))
        statistics.alpha.append(alpha)
        statistics.epsilon.append(epsilon)


def plot_data():
    plotly.tools.set_credentials_file(username='2mawi2', api_key='AylpnmyLc4ghzSemcCwM')
    xy_data = go.Scatter(x=statistics.x, y=statistics.y, mode='markers', marker=dict(size=4), name='reward')
    mov_avg = go.Scatter(x=statistics.x[5:-4], y=statistics.mean_average[5:-4],
                         line=dict(width=2, color='red'), name='Moving reward average')
    epsilon = go.Scatter(x=statistics.x[5:-4], y=statistics.epsilon[5:-4],
                         line=dict(width=2, color='blue'), name='epsilon')
    alpha = go.Scatter(x=statistics.x[5:-4], y=statistics.alpha[5:-4],
                       line=dict(width=2, color='blue'), name='alpha')

    data = [xy_data, mov_avg, epsilon, alpha]

    plotly.plotly.iplot(data, filename='results')


if __name__ == '__main__':
    agent = td_learning(
        num_episodes=1000,
        gamma=1,
        epsilon=1,
    )
    plot_data()
