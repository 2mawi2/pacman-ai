import plotly.graph_objs as go

import plotly

from src.app.action import Action
from src.app.game import Game
from src.td.agent import Agent
import numpy


class Statistics:
    x: [int] = []
    y: [int] = []
    mean_average: [float] = []
    avg_reward: [float] = 0
    max_reward: [int] = 0


statistics = Statistics()

all_collected_states = {}


def td_learning(num_episodes, gamma=0.99, alpha=0.5, epsilon=0.5):
    agent = Agent(gamma, alpha, epsilon)

    for i_episode in range(num_episodes):
        game = Game()
        state = game.get_field_state()
        all_collected_states[hash(state.tostring())] = state
        total_reward = 0

        done = False
        while not done:
            next_state, reward, done = agent.get_greedy_state(game, all_collected_states)

            next_state_hash = hash(next_state.tostring())

            if next_state_hash not in all_collected_states:
                all_collected_states[next_state_hash] = next_state
                print(agent.state_map[next_state_hash])

            total_reward += reward

            agent.learn(next_state, reward, state, done)  # update V
            game.field = next_state
            state = next_state

            if total_reward > statistics.max_reward:
                statistics.max_reward = total_reward

            if done:
                if i_episode > num_episodes - 100:
                    statistics.avg_reward += total_reward
                statistics.x.append(i_episode)
                statistics.y.append(total_reward)
                statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))
                # if i_episode % 100 == 0:
                print(f"episode: {i_episode} finished with reward: {total_reward}")

    return agent


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


def evaluate_policy_greedy(agent: Agent):
    game = Game()

    total_reward = 0
    done = False
    while not done:
        current_state = game.get_state()
        valid_states = agent.get_valid_states(all_collected_states, current_state)
        probs = [agent.V[hash(s.tostring())] for s in valid_states]
        best_next_state = max(zip(valid_states, probs), key=lambda i: i[1])[0]
        reward, done = game.move_to_state(best_next_state)
        game.update_ui()
        total_reward += reward

    print(f"policy evalutated with best reward {total_reward}")


if __name__ == '__main__':
    # iterate_lambda_epsilon()
    agent = td_learning(
        num_episodes=100000,
        gamma=1,
        alpha=1,
        epsilon=0.5,
    )
    # print_best_solution()
    # plot_data()
    evaluate_policy_greedy(agent)
