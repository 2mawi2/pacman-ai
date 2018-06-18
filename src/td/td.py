import plotly.graph_objs as go

import plotly

from src.app.direction import Direction
from src.app.game import Game
from src.td.agent import Agent
import numpy


class Statistics:
    ideal_path: [Direction] = []
    x: [int] = []
    y: [int] = []
    mean_average: [float] = []
    avg_reward: [float] = 0
    max_reward: [int] = 0


statistics = Statistics()

all_collected_states = {}


def td_learning(num_episodes, gamma=0.99, alpha=0.5):
    agent = Agent(gamma, alpha)

    for i_episode in range(num_episodes):
        path = []
        game = Game()
        state = game.get_state_field()

        total_reward = 0

        done = False
        while not done:
            action = agent.get_random_action()
            reward, done, _ = game.move2(action)
            next_state = game.get_state_field()
            agent.learn(next_state, reward, state)  # update V
            state_hash = hash(state.tostring())
            if state_hash not in all_collected_states:
                all_collected_states[state_hash] = state
            state = next_state

            path.append(action)
            total_reward += reward

            if total_reward > statistics.max_reward:
                statistics.max_reward = total_reward
                statistics.ideal_path = path

            if done:
                if i_episode > num_episodes - 100:
                    statistics.avg_reward += total_reward
                statistics.x.append(i_episode)
                statistics.y.append(total_reward)
                statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))
                print(f"episode: {i_episode} finished with reward: {total_reward}")

    return agent


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


def evaluate_policy_greedy(agent: Agent):
    game = Game()
    all_states_list = all_collected_states.values()

    total_reward = 0
    done = False
    while not done:
        valid_states = numpy.array(game.get_valid_states(all_states_list))
        probs = [agent.V[hash(s.tostring())] for s in valid_states]
        assert len(valid_states) == len(probs)

        best_next_state = max(zip(valid_states, probs), key=lambda i: i[1])[0]

        reward, done = game.move_to_state(best_next_state)
        game.update_ui()
        total_reward += reward

    print(f"policy evalutated with best reward {total_reward}")


if __name__ == '__main__':
    # iterate_lambda_epsilon()
    agent = td_learning(
        num_episodes=3000,
        gamma=0.9,
        alpha=1
    )
    # print_best_solution()
    # plot_data()
    evaluate_policy_greedy(agent)
