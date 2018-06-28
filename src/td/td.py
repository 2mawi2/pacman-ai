import plotly.graph_objs as go
import plotly
from src.app.game import Game
from src.td.agent import Agent


class Statistics:
    x: [int] = []
    y: [int] = []
    mean_average: [float] = []
    avg_reward: [float] = 0
    max_reward: [int] = 0


statistics = Statistics()

all_collected_states = {}


def td_learning(num_episodes, gamma=0.99, alpha=0.5, epsilon=0.5, epsilon_decay=0.009):
    agent = Agent(gamma, alpha, epsilon, epsilon_decay)

    for i_episode in range(num_episodes):
        game = Game()
        state = game.get_field_state()
        all_collected_states[hash(state.tostring())] = state
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = agent.get_greedy_state_and_move(game, all_collected_states)
            # game.update_ui()
            next_state_hash = hash(next_state.tostring())
            if next_state_hash not in all_collected_states:
                all_collected_states[next_state_hash] = next_state
            total_reward += reward
            agent.learn(next_state, reward, state)  # update V
            state = next_state
            if i_episode > num_episodes - 100:
                agent.epsilon = 0

            if total_reward > statistics.max_reward:
                statistics.max_reward = total_reward

            if done:
                if i_episode > num_episodes - 100:
                    statistics.avg_reward += total_reward
                statistics.x.append(i_episode)
                statistics.y.append(total_reward)
                statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))
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


if __name__ == '__main__':
    agent = td_learning(
        num_episodes=100000,
        gamma=0.95,
        epsilon=1,
        epsilon_decay=0.999999
    )
    evaluate_policy_greedy(agent)
