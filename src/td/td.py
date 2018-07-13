from src.app.game import Game
from src.td import plot_utils
from src.td.agent import Agent
from src.app.statistics import Statistics

statistics = Statistics()


def is_last_state(i_episode, num_episodes):
    return i_episode > num_episodes - 2


def td_learning(num_episodes, gamma, alpha):
    agent = Agent(gamma, alpha)

    for i_episode in range(num_episodes):
        game = Game()
        state = game.get_field_state()
        total_reward = 0
        done = False

        while not done:
            next_state = agent.get_greedy_state(game)
            reward, done = game.move_to_state(next_state)
            agent.learn(next_state, reward, state)

            #game.update_ui()
            total_reward += reward
            state = next_state
            collect_stats(done, i_episode, total_reward)

            if is_last_state(i_episode, num_episodes):
                game.update_ui()
            if done:
                print(f"episode: {i_episode} finished with reward: {total_reward}")


def collect_stats(done, i_episode, total_reward):
    if done:
        if total_reward > statistics.max_reward:
            statistics.max_reward = total_reward
        statistics.avg_reward += total_reward
        statistics.x.append(i_episode)
        statistics.y.append(total_reward)
        statistics.mean_average.append(statistics.avg_reward / (i_episode + 1))


if __name__ == '__main__':
    td_learning(
        num_episodes=200,
        gamma=0.83,
        alpha=1
    )
