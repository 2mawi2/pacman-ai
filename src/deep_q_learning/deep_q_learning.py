from collections import OrderedDict

from src.deep_q_learning.deep_q_learning_agent import Agent
from src.app.action import Action
from src.app.game import Game
from src.app.fieldtype import FieldType
import numpy as np
import plotly.graph_objs as go
import plotly

n_actions: int = 4
n_states: int = 10_000
episodes = 1000
update_ui = False

agent_first = Agent(
    n_actions=n_actions,  # for right, left, up, down
    n_states=n_states,  # for 10_000 game fields
    discount=0.999,
    alpha=1,  # used for gradient descent optimization
    epsilon=1,  # exploration rate should be between 0 and 1, higher -> more random decissions are taken
    epsilon_decay=0.9999,  # reduction of exploration rate for every epoche
    lambda_=0.25
)


def get_reward(next_state: FieldType):
    switcher = {
        FieldType.EMPTY: -1,
        FieldType.DOOR: 0,
        FieldType.STAR: 10,
        FieldType.GHOST: -100,
        FieldType.POINT: 1,
        FieldType.WALL: -1,
    }
    game_over = next_state == FieldType.DOOR or next_state == FieldType.GHOST
    r = switcher.get(next_state, 0)
    return r, game_over


def parse_action(action: int) -> Action:
    switcher = {
        0: Action.RIGHT,
        1: Action.LEFT,
        2: Action.UP,
        3: Action.DOWN,
    }
    return switcher.get(action, -1)


def parse_state(state: FieldType) -> int:
    switcher = {
        FieldType.GHOST: 0,
        FieldType.EMPTY: 1,
        FieldType.POINT: 2,
        FieldType.STAR: 3,
        FieldType.WALL: 4,
        FieldType.DOOR: 5,
    }
    return switcher.get(state, -1)


state_dict = OrderedDict()


def package_state(s):
    state_dict[s] = None
    s = list(state_dict.keys()).index(s)
    s = convert_to_one_hot(s)
    s = s.reshape(1, -1)
    return s


def convert_to_one_hot(state_number):
    s = np.zeros((1, n_states))
    s[0][state_number] = 1
    return s


success = 0

avg_reward = 0

max_reward = 0
ideal_path = []

x = []
y = []
ma = []

for e in range(episodes):
    game = Game()  # reset game
    state = package_state(game.get_state())  # init S
    total_reward = 0  # init e = 0

    path = []

    done = False
    while not done:
        action, greedy = agent_first.get_e_greedy_action(state)

        i = parse_action(action)
        path.append(i)  # append all path to the movement
        field_type, next_index = game.move(i)

        if update_ui:
            game.update_ui()
        reward, done = get_reward(field_type)

        index = game.find_pacman_index()
        next_state = package_state(game.get_state())  # next_index:int -> to vector of weigth

        agent_first.learn(state, action, next_state, reward, greedy)

        state = next_state
        total_reward += reward

        if done:

            if total_reward > max_reward:
                max_reward = total_reward
                ideal_path = path

            if e >= episodes - 100:
                if field_type == FieldType.DOOR:
                    success += 1

            avg_reward += total_reward
            x.append(e)
            y.append(total_reward)
            ma.append(avg_reward / (e + 1))
            #
            if reward == 0:  # game has been terminated by door
                print(f"episode: {e}/{episodes}, score: {total_reward:.2f} and goal has been found!")

            else:  # game has been terminated by ghost
                print(f"episode: {e}/{episodes}, score: {total_reward:.2f}")
            break

agent_first.reset_e_trace()
agent_first.print_weights()
print(f'RECENT SUCCESS RATE: {success}/{100}')
print(f'total avg reward rate {avg_reward / episodes}')
print(f'max reward rate {max_reward}')

# plot result

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

# print found solution

print(f"Best solution found with maximum reward of: {max_reward}")
print("first agent")
game = Game()
for i in ideal_path:
    game.move(i)
    game.update_ui()

Game().update_ui_with_weights(agent_first.get_weights())
print("second agent")

game = Game()
for i in ideal_path:
    game.move(i)
    game.update_ui()
