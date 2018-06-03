from src.agent import Agent
from src.direction import Direction
from src.game import Game
from src.state import State
import numpy as np

n_actions: int = 4
n_states: int = 72

agent = Agent(
    n_actions=n_actions,  # for right, left, up, down
    n_states=n_states,  # for 72 game fields
    discount=0.95,
    alpha=0.05,
    epsilon=1,
    epsilon_decay=0.99,
    lamb=0.5
)


def get_reward(next_state: State):
    switcher = {
        State.EMPTY: -1,
        State.DOOR: 0,
        State.STAR: 10,
        State.GHOST: -100,
        State.POINT: 2,
        State.WALL: -1,
    }
    return switcher.get(next_state, 0), next_state == State.DOOR


def parse_action(action: int) -> Direction:
    switcher = {
        0: Direction.RIGHT,
        1: Direction.LEFT,
        2: Direction.UP,
        3: Direction.DOWN,
    }
    return switcher.get(action, -1)


def parse_state(state: State) -> int:
    switcher = {
        State.GHOST: 0,
        State.EMPTY: 1,
        State.POINT: 2,
        State.STAR: 3,
        State.WALL: 4,
        State.DOOR: 5,
    }
    return switcher.get(state, -1)


def package_state(s):
    s = convert_to_one_hot(s)
    s = s.reshape(1, -1)
    return s


def convert_to_one_hot(state_number):
    s = np.zeros((1, n_states))
    s[0][state_number] = 1
    return s


success = 0
episodes = 100
for e in range(episodes):
    game = Game()
    state = package_state(game.find_pacman_index())
    total_reward = 0
    done = False

    while not done:
        action, greedy = agent.get_e_greedy_action(state)
        direction = parse_action(action)

        field_type, next_index = game.move(direction)
        game.update_ui()
        reward, done = get_reward(field_type)
        next_state = package_state(next_index)  # next_index:int -> to vector of weights
        agent.learn(state, action, next_state, reward, greedy)

        if done:
            print(f'episode: {e}/{episodes}, score: {total_reward:.2f} pacman found the door')
            success += 1
            break

            # if e >= episodes - 100:
            #    success += 1
            # else:
            #    print(f"episode: {e}/{episodes}, score: {total_reward:.2f}")
            # break

        state = next_state
        total_reward += reward

    agent.reset_e_trace()

agent.print_weights()
print(f'RECENT SUCCESS RATE: {success}/{100}')
