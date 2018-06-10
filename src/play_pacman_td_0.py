import random

from src.direction import Direction
from src.game import Game
from hashlib import sha1
import numpy as np

from src.td0_agent import TD0Agent

episodes = 2000
update_ui = False

success = 0
avg_reward = 0
max_reward = 0
ideal_path = []
x = []
y = []
ma = []

td0Agent = TD0Agent(
    alpha=0.001,  # learning rate should go direction 0
    gamma=0.25,  # Diskontierungsfaktor
    epsilon=1,
    epsilon_decay=0.99
)

for e in range(episodes):
    game = Game()
    state = game.get_state()

    total_reward = 0

    path = []
    done = False
    while not done:
        action = td0Agent.get_action(state)
        field_state, _ = game.move(action)
        reward, done = game.get_reward(field_state)
        # game.update_ui()
        next_state = game.get_state()
        td0Agent.learn(state, next_state, reward, action)

        state = next_state
        # total_reward += reward

        if done:

            if total_reward > max_reward:
                max_reward = total_reward
                ideal_path = path

            if e >= episodes - 100:
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

print(f'RECENT SUCCESS RATE: {success}/{100}')
print(f'total avg reward rate {avg_reward / episodes}')
print(f'max reward rate {max_reward}')

# print found solution

print(f"Best solution found with maximum reward of: {max_reward}")
print("first agent")
game = Game()
for i in ideal_path:
    game.move(i)
    game.update_ui()
