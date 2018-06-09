import random

from src.direction import Direction
from src.game import Game

episodes = 200
update_ui = False

success = 0
avg_reward = 0
max_reward = 0
ideal_path = []
x = []
y = []
ma = []


class TD0Agent:
    def __init__(self, alpha: float, lambda_: float):
        self.lambda_ = lambda_
        self.alpha = alpha

    def get_next_action(self) -> Direction:
        return random.choice([Direction.RIGHT,
                              Direction.LEFT,
                              Direction.DOWN,
                              Direction.UP])


td0Agent = TD0Agent(
    alpha=0.25,  # learning rate
    lambda_=0.25,  # Diskontierungsfaktor
)

for e in range(episodes):
    game = Game()
    pacman_index = game.find_pacman_index()
    total_reward = 0

    path = []
    done = False
    while not done:
        state = game.field
        action = td0Agent.get_next_action()
        game.move(action)
        next_state = game.field
        reward = game.get_field_reward(state, action, next_state, reward, greedy)
        # V(sk ) ← V(sk ) + α (c(sk ) + V(sk+1) − V(sk ))
        action, greedy = agent_first.get_e_greedy_action(state)

        i = parse_action(action)
        path.append(i)  # append all path to the movement
        field_type, next_index = game.move(i)
        # if e % 100 == 0:
        if update_ui:
            game.update_ui()
        reward, done = game.get_reward(field_type)

        index = game.find_pacman_index()
        next_state = package_state(next_index)  # next_index:int -> to vector of weigth

        # if initial_visit:
        agent_first.learn(state, action, next_state, reward, greedy)
        # else:
        #    agent_second.learn(state, action, next_state, reward, greedy)

        state = next_state
        total_reward += reward

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
