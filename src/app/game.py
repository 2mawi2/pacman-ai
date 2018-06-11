from src.app.direction import Direction
from src.app.state import State
import numpy as np


class Game:

    def __init__(self) -> None:
        self.field = np.array([
            ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
            ["o", "W", "W", "W", "W", "W", "o", "W", "o", "W", "W", "o"],
            ["o", "W", "g", "o", "o", "W", "o", "W", "o", "x", "W", "o"],
            ["o", "W", "o", "o", "p", "W", "o", "W", "o", "o", "W", "o"],
            ["o", "W", "o", "o", "W", "W", "g", "W", "o", "o", "W", "o"],
            ["o", "o", "o", "o", "o", "o", "o", "W", "W", "W", "W", "d"],
        ])

    def update_ui(self):
        for line in self.field:
            for field in line:
                print(field + " ", end="")
            print(end="\n")

    def move(self, dir: Direction) -> (State, int):
        delta_x, delta_y = self._get_delta(dir)
        x, y = self.find_pacman()
        state: State = self._get_state(x + delta_x, y + delta_y)

        if state is State.WALL:
            delta_y, delta_x = 0, 0

        self.field[y, x] = " "
        self.field[y + delta_y, x + delta_x] = "p"

        index = self.get_index(x + delta_x, y + delta_y)

        return state, index

    def find_pacman(self) -> (int, int):
        x, y = np.where(self.field == "p")
        return y[0], x[0]

    def get_reward(self, next_state: State):
        switcher = {
            State.EMPTY: -1,
            State.DOOR: 0,
            State.STAR: 10,
            State.GHOST: -100,
            State.POINT: 1,
            State.WALL: -1,
        }
        game_over = next_state == State.DOOR or next_state == State.GHOST
        r = switcher.get(next_state, 0)
        return r, game_over

    def find_pacman_index(self) -> int:
        return np.where(self.field.flatten() == "p")[0]

    def get_index(self, x, y):
        return x + y * 12

    def _get_delta(self, d: Direction) -> (int, int):
        delta_x: int = 0
        delta_y: int = 0
        if d is Direction.RIGHT:
            delta_x = 1
        elif d is Direction.LEFT:
            delta_x = -1
        elif d is Direction.UP:
            delta_y = -1
        elif d is Direction.DOWN:
            delta_y = 1
        return delta_x, delta_y

    def _get_state(self, x, y) -> State:
        if y + 1 > len(self.field) or y < 0 or x + 1 > len(self.field[0]) or x < 0:
            return State.WALL
        switcher = {
            "o": State.POINT,
            "g": State.GHOST,
            "x": State.STAR,
            "d": State.DOOR,
            "W": State.WALL
        }
        return switcher.get(self.field[y, x][0], State.EMPTY)

    def update_ui_with_weights(self, weights: []):
        for y, line in enumerate(self.field):
            for x, _ in enumerate(line):
                if self.field[y, x] == "o":
                    index = self.get_index(x, y)
                    weigth = weights[index]
                    idx = np.argmax(weigth)
                    if not all([w == 0 for w in weigth]):
                        if idx == 0:  # right
                            self.field[y, x] = ">"
                        if idx == 1:  # left
                            self.field[y, x] = "<"
                        if idx == 2:  # up
                            self.field[y, x] = "^"
                        if idx == 3:  # down
                            self.field[y, x] = "▼"
        self.update_ui()

    def get_state(self):
        return hash(self.field.tostring())  # hash game_field for unique state id
