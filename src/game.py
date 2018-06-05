from src.direction import Direction
from src.state import State


class Game:
    def __init__(self) -> None:
        self.field = [
            ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
            ["o", "W", "W", "W", "W", "W", "o", "W", "o", "W", "W", "o"],
            ["o", "W", "g", "o", "o", "W", "o", "W", "o", "x", "W", "o"],
            ["o", "W", "o", "o", "p", "W", "o", "W", "o", "o", "W", "o"],
            ["o", "W", "o", "o", "W", "W", "g", "W", "o", "o", "W", "o"],
            ["o", "o", "o", "o", "o", "o", "o", "W", "o", "o", "W", "d"],
        ]

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

        self.field[y][x] = " "
        self.field[y + delta_y][x + delta_x] = "p"

        index = self._get_index(x + delta_x, y + delta_y)

        return state, index

    def find_pacman(self) -> (int, int):
        for y, line in enumerate(self.field):
            for x, field in enumerate(line):
                if field is "p":
                    return x, y

    def find_pacman_index(self) -> (int, int):
        x, y = self.find_pacman()
        return self._get_index(x, y)

    def _get_index(self, x, y):
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
        return switcher.get(self.field[y][x], State.EMPTY)
