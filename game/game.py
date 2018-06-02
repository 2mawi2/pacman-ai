import os
from enum import Enum


class Direction(Enum):
    RIGHT = 0,
    LEFT = 1,
    UP = 2,
    DOWN = 3,


class State(Enum):
    GHOST = 0,
    EMPTY = 1,
    POINT = 2,
    STAR = 3,
    WALL = 4,
    DOOR = 5,


class Game:
    field = [
        ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
        ["o", " ", " ", " ", " ", " ", "o", " ", "o", " ", " ", "o"],
        ["o", " ", "g", "o", "o", " ", "o", " ", "o", "x", " ", "o"],
        ["o", " ", "o", "o", "p", " ", "o", " ", "o", "o", " ", "o"],
        ["o", " ", "o", "o", " ", " ", "g", " ", "o", "o", " ", "o"],
        ["o", "o", "o", "o", "o", "o", "o", " ", "o", "o", " ", "d"],
    ]

    def update_ui(self):
        for line in self.field:
            for field in line:
                print(field + " ", end="")
            print(end="\n")

    def move(self, dir: Direction) -> State:
        delta_x, delta_y = self._get_delta(dir)
        x, y = self.find_pacman()
        state: State = self._get_state(x + delta_x, y + delta_y)

        if state is State.WALL:
            delta_y, delta_x = 0, 0

        self.field[y][x] = " "
        self.field[y + delta_y][x + delta_x] = "p"

        return state

    def find_pacman(self) -> (int, int):
        for y, line in enumerate(self.field):
            for x, field in enumerate(line):
                if field is "p":
                    return x, y

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
        if y + 1 > len(self.field) or y < 0 or x + 1 > len(self.field) or x < 0:
            return State.WALL
        switcher = {
            "o": State.POINT,
            "g": State.GHOST,
            "x": State.STAR,
            "d": State.DOOR,
        }
        return switcher.get(self.field[y][x], State.EMPTY)
