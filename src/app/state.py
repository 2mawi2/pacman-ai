from enum import Enum


class State(Enum):
    GHOST = 0,
    EMPTY = 1,
    POINT = 2,
    STAR = 3,
    WALL = 4,
    DOOR = 5,
