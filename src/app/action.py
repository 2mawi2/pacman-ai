from enum import Enum, IntEnum


class Action(IntEnum):
    __order__ = 'RIGHT LEFT UP DOWN'
    RIGHT = 0,
    LEFT = 1,
    UP = 2,
    DOWN = 3,
