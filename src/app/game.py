from src.app.action import Action
from src.app.fieldtype import FieldType
import numpy as np
import collections

field_to_reward = {
    FieldType.EMPTY: -1,
    FieldType.DOOR: 0,
    FieldType.STAR: 10,
    FieldType.GHOST: -100,
    FieldType.POINT: 1,
    FieldType.WALL: -1,
}

value_to_fieldtype = {
    "o": FieldType.POINT,
    "g": FieldType.GHOST,
    "x": FieldType.STAR,
    "d": FieldType.DOOR,
    "W": FieldType.WALL,
    " ": FieldType.EMPTY
}


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

    def move2(self, dir: Action) -> (int, bool, int):
        delta_x, delta_y = self._get_delta(dir)
        x, y = self.find_pacman()
        field_state: FieldType = self.get_field_type(x + delta_x, y + delta_y)

        if field_state is FieldType.WALL:
            delta_y, delta_x = 0, 0

        self.field[y, x] = " "
        self.field[y + delta_y, x + delta_x] = "p"

        next_state = self.get_state()
        reward, done = self.get_reward(field_state)

        return reward, done, next_state

    def move(self, dir: Action) -> (FieldType, int):
        delta_x, delta_y = self._get_delta(dir)
        x, y = self.find_pacman()
        state: FieldType = self.get_field_type(x + delta_x, y + delta_y)

        if state is FieldType.WALL:
            delta_y, delta_x = 0, 0

        self.field[y, x] = " "
        self.field[y + delta_y, x + delta_x] = "p"

        index = self.get_index(x + delta_x, y + delta_y)

        return state, index

    def find_pacman(self) -> (int, int):
        y, x = np.where(self.field == "p")
        return x[0], y[0]

    def get_reward(self, field_type: FieldType):
        game_over = field_type == FieldType.DOOR or field_type == FieldType.GHOST
        r = field_to_reward.get(field_type, 0)
        return r, game_over

    def find_pacman_index(self) -> int:
        return np.where(self.field.flatten() == "p")[0]

    def get_index(self, x, y):
        return x + y * 12

    def _get_delta(self, d: Action) -> (int, int):
        delta_x: int = 0
        delta_y: int = 0
        if d is Action.RIGHT:
            delta_x = 1
        elif d is Action.LEFT:
            delta_x = -1
        elif d is Action.UP:
            delta_y = -1
        elif d is Action.DOWN:
            delta_y = 1
        return delta_x, delta_y

    def get_field_type(self, x, y) -> FieldType:
        if y >= 6 or y < 0 or x >= 12 or x < 0:
            return FieldType.WALL
        return value_to_fieldtype[self.field[y, x][0]]

    def get_state(self):
        return hash(self.field.tostring())  # hash game_field for unique state id

    def get_field_state(self):
        return np.copy(self.field)

    def validate_occurence(self, s, x_after, y_after, item: str):
        n_o_after = collections.Counter(s.flatten())[item]
        n_o_before = collections.Counter(self.field.flatten())[item]
        if self.field[y_after, x_after] == item:
            return n_o_before - n_o_after == 1
        else:
            return n_o_before - n_o_after == 0

    def get_reward_for_next_state(self, next_state) -> (int, bool):
        y, x = np.where(next_state == "p")
        next_field_type = self.get_field_type(x, y)
        reward, done = self.get_reward(next_field_type)
        return reward, done

    def move_to_state(self, next_state) -> (int, bool):
        reward, done = self.get_reward_for_next_state(next_state)
        self.field = np.copy(next_state)
        return reward, done

    def get_valid_actions(self) -> [Action]:
        valid_actions = []
        for d in Action:
            delta_x, delta_y = self._get_delta(d)
            x, y = self.find_pacman()
            field_type: FieldType = self.get_field_type(x + delta_x, y + delta_y)
            if field_type is not FieldType.WALL:
                valid_actions.append(d)
        return valid_actions
