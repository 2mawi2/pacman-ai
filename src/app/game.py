from src.app.direction import Direction
from src.app.fieldtype import FieldType
import numpy as np
import collections


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

    def move2(self, dir: Direction) -> (int, bool, int):
        delta_x, delta_y = self._get_delta(dir)
        x, y = self.find_pacman()
        field_state: FieldType = self._get_field_type(x + delta_x, y + delta_y)

        if field_state is FieldType.WALL:
            delta_y, delta_x = 0, 0

        self.field[y, x] = " "
        self.field[y + delta_y, x + delta_x] = "p"

        next_state = self.get_state()
        reward, done = self.get_reward(field_state)

        return reward, done, next_state

    def move(self, dir: Direction) -> (FieldType, int):
        delta_x, delta_y = self._get_delta(dir)
        x, y = self.find_pacman()
        state: FieldType = self._get_field_type(x + delta_x, y + delta_y)

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
        switcher = {
            FieldType.EMPTY: -1,
            FieldType.DOOR: 0,
            FieldType.STAR: 10,
            FieldType.GHOST: -100,
            FieldType.POINT: 1,
            FieldType.WALL: -1,
        }
        game_over = field_type == FieldType.DOOR or field_type == FieldType.GHOST
        r = switcher.get(field_type, 0)
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

    def _get_field_type(self, x, y) -> FieldType:
        if y + 1 > len(self.field) or y < 0 or x + 1 > len(self.field[0]) or x < 0:
            return FieldType.WALL
        switcher = {
            "o": FieldType.POINT,
            "g": FieldType.GHOST,
            "x": FieldType.STAR,
            "d": FieldType.DOOR,
            "W": FieldType.WALL
        }
        return switcher.get(self.field[y, x][0], FieldType.EMPTY)

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

    def get_state_field(self):
        return np.copy(self.field)

    def get_valid_states(self, states):
        return [s for s in states if self._is_valid_state(s)]

    def _is_valid_state(self, s):
        x_before, y_before = self.find_pacman()
        after = np.where(s == "p")
        if len(after) == 0:
            print("invalid state")

        x_after, y_after = after[1][0], after[0][0]
        delta_x, delta_y = x_before - x_after, y_before - y_after
        has_pacman_moved_in_range = (delta_x == 1 or delta_x == 0) \
                                    and (delta_y == 1 or delta_y == 0) \
                                    and not delta_x == delta_y

        is_before_field_empty = s[y_before, x_before] == " "

        is_valid_number_of_os = self.validate_occurence(s, x_after, y_after, "o")
        is_valid_number_of_stars = self.validate_occurence(s, x_after, y_after, "x")
        is_valid_number_of_doors = self.validate_occurence(s, x_after, y_after, "d")

        return has_pacman_moved_in_range \
               and is_before_field_empty \
               and is_valid_number_of_os \
               and is_valid_number_of_stars \
               and is_valid_number_of_doors \

    def validate_occurence(self, s, x_after, y_after, item: str):
        n_o_after = collections.Counter(s.flatten())[item]
        n_o_before = collections.Counter(self.field.flatten())[item]
        if self.field[y_after, x_after] == item:
            return n_o_before - n_o_after == 1
        else:
            return n_o_before - n_o_after == 0

    def _get_reward_for_next_state(self, next_state) -> (int, bool):
        y, x = np.where(next_state == "p")
        next_field_type = self._get_field_type(x, y)
        reward, done = self.get_reward(next_field_type)
        return reward, done

    def move_to_state(self, next_state) -> (int, bool):
        if not self._is_valid_state(next_state):
            raise ValueError()

        reward, done = self._get_reward_for_next_state(next_state)

        self.field = next_state
        return reward, done
