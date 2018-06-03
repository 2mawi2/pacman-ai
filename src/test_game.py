from unittest import TestCase

from src.game import Game
from src.direction import Direction
from src.state import State


class TestGame(TestCase):
    game: Game = Game()

    def setUp(self):
        # reset field
        self.reset_game_space()

    def reset_game_space(self):
        self.game.field = [
            ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
            ["o", " ", " ", " ", " ", " ", "o", " ", "o", " ", " ", "o"],
            ["o", " ", "g", "o", "o", " ", "o", " ", "o", "x", " ", "o"],
            ["o", " ", "o", "o", "p", " ", "o", " ", "o", "o", " ", "o"],
            ["o", " ", "o", "o", " ", " ", "g", " ", "o", "o", " ", "o"],
            ["o", "o", "o", "o", "o", "o", "o", " ", "o", "o", " ", "d"],
        ]

    def test_find_pacman_index(self):
        self.game.field[3][4] = " "
        self.game.field[5][11] = "p"
        result = self.game.find_pacman_index()
        self.assertEqual(71, result)

    def test_find_pacman_index2(self):
        self.game.field[3][4] = " "
        self.game.field[0][0] = "p"
        result = self.game.find_pacman_index()
        self.assertEqual(0, result)

    def test_move_should_return_index_of_field(self):
        _, result = self.game.move(Direction.DOWN)
        self.assertEqual(52, result)

    def test_find_pacman(self):
        x, y = self.game.find_pacman()
        self.assertEqual(4, x)
        self.assertEqual(3, y)

    def test_move_field_cleaned_up(self):
        x_old, y_old = self.game.find_pacman()
        self.game.move(Direction.RIGHT)
        self.assertEqual(" ", self.game.field[y_old][x_old])

    def test_move_right(self):
        x_old, y_old = self.game.find_pacman()
        self.game.move(Direction.RIGHT)
        self.assertEqual("p", self.game.field[y_old][x_old + 1])

    def test_move_left(self):
        x_old, y_old = self.game.find_pacman()
        self.game.move(Direction.LEFT)
        self.assertEqual("p", self.game.field[y_old][x_old - 1])

    def test_move_up(self):
        x_old, y_old = self.game.find_pacman()
        self.game.move(Direction.UP)
        self.assertEqual("p", self.game.field[y_old - 1][x_old])

    def test_move_down(self):
        x_old, y_old = self.game.find_pacman()
        self.game.move(Direction.DOWN)

        self.assertEqual("p", self.game.field[y_old + 1][x_old])

    def set_pacman(self, x, y):
        self.game.field[3][4] = " "
        self.game.field[y][x] = "p"

    def test_move_wall(self):
        self.set_pacman(10, 3)
        field_type, idx = self.game.move(Direction.RIGHT)
        self.assertEqual(State.POINT, field_type)

    def test_does_not_move_out_of_boundaries_y(self):
        for _ in range(4):
            self.game.move(Direction.DOWN)
        x, y = self.game.find_pacman()
        self.assertEqual(len(self.game.field) - 1, y)

    def test_does_not_move_out_of_boundaries_x(self):
        for _ in range(20):
            self.game.move(Direction.LEFT)
        x, y = self.game.find_pacman()
        self.assertEqual(0, x)

    def test_move_should_return_point_state(self):
        result, _ = self.game.move(Direction.UP)
        self.assertEqual(State.POINT, result)

    def test_move_should_return_ghost_state(self):
        self.game.move(Direction.UP)
        self.game.move(Direction.LEFT)
        result, _ = self.game.move(Direction.LEFT)
        self.assertEqual(State.GHOST, result)

    def test_move_should_return_empty_state(self):
        result, _ = self.game.move(Direction.DOWN)
        self.assertEqual(State.EMPTY, result)

    def test_move_should_return_star_state(self):
        x, y = self.game.find_pacman()
        self.game.field[y][x + 1] = "x"
        result, _ = self.game.move(Direction.RIGHT)
        self.assertEqual(State.STAR, result)

    def test_move_should_return_door_state(self):
        x, y = self.game.find_pacman()
        self.game.field[y][x + 1] = "d"
        result, _ = self.game.move(Direction.RIGHT)
        self.assertEqual(State.DOOR, result)
