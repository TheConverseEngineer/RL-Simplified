import numpy as np
from typing import Self
from enum import Enum

HASH_POW_TABLE = 3 ** (np.arange(9).reshape(3, 3))

class Player(Enum):
    X = 2
    O = 1

class State:

    def __init__(self, board: np.ndarray):
        """
        Represents the state of a tic-tac-toe table as a numpy array

        Each element in the array is 0: empty, 1: O, or 2: X
        :param board:
        """
        self.__board = board

    def __eq__(self, other: Self) -> bool:
        """Determines if two tic-tac-toe states are equal"""
        return np.all(self.__board == other.__board)

    @property
    def board(self) -> np.ndarray:
        """Returns the backing numpy array"""
        return self.__board

    def __hash__(self) -> int:
        """
        Calculates a unique hash for every tic-tac-toe board
        """

        return int(np.sum(self.board * HASH_POW_TABLE))

    def __str__(self) -> str:
        """
        Returns a formatted string arrangement of the tic-tac-toe game
        """
        str_board = [[' ' for _2 in range(3)] for _1 in range(3)]
        for i in range(3):
            for j in range(3):
                str_board[i][j] = 'O' if (self.board[i, j] == 1) else ('X' if (self.board[i, j] == 2) else ' ')

        return str('\n'.join(map(str, str_board)))

    def get_next_states(self, current_player: int) -> tuple[Self, ...]:
        """
        Returns a tuple of states where each state is a state that comes after this state given the current player

        :param current_player:  the current player (see Player enum)
        :return: a tuple of next states
        """
        indices = np.transpose(np.nonzero(self.board == 0))

        options = tuple(State(self.board.copy()) for _ in range(len(indices)))
        for i, index in enumerate(indices):
            options[i].__board[*index] = current_player

        return options

    @staticmethod
    def empty_state():
        """
        Returns a new state representing an empty board
        """
        return State(np.zeros((3, 3)))

    def is_board_full(self) -> bool:
        """
        Returns true if every spot on this board is taken
        """
        return not np.any(self.board == 0)

    def calculate_reward(self) -> float:
        """
        Calculates the reward for this state of the tic-tac-toe game

        :return: 1 if X wins, 0 if O wins, and 0.5 otherwise
        """
        for i in range(3):
            if np.all(self.board[i] == 2): return 0
            elif np.all(self.board[i] == 1): return 1

            elif np.all(self.board[:,i] == 2): return 0
            elif np.all(self.board[:,i] == 1): return 1

        if self.board[0][0] == self.board[1][1] == self.board[2][2]:
            if self.board[0][0] == 2: return 0
            elif self.board[0][0] == 1: return 1

        return 0.5
