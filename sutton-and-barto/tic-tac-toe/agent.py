from abc import ABC, abstractmethod
from typing import override

from state import State, Player
import numpy as np

class TicTacToePlayer(ABC):
    """
    Abstract class representing an arbitrary Tic-Tac-Toe player
    """

    @abstractmethod
    def play(self, current_state: State, playing_as: int) -> State:
        """
        Returns the next state that this player would make

        :param current_state: the current state of the board
        :param playing_as:  who this player is playing as
        :return:    the new state of the board
        """
        raise NotImplementedError()

def get_other(x: int) -> int:
    """
    Returns the "other player", ie 2 if inputted 1 and 1 if inputted 2
    """
    if x == 1:
        return 2
    else:
        return 1

class TicTacToeAgent(TicTacToePlayer):

    # noinspection PyTypeChecker
    def __init__(self, step_size: float, exploration_rate: float):
        """
        Create a TicTacToe agent that uses q-learning
        :param step_size:           between 0 and 1, higher means agent learns faster
        :param exploration_rate:    between 0 and 1, higher means agent chooses at random to "explore" more often
        """
        self.__q_table: dict[State, float] = dict()
        self.__step_size = step_size
        self.__exploration_rate = exploration_rate

    @override
    def play(self, current_state: State, playing_as: int) -> State:
        options = current_state.get_next_states(playing_as)

        for option in options:
            if option not in self.__q_table:
                self.__q_table[option] = option.calculate_reward()


        if np.random.random() < self.__exploration_rate:
            return np.random.choice(options)
        elif playing_as == 2:
            return max(options, key=lambda x: self.__q_table.get(x))
        else:
            return min(options, key=lambda x: self.__q_table.get(x))

    def train(self, playing_as: int, opponent: TicTacToePlayer) -> float:
        """
        Runs through a full game against the specified opponent and updates q-table values

        :param playing_as:  which side this agent should play as (see Player enum)
        :param opponent:    instance that this agent should play against (can be itself)
        :return:            1 if the agent won, 0 if it lost, and 0.5 if it tied
        """
        agent_turn = (playing_as == 1)

        current_state: State = State.empty_state()

        state_list: list[State] = []

        while True:
            state_list.append(current_state)
            if current_state not in self.__q_table:
                self.__q_table[current_state] = current_state.calculate_reward()

            if not agent_turn:
                current_state = opponent.play(current_state, get_other(playing_as))
            else:
                current_state = self.play(current_state, playing_as)

            agent_turn = not agent_turn

            if current_state.is_board_full() or current_state.calculate_reward() != 0.5:
                break

        if current_state not in self.__q_table:
            self.__q_table[current_state] = current_state.calculate_reward()

        result = current_state.calculate_reward()
        if result + 1 == playing_as:
            result = 1
        elif result != 0.5:
            result = 0

        for i in reversed(state_list):
            self.__q_table[i] += self.__step_size*(self.__q_table[current_state] - self.__q_table[i])
            current_state = i

        return result

class RandomTicTacToePlayer(TicTacToePlayer):
    """
    An implementation of TicTacToePlayer that always chooses a move at random
    """

    def play(self, current_state: State, playing_as: int) -> State:
        options = current_state.get_next_states(playing_as)
        return np.random.choice(options)

class PredictableTicTacToePlayer(TicTacToePlayer):
    """
   An implementation of TicTacToePlayer that always chooses the same move in a given scenario
   """

    def play(self, current_state: State, playing_as: int) -> State:
        options = current_state.get_next_states(playing_as)
        return options[0]
