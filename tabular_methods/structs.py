from dataclasses import dataclass
from enum import Enum


@dataclass
class Parameters:
    grid_n: int
    gamma: float
    test_episodes: float


State = int

def coord_to_state(x: int, y: int, params: Parameters) -> State:
    """
    Takes the coordinates representing the current position on the grid, and converts it into a state
    """
    return x * params.grid_n + y


def state_to_coord(state: State, params: Parameters) -> tuple[int, int]:
    """
    Given a state, converts it into coordinates
    """
    return state // params.grid_n, state % params.grid_n


class Action(Enum):
    """
    Represents the action space in this environment
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @staticmethod
    def from_int(val: int):
        """
        Converts an integer into the corresponding action
        :param val:
        :return:
        """
        match val:
            case Action.UP.value: return Action.UP
            case Action.RIGHT.value: return Action.RIGHT
            case Action.DOWN.value: return Action.DOWN
            case Action.LEFT.value: return Action.LEFT

        print(f"ERROR, unknown action {val} given")
        return Action.UP

