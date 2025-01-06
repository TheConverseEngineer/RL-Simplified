from abc import ABC, abstractmethod

from ..bandit_machine import BanditMachine

class SlotAgent(ABC):
    """
    Represents an arbitrary agent that plays slot machines
    """

    def __init__(self, num_handles: int):
        """
        :param num_handles: the number of handles on slot machines that this agent plays
        """
        self.__num_handles = num_handles

    @property
    def num_handles(self):
        return self.__num_handles

    @abstractmethod
    def play(self, machine: BanditMachine) -> float:
        """
        Agent plays one round on the bandit slot machine

        :return: the payout from this round
        """
        raise NotImplementedError

