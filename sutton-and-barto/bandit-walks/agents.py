from bandit_machine import BanditMachine

import numpy as np

from abc import ABC, abstractmethod

class Agent(ABC):
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


class SampleAverageAgent(Agent):
    """
    A slot machine-playing agent that chooses the handle with the highest average payout across all attempts
    """

    def __init__(self, num_handles: int, exploration_rate: float):
        """
        :param num_handles:         the number of handles on slot machines that this agent plays
        :param exploration_rate:    the chance that the agent may make a random move in order to foster exploration
        """
        super().__init__(num_handles)

        self.__averages = np.zeros(num_handles, dtype=float)
        self.__attempts = np.zeros(num_handles, dtype=np.int32)

        self.__exploration_rate = exploration_rate

    @property
    def exploration_rate(self):
        """
        :return: The exploration rate for this agent
        """
        return self.__exploration_rate

    def play(self, machine: BanditMachine) -> float:
        if np.random.rand() < self.exploration_rate: # pick a random move
            a = np.random.randint(0, self.num_handles)
        else:
            a = np.argmax(self.__averages)

        result = machine.pull(a)

        # Allows us to update the average without having to store every value
        self.__attempts[a] += 1
        self.__averages[a] += (1/self.__attempts[a])*(result - self.__averages[a])

        return result


class SteppingAgent(Agent):
    """
    A slot machine-playing agent that calculates expected payout by weighting the most recent attempt more than past attempts
    """

    def __init__(self, num_handles: int, exploration_rate: float, alpha: float):
        """
        :param num_handles:         the number of handles on slot machines that this agent plays
        :param exploration_rate:    the chance that the agent may make a random move in order to foster exploration
        :param alpha:               the step size (ie, how much recent values should be preferred over older ones)
        """
        super().__init__(num_handles)

        self.__values = np.zeros(num_handles)

        self.__exploration_rate = exploration_rate
        self.__alpha = alpha

    @property
    def exploration_rate(self):
        """
        :return: The exploration rate for this agent
        """
        return self.__exploration_rate

    @property
    def alpha(self):
        """
        :return: the step size (ie, how much recent values should be preferred over older ones)
        """
        return self.__alpha

    def play(self, machine: BanditMachine) -> float:
        if np.random.rand() < self.exploration_rate: # pick a random move
            a = np.random.randint(0, self.num_handles)
        else:
            a = np.argmax(self.__values)

        result = machine.pull(a)

        # Allows us to update the average without having to store every value
        self.__values[a] += self.alpha * (result - self.__values[a])

        return result


class OptimisticAgent(SteppingAgent):

    def __init__(self, num_handles: int, exploration_rate: float, alpha: float):
        """
        :param num_handles:         the number of handles on slot machines that this agent plays
        :param alpha:               the step size (ie, how much recent values should be preferred over older ones)
        """
        super().__init__(num_handles, exploration_rate, alpha)
        self.__values = np.full((num_handles,), 5)

class UpperConfidenceBoundAgent(Agent):

    def __init__(self, num_handles: int, c: float, alpha: float):
        """
        :param num_handles:         the number of handles on slot machines that this agent plays
        :param c:                   the upper confidence bound selection rate
        :param alpha:               the step size (ie, how much recent values should be preferred over older ones)
        """
        super().__init__(num_handles)

        self.__c = c
        self.__alpha = alpha
        self.__pulls = 1

        self.__averages = np.zeros(num_handles, dtype=float)
        self.__attempts = np.ones(num_handles, dtype=np.int32)

    @property
    def c(self):
        """
        :return: the upper confidence bound selection rate
        """
        return self.__c

    @property
    def alpha(self):
        """
        :return: the step size (ie, how much recent values should be preferred over older ones)
        """
        return self.__alpha

    def play(self, machine: BanditMachine) -> float:
        ucb = self.__averages + self.c*np.sqrt(np.log(self.__pulls)/self.__attempts)

        a = int(np.argmax(ucb))

        self.__pulls += 1
        self.__attempts[a] += 1

        result = machine.pull(a)

        # Allows us to update the average without having to store every value
        self.__averages[a] += self.alpha * (result - self.__averages[a])

        return result