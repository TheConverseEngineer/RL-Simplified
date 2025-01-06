from abc import ABC, abstractmethod
from typing import override

import numpy as np

class BanditMachine(ABC):
    """
    Represents an arbitrary "bandit" slot machine with n handles
    """

    def __init__(self, num_handles: int):
        """
        :param num_handles: the number of handles on this slot machine
        """
        self.__num_handles = num_handles

    @property
    def num_handles(self):
        """
        :return: The number of handles on this slot machine
        """
        return self.__num_handles

    @abstractmethod
    def pull(self, arm: int) -> float:
        """
        Equivalent to pulling a handle on this slot machine

        :param arm: the handle to pull
        :return:    the payout after pulling that handle
        """
        raise NotImplementedError()

    @abstractmethod
    def get_best_expected_value(self) -> float:
        """
        :return: Returns the optimal expected payout from the slot machine at this moment
        """
        raise NotImplementedError()

    def update(self) -> None:
        """
        Some implementations may require this method to be called every iteration
        :return:
        """
        pass


class WalkingBanditMachine(BanditMachine):
    """
    Represents a "bandit" slot machine with n handles where the average reward of each handle shifts over time
    """

    def __init__(self, num_handles: int, payout_variance: float, payout_step_variance):
        """
        Creates a Bandit Machine where the average returns for each handle change over time

        :param num_handles:             the number of handles
        :param payout_variance:         the standard deviation in the payout for any given turn
        :param payout_step_variance:    the standard deviation of how much each payout during an update
        """
        super().__init__(num_handles)
        self.__payout_variance = payout_variance
        self.__payout_step_variance = payout_step_variance

        self.rewards = np.ones(num_handles)

    @override
    def pull(self, arm: int) -> float:
        return np.random.normal(loc=self.rewards[arm], scale=self.__payout_variance)

    @override
    def update(self) -> None:
        """
        Randomly shift the reward of each handle
        """
        self.rewards += np.random.normal(loc=0, scale=self.__payout_step_variance, size=self.num_handles)

    @override
    def get_best_expected_value(self) -> float:
        return np.max(self.rewards)


class FixedBanditMachine(BanditMachine):
    """
    Represents a "bandit" slot machine with n handles where the payouts do not change
    """

    def __init__(self, num_handles: int, payout_mean_variance: float, payout_variance: float):
        """
        Creates a Bandit Machine where the average returns for each handle are fixed in advance

        :param num_handles:             the number of handles
        :param payout_mean_variance:    the standard deviation in the means of the payout
        :param payout_variance:         the standard deviation in any individual payout for any given turn
        """
        super().__init__(num_handles)
        self.__variance = payout_variance

        self.rewards = np.random.normal(loc=0, scale=payout_mean_variance, size=num_handles)

    @override
    def pull(self, arm: int) -> float:
        return np.random.normal(loc=self.rewards[arm], scale=self.__variance)

    @override
    def get_best_expected_value(self) -> float:
        return np.max(self.rewards)
