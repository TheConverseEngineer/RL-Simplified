from collections import deque

import numba
import numpy as np
import torch

CPU_DEVICE = torch.device('cpu')
MPS_DEVICE = torch.device('mps:0')


def verify_tensor_inner_dims(
        item: torch.Tensor, inner_dims: tuple[int, ...], 
        dtype: torch.dtype, device: torch.device,
        assertion_fail_name: str
):
    """
    Utility debug method that checks if item is a tensor with the specified inner
    dimensions (it is assumed that there is exactly one outer dimension) with the specified
    datatype and device. Throws an assertion if any check fails.

    :param item: tensor to be checked
    :param inner_dims: list of integers specifying the inner dimensions of the tensor
    :param dtype: the torch dtype that this tensor should have
    :param device: the torch device that this tensor should have
    :param assertion_fail_name: If an assertion fails, this will be the name used to describe what caused the error
    :raises AssertionError: if any check fails
    """

    assert isinstance(item, torch.Tensor), \
        f"{assertion_fail_name} was of type {type(item)}, expected torch.Tensor"
    assert item.shape[1:] == inner_dims, \
        f"{assertion_fail_name} inner dimensions were of shape {item.shape[1:]}, expected {inner_dims}"
    assert item.dtype == dtype, \
        f"{assertion_fail_name} tensor was of type {item.dtype}, expected {dtype}"
    assert item.device == device, \
        f"{assertion_fail_name} tensor was on device {item.device}, expected {device}"


def verify_tensor_dims(
        item: torch.Tensor, dims: tuple[int, ...],
        dtype: torch.dtype, device: torch.device,
        assertion_fail_name: str
):
    """
    Utility debug method that checks if item is a tensor with the specified dimensions
    with the specified datatype and device. Throws an assertion if any check fails.

    :param item: tensor to be checked
    :param dims: list of integers specifying the exact dimensions of the tensor
    :param dtype: the torch dtype that this tensor should have
    :param device: the torch device that this tensor should have
    :param assertion_fail_name: If an assertion fails, this will be the name used to describe what caused the error
    :raises AssertionError: if any check fails
    """

    assert isinstance(item, torch.Tensor), \
        f"{assertion_fail_name} was of type {type(item)}, expected torch.Tensor"
    assert item.shape == dims, \
        f"{assertion_fail_name} was of shape {item.shape}, expected {dims}"
    assert item.dtype == dtype, \
        f"{assertion_fail_name} tensor was of type {item.dtype}, expected {dtype}"
    assert item.device == device, \
        f"{assertion_fail_name} tensor was on device {item.device}, expected {device}"


@numba.jit(nopython=True) # Since we are just looping over a numpy array, numba jit works great here
def calculate_discounted_rewards(reward: np.ndarray, discount_factor: float) -> np.ndarray:
    """
    Given the raw reward at each step, this method calculates the total discounted reward at each step

    :param reward: A 1-d array representing the reward at each step
    :param discount_factor: The discount factor to use
    :return:       A 1-d array representing the total (present and future) discounted reward obtained at each step

    Credit: https://stackoverflow.com/questions/78923906/efficient-computation-of-sum-of-discounted-rewards-in-rl
    """
    n = reward.shape[0]
    discounted = np.zeros(n + 1) # The final item has a value 0 for padding
    for i in np.arange(n - 1, -1, -1):
        discounted[i] = reward[i] + discount_factor * discounted[i + 1]
    return discounted[:-1] # Ignore the last item

class RollingAverage:
    """Utility class that maintains the rolling average of some data value"""

    def __init__(self, window_size: int, default_value: float = -10000000):
        """
        Constructor for a utility class that maintains the rolling average of some data value

        :param window_size: The number of items to use when calculating the average
        """
        self.__queue = deque(maxlen=window_size)
        self.__window_size = window_size
        self.__sum = 0.0

        self._default_value = default_value

    def append(self, item: float) -> None:
        """
        Add a new datapoint to the rolling average

        :param item: The datapoint to add
        """
        self.__sum += item
        if len(self.__queue) == self.__window_size: self.__sum -= self.__queue[0]
        self.__queue.append(item)

    @property
    def average(self):
        """Returns the average of the last n data values (n is defined as the window_size in the constructor)"""
        if len(self.__queue) > 0:
            return self.__sum / len(self.__queue)
        else:
            return self._default_value
