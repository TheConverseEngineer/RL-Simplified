import math
import random
from collections import deque
from typing import Sequence

import numba
import numpy as np
import torch
from torch import nn

"""An extension of DQN with a prioritized replay buffer, dueling networks"""

CPU_DEVICE = torch.device('cpu')
"""Represents the generic cross-platform CPU device"""

MPS_DEVICE = torch.device('mps')
"""Represents the Metal Performance Shader GPU type on Apple Silicon devices """


class SizedBuffer:
    """
    Compared to collections.deque, this has a significantly faster random selection
    operation at the price of a slower append operation
    \n
    This is a fixed size buffer which supports querying a random batch
    """

    def __init__(self, buffer_length: int):
        self.__backing_array = [None] * buffer_length
        self.__current_end_pointer = 0
        self.__buffer_length = buffer_length

    def append(self, item):
        """
        Add an item to the end of the buffer, overwriting the earliest added item if the buffer is full.

        :param item: The item to add
        """
        self.__backing_array[self.__current_end_pointer] = item
        self.__current_end_pointer = (1 + self.__current_end_pointer) % self.__buffer_length

    def get_random_content(self, batch_size: int) -> list:
        """
        Returns a list of items randomly sampled without replacement from the buffer
        Only works if the buffer is full

        :param batch_size: The number of items to sample
        :return: The batch of randomly selected items.
        """
        return random.sample(self.__backing_array, batch_size)


class RollingAverage:
    """Utility class that maintains the rolling average of some data value"""

    def __init__(self, window_size: int):
        """
        Constructor for a utility class that maintains the rolling average of some data value

        :param window_size: The number of items to use when calculating the average
        """
        self.__queue = deque(maxlen=window_size)
        self.__window_size = window_size
        self.__sum = 0.0

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
        return self.__sum / len(self.__queue)


@numba.jit(nopython=True) # Since we are just looping over a numpy array, numba jit works great here
def calculate_total_rewards(reward: np.ndarray, discount_factor: float) -> np.ndarray:
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


class PrioritizedReplayBuffer:
    """
        Compared to SizedBuffer, this class uses a segment tree to provide efficient
        prioritized sampling
        \n
        This is a fixed size buffer which supports querying a random prioritized batch
        """

    def __init__(self, buffer_length: int, beta_increment: float, alpha: float = 0.6, beta_start: float = 0):
        self.__backing_array = [None] * buffer_length
        self.__priority_tree = np.zeros((buffer_length*2, 2)) # Stores sum, min
        self.__current_end_pointer = 0
        self.__buffer_length = buffer_length

        self.__alpha = alpha
        self.__beta = beta_start
        self.__beta_increment = beta_increment
        self.__max_priority = 1.0

    def update_beta(self):
        self.__beta = min(self.__beta + self.__beta_increment, 1.0)

    def __set_priority(self, idx: int, priority: float):
        """Note that idx should be the index in the buffer array, not the tree, i.e. in the range (0, length]"""
        self.__priority_tree[self.__buffer_length + idx][0] = priority
        self.__priority_tree[self.__buffer_length + idx][1] = priority

        self.__propagate_value(self.__buffer_length + idx)

        self.__max_priority = max(self.__max_priority, priority)

    def __propagate_value(self, tree_idx: int):
        """Note that tree idx should be the index in the actual tree, i.e. length + buffer index"""
        tree_idx //= 2
        while tree_idx > 0:
            self.__priority_tree[tree_idx, 0] = self.__priority_tree[tree_idx * 2, 0] + self.__priority_tree[tree_idx * 2 + 1, 0]
            self.__priority_tree[tree_idx, 1] = min(self.__priority_tree[tree_idx * 2, 1], self.__priority_tree[tree_idx * 2 + 1, 1])
            tree_idx //= 2

    def find_prefix_sum_idx(self, desired_sum):
        """
        Finds the first node whose prefix sum is greater than or equal to the desired sum
        :return: index in the tree
        """
        current_node_idx = 1
        while current_node_idx < self.__buffer_length:
            if current_node_idx * 2 + 1 < self.__buffer_length * 2 and self.__priority_tree[current_node_idx, 0] < desired_sum:

                desired_sum -= self.__priority_tree[current_node_idx, 0]
                current_node_idx = current_node_idx * 2 + 1
            else:
                current_node_idx = current_node_idx * 2

        return current_node_idx


    @property
    def total_priority(self) -> float:
        return self.__priority_tree[1, 0]

    @property
    def min_priority(self) -> float:
        return self.__priority_tree[1, 1]

    @property
    def max_priority(self) -> float:
        return self.__max_priority

    def append(self, item):
        """
        Add an item to the end of the buffer, overwriting the earliest added item if the buffer is full.

        :param item: The item to add
        """
        self.__backing_array[self.__current_end_pointer] = item
        self.__set_priority(self.__current_end_pointer, self.max_priority ** self.__alpha)
        self.__current_end_pointer = (1 + self.__current_end_pointer) % self.__buffer_length

    def get_random_content(self, batch_size: int) -> tuple[list, list, np.ndarray]:
        """
        Returns a list of items randomly sampled without replacement from the buffer
        Only works if the buffer is full
        Uses item priorities when sampling

        :param batch_size: The number of items to sample
        :return: The batch of randomly selected items, followed by their indices in the buffer, followed by their weights
        """
        query_ids = [self.find_prefix_sum_idx(self.total_priority * random.random()) for _ in range(batch_size)]

        max_weight = (self.min_priority/self.total_priority * self.__buffer_length) ** (-self.__beta)

        weights = []
        for idx in query_ids:
            p_sample = self.__priority_tree[idx, 0] / self.total_priority
            weight = (p_sample * self.__buffer_length) ** (-self.__beta)
            weights.append(weight / max_weight)
        sample = [self.__backing_array[idx - self.__buffer_length] for idx in query_ids]

        return sample, query_ids, np.array(weights, dtype=np.float32)

    def update_priorities(self, ids, priorities):
        for idx, priority in zip(ids, priorities):
            self.__set_priority(idx - self.__buffer_length, priority ** self.__alpha)



# Copied from torchRL
class NoisyLinear(nn.Linear):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
    be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
    with gradient descent along with any other remaining network weights. Factorized Gaussian
    noise is the type of noise usually employed.


    Args:
        in_features (int): input features dimension
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to ``None`` (default pytorch dtype)
        std_init (scalar, optional): initial value of the Gaussian standard deviation before optimization.
            Defaults to ``0.1``

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int | torch.Size | Sequence) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self) -> torch.Tensor:
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> torch.Tensor | None:
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None