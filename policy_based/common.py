import numpy as np
import torch
from torch import nn

from utils.exploration import AggregateStep


class PolicyGradientNetwork(nn.Module):
    """
    Represents the deep learning network used by our vanilla policy gradient
    """

    def __init__(self, input_size: int, num_actions: int):
        """
        Constructor for the PolicyGradientNetwork class
        This class represents the deep learning network used by our vanilla policy gradient
        \n
        Note that this network does NOT apply the softmax activation function

        :param input_size:  The size of the input
        :param num_actions:   The number of possible actions
        """
        super(PolicyGradientNetwork, self).__init__()

        # Thankfully, cartpole requires a fairly simple model
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
