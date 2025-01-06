import torch
import torch.nn as nn

from cross_entropy.structs import Parameters


class CrossEntropyPolicyNetwork(nn.Module):
    """
    Simple 2 layer neural network that we use to represent our policy
    """

    def __init__(self, params: Parameters):
        """
        Parameters used:
            - observation_space_size: the size of the observation space
            - action_space_size: the size of the action space
            - hidden_layer_size: the size of the (single) hidden layer in this network

        :param params: The parameters to use (see above for which parameters are used here)
        """
        super(CrossEntropyPolicyNetwork, self).__init__()
        self.__model = nn.Sequential(
            nn.Linear(params.observation_space_size, params.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(params.hidden_layer_size, params.action_space_size),
        )

    @property
    def model(self):
        """Returns the underlying Sequential model"""
        return self.__model

    def forward(self, x: torch.Tensor):
        return self.model(x)
