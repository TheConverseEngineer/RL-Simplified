from abc import ABC, abstractmethod

import torch

from rlite.utils import CPU_DEVICE


class Agent(ABC):

    @abstractmethod
    def choose_actions(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Agent abstract method choose_actions was not implemented")

    @property
    def device(self) -> torch.device:
        return CPU_DEVICE
