from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    observation_space_size: int
    action_space_size: int
    hidden_layer_size: int
    batch_size: int
    episode_retention_percentile: float
    batch_size: int
    reward_discount_factor: float
    learning_rate: float


@dataclass
class EpisodeStep:
    """Represents a single "step" in an episode (consisting of the initial observation and the resultant action)"""
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    """Represents a single training episode (consisting of a list of episode steps and a total reward)"""
    reward: float
    steps: list[EpisodeStep]
