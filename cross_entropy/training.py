from typing import Generator

import torch
import torch.nn as nn

import numpy as np

import gymnasium as gym

from cross_entropy.policy import CrossEntropyPolicyNetwork
from cross_entropy.structs import Parameters, Episode, EpisodeStep


# noinspection DuplicatedCode
def visualize_run(env: gym.Env, policy: CrossEntropyPolicyNetwork) -> None:
    current_observation, _ = env.reset()
    softmax = nn.Softmax(dim=1)

    while True:
        # Transform the current observation into a tensor and un-squeeze it so that it is a 1-item batch
        observation_tensor = torch.tensor(current_observation, dtype=torch.float32)
        observation_tensor.unsqueeze_(0)

        # Pass the current observation through the policy network, and take the
        # softmax to determine the probability of each action
        # From the result, index i is the probability of performing action i
        action_probabilities = softmax(
            policy(observation_tensor)
        ).detach().numpy()[0] # We index 0 to get the first (and only) item in the batch (i.e. undo the un-squeeze operation)

        # Select a random action
        action = np.random.choice(len(action_probabilities), p=action_probabilities)

        # Perform the selected action
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        # Render
        env.render()

        if is_done or is_trunc:
            break

        else:
            current_observation = next_obs


# noinspection DuplicatedCode
def generate_training_batch(
        env: gym.Env,
        policy: CrossEntropyPolicyNetwork,
        params: Parameters
) -> Generator[list[Episode], None, None]:
    """
    Generator function that generates batches of episodes of a predetermined size

    Parameters used:
        - Batch Size: The number of episodes to generate in each call
        - Reward Discount Factor: the discount factor to use when calculating rewards

    :param env:     the OpenAI gym environment to use
    :param policy:  the policy to use when choosing actions
    :param params:  the parameters that dictate how batches are made (see method description for which parameters are used)
    :return:        yields a list (of size batch size) of episodes on each call
    """

    batch = []
    softmax = nn.Softmax(dim=1)

    # Set the initial observation, reward, and initialize the list of episode steps
    current_observation, _ = env.reset()
    current_reward = 0
    episode_steps = []

    while True:
        # Transform the current observation into a tensor and un-squeeze it so that it is a 1-item batch
        observation_tensor = torch.tensor(current_observation, dtype=torch.float32)
        observation_tensor.unsqueeze_(0)

        # Pass the current observation through the policy network, and take the
        # softmax to determine the probability of each action
        # From the result, index i is the probability of performing action i
        action_probabilities = softmax(
            policy(observation_tensor)
        ).detach().numpy()[0] # We index 0 to get the first (and only) item in the batch (i.e. undo the un-squeeze operation)

        # Select a random action
        action = np.random.choice(len(action_probabilities), p=action_probabilities)

        # Perform the selected action
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        # Update the reward for this episode
        current_reward = float(reward) + params.reward_discount_factor * current_reward

        # Add this episode step to the list
        episode_steps.append(
            EpisodeStep(observation=current_observation, action=action)
        )

        if is_done or is_trunc:
            # Generate the episode and add it to the batch
            episode = Episode(reward=current_reward, steps=episode_steps)
            batch.append(episode)

            # Reset the initial observation, reward, and list of episode steps
            current_observation, _ = env.reset()
            current_reward = 0
            episode_steps = []

            # Yield the training batch if we have enough samples
            if len(batch) >= params.batch_size:
                yield batch
                batch = []
        else:
            # Continue the episode
            current_observation = next_obs


def filter_training_batch(
        batch: list[Episode],
        params: Parameters
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Given a training batch, this method filters out the "best" episodes, and returns training data that
    can be used by the policy network based on those episodes.

    Parameters used:
        - Episode Retention Percentile: what percentile to use when selecting the "best" episodes

    For example, an episode retention percentile of 70 will keep the best 30% of episodes.

    Returns:
        1. A tensor representing the batched input to the policy
        2. A tensor representing the ideal output of the policy given the input specified in (1)
        3. The mean reward in this batch (for logging)
        4. The standard deviation of the reward in this batch (for logging)
        5. The cutoff for a "good reward" (for logging)

    :param batch:   a list of episodes
    :param params:  the parameters to use (see method description for which parameters are used)
    :return:        a tuple of values (see method description for what each value is)
    """

    # Extract the rewards from the episode list
    rewards = np.array(list(map(lambda x: x.reward, batch)))

    # Calculate the cutoff of what rewards to keep
    reward_cutoff = float(np.percentile(rewards, params.episode_retention_percentile))

    # Calculate the mean and standard deviation of the reward (just for monitoring)
    _reward_mean = float(np.mean(rewards))
    _reward_std = float(np.std(rewards))

    # For these lists item i in train_actions represents a "good" response to item i in train_observations
    train_observations: list[np.ndarray] = []
    train_actions: list[int] = []

    for episode in batch:
        if episode.reward < reward_cutoff: # Ignore "bad" episodes
            continue

        train_observations.extend(map(lambda x: x.observation, episode.steps))
        train_actions.extend(map(lambda x: x.action, episode.steps))


    # Now convert the training sets to tensors and return them, along with the tracking metrics
    return (
        torch.tensor(np.vstack(train_observations), dtype=torch.float32),
        torch.tensor(train_actions, dtype=torch.long),
        _reward_mean,
        _reward_std,
        reward_cutoff
    )
