import random
from dataclasses import dataclass
from typing import Generator, Self

import gymnasium as gym
import numpy as np
import torch
from collections import deque

from rlite.agent import Agent
from rlite.utils import CPU_DEVICE, verify_tensor_dims, verify_tensor_inner_dims, calculate_discounted_rewards, \
    RollingAverage


class SimpleExperienceBatch:
    """
    Simple immutable dataclass which stores a set of experiences
    """

    def __init__(
            self,
            observations: torch.Tensor, actions: torch.Tensor,
            rewards: torch.Tensor, is_complete: torch.Tensor
    ):
        if __debug__:
            assert isinstance(observations, torch.Tensor), \
                f"Observations must be of type torch.Tensor, received {type(observations)}"
            assert isinstance(actions, torch.Tensor), \
                f"Actions must be of type torch.Tensor, received {type(actions)}"
            assert isinstance(rewards, torch.Tensor), \
                f"Rewards must be of type torch.Tensor, received {type(rewards)}"
            assert isinstance(is_complete, torch.Tensor), \
                f"Is-Complete must be of type torch.Tensor, received {type(rewards)}"
            assert observations.shape[0] == actions.shape[0] == rewards.shape[0] == is_complete.shape[0], \
                (f"All parameters must have the same outer dimensions, received {observations.shape[0]}, "
                 f"{actions.shape[0]}, {rewards.shape[0]}, {is_complete.shape[0]}")
            assert observations.device == actions.device == rewards.device == is_complete.device, \
                (f"All parameters must be on the same device, received {observations.device}, "
                 f"{actions.device}, {rewards.device}, {is_complete.device}")
            assert len(rewards.shape) == 2 and rewards.shape[1] == 1, \
                f"Rewards must have a single inner dimension of size 1, received dimensions {rewards.shape}"
            assert len(is_complete.shape) == 2 and is_complete.shape[1] == 1, \
                f"Is-Complete must have a single inner dimension of size 1, received dimensions {is_complete.shape}"
            assert rewards.dtype == torch.float32, \
                f"Rewards must be of type torch.float32, received {rewards.dtype}"
            assert is_complete.dtype == torch.bool, \
                f"Is-Complete must be of type torch.bool, received {is_complete.dtype}"

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._is_complete = is_complete

    @property
    def observations(self) -> torch.Tensor:
        return self._observations

    @property
    def actions(self) -> torch.Tensor:
        return self._actions

    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards

    @property
    def is_complete(self) -> torch.Tensor:
        return self._is_complete

    def __len__(self) -> int:
        return self._rewards.shape[0]

    def __getitem__(self, item: slice) -> Self:
        """Convenience method that returns a new SimpleExperienceBatch with a slice of the original data"""
        if __debug__:
            assert isinstance(item, slice), \
                f"SimpleExperienceBatch only supports getting items via slice, received {type(item)}"

        return SimpleExperienceBatch(
            self._observations[item],
            self._actions[item],
            self._rewards[item],
            self._is_complete[item]
        )


class SimpleAggregateExperienceBatch(SimpleExperienceBatch):
    """
    Simple immutable dataclass which stores a set of aggregate experiences
    """

    def __init__(
            self,
            observations: torch.Tensor, actions: torch.Tensor,
            rewards: torch.Tensor, is_complete: torch.Tensor, final_observations: torch.Tensor
    ):
        if __debug__:
            assert isinstance(final_observations, torch.Tensor), \
                f"Final observations must be of type torch.Tensor, received {type(observations)}"
            assert observations.shape == final_observations.shape, \
                (f"Final observations must have the same shape as observations, received "
                 f"{final_observations.shape} and {observations.shape}")
            assert final_observations.device == observations.device, \
                (f"Final observations must be on the correct device, received "
                 f"{final_observations.device} and {observations.device}")
            assert observations.dtype == final_observations.dtype, \
                (f"Observations and final observations must be of the same type, received "
                 f"{type(observations)} and {type(final_observations)}")

        super(SimpleAggregateExperienceBatch, self).__init__(observations, actions, rewards, is_complete)
        self._final_observations = final_observations

    @property
    def final_observations(self) -> torch.Tensor:
        return self._final_observations

    def __getitem__(self, item: slice) -> Self:
        """Convenience method that returns a new SimpleExperienceBatch with a slice of the original data"""
        if __debug__:
            assert isinstance(item, slice), \
                f"SimpleExperienceBatch only supports getting items via slice, received {type(item)}"

        return SimpleAggregateExperienceBatch(
            self._observations[item],
            self._actions[item],
            self._rewards[item],
            self._is_complete[item],
            self._final_observations[item]
        )


class ExperienceBatch:
    """
    This class represents a fixed-max-size batch of experiences from one or more episode.
    Unlike SimpleExperienceBatch, this class is intended to be appended to and sampled from

    Each experience stores the initial observation, the action taken, the reward received,
    and whether the episode was terminated/truncated after this episode.
    """

    def __init__(
            self, size: int,
            observation_shape: tuple[int, ...], action_shape: tuple[int, ...],
            observation_dtype: torch.dtype = torch.float32,
            action_dtype: torch.dtype = torch.int32,
            device: torch.device = CPU_DEVICE,
    ):
        if __debug__:
            assert isinstance(size, int) and size > 0, \
                f"Inputted size was {size}, must be an integer greater than 0"
            assert isinstance(observation_shape, tuple), \
                f"Observation shape must be a tuple with at least one element, received type {type(observation_shape)}"
            assert isinstance(action_shape, tuple), \
                f"Action shape must be a tuple with at least one element, received type {type(action_shape)}"
            assert isinstance(observation_dtype, torch.dtype), \
                f"Observation dtype must be of type torch.dtype, received type {type(observation_dtype)}"
            assert isinstance(action_dtype, torch.dtype), \
                f"Action dtype must be of type torch.dtype, received type {type(action_dtype)}"
            assert isinstance(device, torch.device), \
                f"Device must be of type torch.device, received type {type(device)}"

        self.observations = torch.zeros((size, *observation_shape), dtype=observation_dtype, device=device)
        """A tensor consisting of observations. The outermost dimension is the batch size."""
        self.actions = torch.zeros((size, *action_shape), dtype=action_dtype, device=device)
        """A tensor consisting of actions. The outermost dimension is the batch size."""
        self.rewards = torch.zeros((size, 1), dtype=torch.float32, device=device)
        """A tensor consisting of the rewards received in each experience. 
        The shape is batch-size x 1 (there is an un-flattened dimension)"""
        self.is_complete = torch.zeros((size, 1), dtype=torch.bool, device=device)
        """A tensor consisting of whether or not each experience terminated that given episode. 
        The shape is batch-size x 1 (there is an un-flattened dimension)"""

        self._device = device
        """Represents the device that all tensors are stored on"""
        self._current_tail_ptr = 0
        """Represents the index where the next experience should be written to"""

        self.__INDICES_FOR_SAMPLING: np.ndarray | None = None
        """Lazily instantiated constant list used to sample random distinct indices"""

    @property
    def device(self) -> torch.device:
        return self._device

    def __len__(self) -> int:
        return self.rewards.shape[0]

    def append_single_experience(
            self, observation: torch.tensor, action: torch.tensor,
            reward: float, is_complete: bool
    ) -> None:
        """
        Appends a single experience to the batch, overwriting a previous experience if needed

        :param observation: The observation of the current experience. Note that this observations
                            must be a tensor with the exact shape, type, and device specified in the
                            constructor
        :param action:      The action for the current experience. Note that this action must be a
                            tensor with the exact shape, type, and device specified in the constructor
        :param reward:      The undiscounted reward received for the current experience.
        :param is_complete: Whether this experience terminated or truncated the episode.
        """
        if __debug__:
            verify_tensor_dims(
                observation, self.observations.shape[1:], self.observations.dtype,
                self._device, 'Observation'
            )
            verify_tensor_dims(
                action, self.actions.shape[1:], self.actions.dtype,
                self._device, 'Action'
            )
            assert isinstance(reward, float), \
                f"Reward must be of type float, received type {type(reward)}"
            assert isinstance(is_complete, bool), \
                f"Is_complete must be of type bool, received type {type(is_complete)}"

        self.observations[self._current_tail_ptr] = observation
        self.actions[self._current_tail_ptr] = action
        self.rewards[self._current_tail_ptr] = reward
        self.is_complete[self._current_tail_ptr] = is_complete

        self._current_tail_ptr = (self._current_tail_ptr + 1) % self.rewards.shape[0]

    def append_multiple_experiences(
            self, observation_batch: torch.tensor, action_batch: torch.tensor,
            reward_batch: torch.Tensor, is_complete_batch: torch.Tensor
    ) -> None:
        """
        Appends multiple experience to the batch, overwriting previous experiences if needed.
        If the number of provided experiences it greater than n (where n is the length of the batch),
        then only the final n experiences will be stored.

        This method is functionally identical to append_experience_batch, albeit with different parameters.

        :param observation_batch: The observations from each experience. Note that this parameter must be a
                                tensor with the exact type and device specified in the constructor. The
                                outermost dimensions of the tensor should separate different experiences, and
                                the remaining dimensions should match the shape specified in the constructor.

        :param action_batch: The actions from each experience. Note that this parameter must be a tensor with
                                the exact type and device specified in the constructor. The outermost dimensions
                                of the tensor should separate different experiences, and the remaining dimensions
                                should match the shape specified in the constructor.

        :param reward_batch: The undiscounted reward received from each experience. Note that this parameter must
                                be a tensor of type torch.float32, and that it must be stored on the device
                                specified in the constructor. The outer dimension of this tensor should separate
                                different experiences, and there should be a second dimension of 1 (unflatten the
                                tensor if necessary).

        :param is_complete_batch: Whether this experience terminated or truncated the episode. Note that this
                                parameter must be a tensor of type torch.bool, and that it must be stored on the
                                device specified in the constructor. The outer dimension of this tensor should
                                separate different experiences, and there should be a second dimension of 1
                                (unflatten the tensor if necessary).
        """
        if __debug__:
            verify_tensor_inner_dims(
                observation_batch, self.observations.shape[1:], self.observations.dtype,
                self._device, 'Observation batch'
            )
            verify_tensor_dims(
                action_batch, (observation_batch.shape[0], *self.actions.shape[1:]),
                self.actions.dtype, self._device, 'Action batch'
            )
            verify_tensor_dims(
                reward_batch, (observation_batch.shape[0], 1),
                torch.float32, self._device, 'Reward batch'
            )
            verify_tensor_dims(
                is_complete_batch, (observation_batch.shape[0], 1),
                torch.bool, self._device, 'Is-Complete batch'
            )

        if reward_batch.shape[0] > self.rewards.shape[0]:
            # Truncate the batch if we try to append more items than the max size
            start = reward_batch.shape[0] - self.rewards.shape[0]
            observation_batch = observation_batch[start:]
            action_batch = action_batch[start:]
            reward_batch = reward_batch[start:]
            is_complete_batch = is_complete_batch[start:]

            # Increment tail pointer as if we added those items
            self._current_tail_ptr = (self._current_tail_ptr + start) % self.rewards.shape[0]


        if self._current_tail_ptr + reward_batch.shape[0] <= self.rewards.shape[0]:
            end = self._current_tail_ptr + reward_batch.shape[0]
            # No wrapping required
            self.observations[self._current_tail_ptr:end] = observation_batch
            self.actions[self._current_tail_ptr:end] = action_batch
            self.rewards[self._current_tail_ptr:end] = reward_batch
            self.is_complete[self._current_tail_ptr:end] = is_complete_batch
        else:
            # need to wrap
            append_end_size = self.rewards.shape[0] - self._current_tail_ptr
            self.observations[self._current_tail_ptr:] = observation_batch[:append_end_size]
            self.actions[self._current_tail_ptr:] = action_batch[:append_end_size]
            self.rewards[self._current_tail_ptr:] = reward_batch[:append_end_size]
            self.is_complete[self._current_tail_ptr:] = is_complete_batch[:append_end_size]

            remaining_item_count = reward_batch.shape[0] - append_end_size
            self.observations[:remaining_item_count] = observation_batch[append_end_size:]
            self.actions[:remaining_item_count] = action_batch[append_end_size:]
            self.rewards[:remaining_item_count] = reward_batch[append_end_size:]
            self.is_complete[:remaining_item_count] = is_complete_batch[append_end_size:]

        # Increment the tail pointer
        self._current_tail_ptr = (self._current_tail_ptr + reward_batch.shape[0]) % self.rewards.shape[0]

    def append_experience_batch(self, batch: SimpleExperienceBatch) -> None:
        """
        Appends multiple experience to the batch, overwriting previous experiences if needed.
        If the number of provided experiences it greater than n (where n is the length of the batch),
        then only the final n experiences will be stored.

        This method is functionally identical to append_multiple_experiences, albeit with different parameters.

        :param batch: The batch of experiences to add. Note that the observation and action shapes and types must
                      match what was specified in the constructor. All tensors must also be on the correct device
        """

        if __debug__:
            verify_tensor_inner_dims(
                batch.observations, self.observations.shape[1:],
                self.observations.dtype, self._device, 'Observations'
            )
            verify_tensor_inner_dims(
                batch.actions, self.actions.shape[1:],
                self.actions.dtype, self._device, 'Actions'
            )
            # Everything else is already verified by the SimpleExperienceBatch class


        if len(batch) > self.rewards.shape[0]:
            # Truncate the batch if we try to append more items than the max size
            start = len(batch) - self.rewards.shape[0]
            batch = batch[start:]

            # Increment tail pointer as if we added those items
            self._current_tail_ptr = (self._current_tail_ptr + start) % self.rewards.shape[0]

        if self._current_tail_ptr + len(batch) <= self.observations.shape[0]:
            end = self._current_tail_ptr + len(batch)
            # No wrapping required
            self.observations[self._current_tail_ptr:end] = batch.observations
            self.actions[self._current_tail_ptr:end] = batch.actions
            self.rewards[self._current_tail_ptr:end] = batch.rewards
            self.is_complete[self._current_tail_ptr:end] = batch.is_complete
        else:
            # need to wrap
            append_end_size = self.observations.shape[0] - self._current_tail_ptr
            self.observations[self._current_tail_ptr:] = batch.observations[:append_end_size]
            self.actions[self._current_tail_ptr:] = batch.actions[:append_end_size]
            self.rewards[self._current_tail_ptr:] = batch.rewards[:append_end_size]
            self.is_complete[self._current_tail_ptr:] = batch.is_complete[:append_end_size]

            remaining_item_count = len(batch) - append_end_size
            self.observations[:remaining_item_count] =  batch.observations[append_end_size:]
            self.actions[:remaining_item_count] = batch.actions[append_end_size:]
            self.rewards[:remaining_item_count] = batch.rewards[append_end_size:]
            self.is_complete[:remaining_item_count] = batch.is_complete[append_end_size:]

        # Increment the tail pointer
        self._current_tail_ptr = (self._current_tail_ptr + len(batch)) % self.observations.shape[0]

    def generate_random_minibatches(self, num_batches: int) -> \
            Generator[tuple[SimpleExperienceBatch, torch.Tensor], None, None]:
        """
        Divides the buffer into random batches and yields them one at a time.
        Note that every item in this buffer will be yielded exactly once.
        This method will only work if the buffer is full.

        :param num_batches: The number of batches to produce (must be positive and less than the buffer size)
        :return: A generator that yields both the random minibatches and the indices of these experiences
        in the full buffer
        """
        if __debug__:
            assert isinstance(num_batches, int), f"num_batches must be an integer, received {type(num_batches)}"
            assert 0 < num_batches < self.rewards.shape[0], \
                (f"num_batches must be a positive integer less than the buffer size of {self.rewards.shape[0]}, "
                 f"received {num_batches}")

        indices = torch.randperm(self.rewards.shape[0])

        minibatch_size = self.rewards.shape[0] // num_batches
        for starting_index in range(0, self.rewards.shape[0], minibatch_size):
            idxes = indices[starting_index:starting_index + minibatch_size]
            yield SimpleExperienceBatch(
                self.observations[idxes],
                self.actions[idxes],
                self.rewards[idxes],
                self.is_complete[idxes],
            ), idxes

    def randomly_select_batch(self, batch_size: int) -> \
            tuple[SimpleExperienceBatch, torch.Tensor]:

        if self.__INDICES_FOR_SAMPLING is None:
            self.__INDICES_FOR_SAMPLING = list(range(self.rewards.shape[0]))

        idxes = torch.as_tensor(
            np.asarray(random.sample(self.__INDICES_FOR_SAMPLING, batch_size)),
            dtype=torch.int64,
        )

        return SimpleExperienceBatch(
            self.observations[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.is_complete[idxes],
        ), idxes


class AggregateExperienceBatch(ExperienceBatch):

    def __init__(
        self, size: int,
        observation_shape: tuple[int, ...], action_shape: tuple[int, ...],
        observation_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.int32,
        device: torch.device = CPU_DEVICE,
    ):
        super(AggregateExperienceBatch, self).__init__(
            size, observation_shape, action_shape,
            observation_dtype, action_dtype, device
        )

        self.final_observations = torch.zeros((size, *observation_shape), dtype=observation_dtype, device=device)

    def append_single_experience(
            self, observation: torch.tensor, action: torch.tensor,
            reward: float, is_complete: bool, final_observation: torch.tensor = None
    ) -> None:
        """
        Appends a single experience to the batch, overwriting a previous experience if needed

        :param observation: The observation of the current experience. Note that this observations
                            must be a tensor with the exact shape, type, and device specified in the
                            constructor.
        :param action:      The action for the current experience. Note that this action must be a
                            tensor with the exact shape, type, and device specified in the constructor
        :param reward:      The discounted cumulative reward received for the current set of experiences.
        :param is_complete: Whether this experience terminated or truncated the episode.
        :param final_observation: If the episode has not yet ended, this parameter should store the final
                            observation. Note that this observations must be a tensor with the exact shape,
                            type, and device specified in the constructor
        """
        if __debug__:
            if final_observation is not None:
                verify_tensor_dims(
                    final_observation, observation.shape, observation.dtype,
                    self._device, 'Final observations'
                )

        if final_observation is not None:
            self.final_observations[self._current_tail_ptr] = final_observation

        super(AggregateExperienceBatch, self).append_single_experience(observation, action, reward, is_complete)

    def append_multiple_experiences(
            self, observation_batch: torch.tensor, action_batch: torch.tensor,
            reward_batch: torch.Tensor, is_complete_batch: torch.Tensor, final_observation_batch: torch.tensor = None
    ) -> None:
        """
        Appends multiple experience to the batch, overwriting previous experiences if needed.
        If the number of provided experiences it greater than n (where n is the length of the batch),
        then only the final n experiences will be stored.

        This method is functionally identical to append_experience_batch, albeit with different parameters.

        :param observation_batch: The observations from each experience. Note that this parameter must be a
                                tensor with the exact type and device specified in the constructor. The
                                outermost dimensions of the tensor should separate different experiences, and
                                the remaining dimensions should match the shape specified in the constructor.

        :param action_batch: The actions from each experience. Note that this parameter must be a tensor with
                                the exact type and device specified in the constructor. The outermost dimensions
                                of the tensor should separate different experiences, and the remaining dimensions
                                should match the shape specified in the constructor.

        :param reward_batch: The discounted cumulative reward received from each set of experiences. Note that this
                                parameter must be a tensor of type torch.float32, and that it must be stored on the
                                device specified in the constructor. The outer dimension of this tensor should
                                separate different experiences, and there should be a second dimension of 1
                                (unflatten the tensor if necessary).

        :param is_complete_batch: Whether this experience terminated or truncated the episode. Note that this
                                parameter must be a tensor of type torch.bool, and that it must be stored on the
                                device specified in the constructor. The outer dimension of this tensor should
                                separate different experiences, and there should be a second dimension of 1
                                (unflatten the tensor if necessary).

        :param final_observation_batch: The final observations from each experience set. Note that this parameter
                                must be a tensor with the exact type and device specified in the constructor.
                                The outermost dimensions of the tensor should separate different experiences, and
                                the remaining dimensions should match the shape specified in the constructor.
        """
        if __debug__:
            if final_observation_batch is not None:
                verify_tensor_inner_dims(
                    final_observation_batch, self.observations.shape[1:], self.observations.dtype,
                    self._device, 'Observation batch'
                )
                assert self.final_observations.shape[0] == self.observations.shape[0]

        if final_observation_batch is not None:
            if final_observation_batch.shape[0] > self.rewards.shape[0]:
                # Truncate the final observation batch
                start = reward_batch.shape[0] - self.rewards.shape[0]
                final_observation_batch = final_observation_batch[start:]

                # Increment tail pointer as if we added those items
                tail = (self._current_tail_ptr + start) % self.rewards.shape[0]
            else:
                tail = self._current_tail_ptr

            if self._current_tail_ptr + reward_batch.shape[0] <= self.rewards.shape[0]:
                end = tail + reward_batch.shape[0]
                # No wrapping required
                self.final_observations[tail:end] = final_observation_batch
            else:
                # need to wrap
                append_end_size = self.rewards.shape[0] - tail
                remaining_item_count = reward_batch.shape[0] - append_end_size

                self.final_observations[tail:] = final_observation_batch[:append_end_size]
                self.final_observations[:remaining_item_count] = final_observation_batch[append_end_size:]

        super(AggregateExperienceBatch, self).append_multiple_experiences(
            observation_batch, action_batch, reward_batch, is_complete_batch
        )

    def append_experience_batch(self, batch: SimpleAggregateExperienceBatch) -> None:
        """
        Appends multiple experience to the batch, overwriting previous experiences if needed.
        If the number of provided experiences it greater than n (where n is the length of the batch),
        then only the final n experiences will be stored.

        This method is functionally identical to append_multiple_experiences, albeit with different parameters.

        :param batch: The batch of experiences to add. Note that the observation and action shapes and types must
                      match what was specified in the constructor. All tensors must also be on the correct device
        """

        if __debug__:
            assert isinstance(batch, SimpleAggregateExperienceBatch), \
                f"batch must be of type SimpleAggregateExperienceBatch, got {type(batch)}"
            verify_tensor_inner_dims(
                batch.observations, self.observations.shape[1:],
                self.observations.dtype, self._device, 'Observations'
            )
            verify_tensor_inner_dims(
                batch.actions, self.actions.shape[1:],
                self.actions.dtype, self._device, 'Actions'
            )
            # Everything else is already verified by the SimpleAggregateExperienceBatch class

        if len(batch) > self.rewards.shape[0]:
            # Truncate the batch if we try to append more items than the max size
            start = len(batch) - self.rewards.shape[0]
            batch = batch[start:]

            # Increment tail pointer as if we added those items
            self._current_tail_ptr = (self._current_tail_ptr + start) % self.rewards.shape[0]

        if self._current_tail_ptr + len(batch) <= self.observations.shape[0]:
            # No wrapping required
            end = self._current_tail_ptr + len(batch)
            self.final_observations[self._current_tail_ptr:end] = batch.final_observations

        else:
            # need to wrap
            append_end_size = self.observations.shape[0] - self._current_tail_ptr
            remaining_item_count = len(batch) - append_end_size
            self.final_observations[self._current_tail_ptr:] = batch.final_observations[:append_end_size]
            self.final_observations[:remaining_item_count] = batch.final_observations[append_end_size:]

        # Call super method
        super(AggregateExperienceBatch, self).append_experience_batch(batch)

    def generate_random_minibatches(self, num_batches: int) -> \
            Generator[tuple[SimpleAggregateExperienceBatch, torch.Tensor], None, None]:
        """
        Divides the buffer into random batches and yields them one at a time.
        Note that every item in this buffer will be yielded exactly once.
        This method will only work if the buffer is full.

        :param num_batches: The number of batches to produce (must be positive and less than the buffer size)
        :return: A generator that yields both the random minibatches and the indices of these experiences
        in the full buffer
        """
        for batch, idxes in super(AggregateExperienceBatch, self).generate_random_minibatches(num_batches):
            yield SimpleAggregateExperienceBatch(
                batch.observations,
                batch.actions,
                batch.rewards,
                batch.is_complete,
                self.final_observations[idxes],
            ), idxes

    # noinspection PyUnresolvedReferences, PyAttributeOutsideInit
    def randomly_select_batch(self, batch_size: int) -> \
            tuple[SimpleAggregateExperienceBatch, torch.Tensor]:

        if self._ExperienceBatch__INDICES_FOR_SAMPLING is None:
            self._ExperienceBatch__INDICES_FOR_SAMPLING = list(range(self.rewards.shape[0]))

        idxes = torch.as_tensor(
            np.asarray(random.sample(self._ExperienceBatch__INDICES_FOR_SAMPLING, batch_size)),
            dtype=torch.int64,
        )

        return SimpleAggregateExperienceBatch(
            self.observations[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.is_complete[idxes],
            self.final_observations[idxes],
        ), idxes


def create_episode_batch(
        env: gym.Env, agent: Agent, discount_factor: float = 0.99, single_action: bool = True,
        observation_dtype: torch.dtype = torch.float32, action_dtype: torch.dtype = torch.int32,
) -> tuple[SimpleExperienceBatch, float]:
    """
    Loop through a complete episode and store experiences with discounted rewards in a batch
    All tensors will be stored on the same device as the agent

    :param env: The gym environment to use to produce episodes (this environment will be reset at the
                start but NOT at the end)
    :param agent: The agent to use to select actions
    :param discount_factor: The discount factor to use when calculating rewards
    :param single_action: if true, the action will be interpreted as a single value, instead of an array
    :param observation_dtype: The datatype of the observations
    :param action_dtype: The datatype of the actions
    :return: A tuple consisting of the produced batch and the total undiscounted reward, for reporting
    """

    if __debug__:
        assert isinstance(env, gym.Env), f"env must be an instance of gym.Env, but received {type(env)}"
        assert isinstance(agent, Agent), f"agent must be an instance of Agent, but received {type(agent)}"
        assert isinstance(discount_factor, float), \
            f"discount_factor must be an instance of float, received {type(discount_factor)}"
        assert 0 < discount_factor <= 1.0, \
            f"discount_factor must be in the range (0, 1], received {discount_factor}"

    current_state, _ = env.reset()

    observations = []
    actions = []
    rewards = []
    dones = []

    while True:
        # Convert the current state into a tensor that can be passed to the agent
        current_state_tensor = torch.as_tensor(current_state, dtype=observation_dtype, device=agent.device)
        current_state_tensor.unsqueeze_(0)

        # Use the agent to select an action and step the environment
        if single_action:
            action = agent.choose_actions(current_state_tensor)[0, 0].item()
        else:
            action = agent.choose_actions(current_state_tensor).detach().cpu().numpy()
        new_state, reward, is_trunc, is_term, _ = env.step(action)

        # Store the data
        observations.append(current_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(is_trunc or is_term)

        # Update the current state
        current_state = new_state

        # Stop once the episode ends
        if is_trunc or is_term: break

    # Calculate the discounted and total reward
    rewards = np.asarray(rewards)
    total_undiscounted_reward = np.sum(rewards)
    rewards = calculate_discounted_rewards(rewards, discount_factor)

    # Convert the reward and is_done array into tensors
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=agent.device)
    done_tensor = torch.as_tensor(np.asarray(dones), dtype=torch.bool, device=agent.device)
    reward_tensor.unsqueeze_(-1)
    done_tensor.unsqueeze_(-1)

    # Return the experience batch and the total undiscounted reward!
    return SimpleExperienceBatch(
        torch.as_tensor(np.asarray(observations), dtype=observation_dtype, device=agent.device),
        torch.as_tensor(np.asarray(actions), dtype=action_dtype, device=agent.device),
        reward_tensor, done_tensor
    ), total_undiscounted_reward


class ExperiencePopulationWrapper:
    """
    Convenience class that automates populating an experience batch with experiences from a gymnasium environment
    """

    def __init__(
            self, batch: ExperienceBatch, env: gym.Env, agent: Agent,
            single_action: bool = True, state_dtype: torch.dtype = torch.float32):

        if __debug__:
            assert isinstance(env, gym.Env), f"env must be an instance of gym.Env, received {type(env)}"
            assert isinstance(agent, Agent), f"agent must be an instance of Agent, received {type(agent)}"
            assert isinstance(single_action, bool), f"single_action must be boolean, received {type(single_action)}"
            assert isinstance(state_dtype, torch.dtype), \
                f"state_dtype must be torch.dtype, received {type(state_dtype)}"
            assert isinstance(batch, ExperienceBatch), \
                f"Batch must be of type ExperienceBatch, received {type(batch)}"
            if isinstance(batch, AggregateExperienceBatch):
                print("WARNING: using experience population wrapper with AggregateExperienceBatch. Consider using "
                      "AggregateExperiencePopulationWrapper instead.")

        self._batch = batch
        self._env = env
        self._agent = agent

        self._single_action = single_action
        self._state_dtype = state_dtype

        self._current_state, _ = env.reset()

    @property
    def batch(self) -> ExperienceBatch:
        """Returns the batch where experiences are being stored"""
        return self._batch

    # noinspection DuplicatedCode
    @torch.no_grad()
    def populate(self, count: int) -> None:
        """
        Adds the specified number of experiences to the batch

        :param count: The number of experiences to add
        """
        if __debug__:
            assert isinstance(count, int), f"count must be an integer, received {type(count)}"

        for _ in range(count):
            # Convert the state into a tensor that can be passed to the agent
            current_state_tensor = torch.as_tensor(
                self._current_state, dtype=self._state_dtype, device=self._agent.device
            )
            current_state_tensor.unsqueeze_(0)

            # Use the agent to select an action and step the environment
            agent_output = self._agent.choose_actions(current_state_tensor)[0]
            if self._single_action:
                action = agent_output.item()
            else:
                action = agent_output.detach().cpu().numpy()
            new_state, reward, is_trunc, is_term, _ = self._env.step(action)

            current_state_tensor.squeeze_(0)


            # Add the data to the batch
            self._batch.append_single_experience(
                current_state_tensor,
                agent_output,
                float(reward),
                is_trunc or is_term,
            )

            # If the episode has ended, restart it
            if is_trunc or is_term:
                self._current_state, _ = self._env.reset()
            else:
                self._current_state = new_state


class AggregateExperiencePopulationWrapper:
    """
    Convenience class that automates populating an aggregate experience batch with aggregate
    experiences from a gymnasium environment
    """

    @dataclass
    class AggregateExperience:
        state: torch.Tensor
        action: torch.Tensor
        reward: float
        done: bool
        final_state: torch.Tensor

    def __init__(
            self, batch: AggregateExperienceBatch, env: gym.Env, agent: Agent,
            aggregate_step_size: int, discount_factor: float,
            single_action: bool = True, state_dtype: torch.dtype = torch.float32,
            rolling_average_reward_window: int = 15, rolling_average_reward_default: int = -10000000
    ):

        if __debug__:
            assert isinstance(env, gym.Env), f"env must be an instance of gym.Env, received {type(env)}"
            assert isinstance(agent, Agent), f"agent must be an instance of Agent, received {type(agent)}"
            assert isinstance(single_action, bool), f"single_action must be boolean, received {type(single_action)}"
            assert isinstance(state_dtype, torch.dtype), \
                f"state_dtype must be torch.dtype, received {type(state_dtype)}"
            assert isinstance(batch, AggregateExperienceBatch), \
                f"Batch must be of type AggregateExperienceBatch, received {type(batch)}"
            assert isinstance(aggregate_step_size, int) and aggregate_step_size > 0, \
                f"Aggregate step size must be a positive integer, received {aggregate_step_size}"
            assert isinstance(discount_factor, float) and 0 <= discount_factor <= 1, \
                f"Discount factor must be a float between 0 and 1, inclusive, received {discount_factor}"

        self._batch = batch
        self._env = env
        self._agent = agent

        self._aggregate_step_size = aggregate_step_size
        self._discount_factor = discount_factor

        self._single_action = single_action
        self._state_dtype = state_dtype

        self._step_gen = self.aggregate_step_generator()

        self._average_episode_reward = RollingAverage(rolling_average_reward_window, rolling_average_reward_default)
        self._num_episodes = 0

    @property
    def batch(self) -> AggregateExperienceBatch:
        """Returns the batch where experiences are being stored"""
        return self._batch

    @property
    def average_reward(self) -> float:
        return self._average_episode_reward.average

    # noinspection DuplicatedCode
    @torch.no_grad()
    def aggregate_step_generator(self) -> Generator[AggregateExperience, None, None]:
        current_state, _ = self._env.reset()
        current_state_tensor = torch.as_tensor(
            current_state, dtype=self._state_dtype, device=self._agent.device
        )
        prior_state_queue: deque[torch.Tensor] = deque(maxlen=self._aggregate_step_size)
        prior_rewards_queue: deque[float] = deque(maxlen=self._aggregate_step_size)
        prior_actions_queue: deque[torch.Tensor] = deque(maxlen=self._aggregate_step_size)

        total_episode_reward = 0

        while True:
            # Use the agent to select an action and step the environment
            current_state_tensor.unsqueeze_(0)
            agent_output = self._agent.choose_actions(current_state_tensor)[0]
            if self._single_action:
                action = agent_output.item()
            else:
                action = agent_output.detach().cpu().numpy()
            new_state, reward, is_trunc, is_term, _ = self._env.step(action)
            current_state_tensor.squeeze_(0)

            new_state_tensor = torch.as_tensor(
                new_state, dtype=self._state_dtype, device=self._agent.device
            )

            # Add data to queues
            prior_state_queue.append(current_state_tensor)
            prior_rewards_queue.append(float(reward))
            prior_actions_queue.append(agent_output)

            total_episode_reward += float(reward)

            if is_trunc or is_term:
                self._num_episodes += 1
                self._average_episode_reward.append(total_episode_reward)
                total_episode_reward = 0

                final_rewards = list(prior_rewards_queue)
                for i in reversed(range(len(prior_rewards_queue) - 1)):
                    final_rewards[i] += final_rewards[i + 1] * self._discount_factor

                for i in range(len(prior_rewards_queue)):
                    yield self.AggregateExperience(
                        prior_state_queue[0],
                        prior_actions_queue[0],
                        final_rewards[i],
                        True,
                        new_state_tensor,
                    )
                    prior_state_queue.popleft()
                    prior_actions_queue.popleft()

                prior_rewards_queue.clear()

                current_state, _ = self._env.reset()
                current_state_tensor = torch.as_tensor(
                    current_state, dtype=self._state_dtype, device=self._agent.device
                )

            else:
                if len(prior_state_queue) == self._aggregate_step_size: # We have an aggregate step!
                    total_reward = 0
                    for reward in reversed(prior_rewards_queue):
                        total_reward = total_reward * self._discount_factor + reward

                    yield self.AggregateExperience(
                        prior_state_queue[0],
                        prior_actions_queue[0],
                        total_reward,
                        False,
                        new_state_tensor,
                    )

                current_state_tensor = new_state_tensor

    @torch.no_grad()
    def populate(self, count: int) -> None:
        """
        Adds the specified number of experiences to the batch

        :param count: The number of experiences to add
        """
        if __debug__:
            assert isinstance(count, int), f"count must be an integer, received {type(count)}"

        for _ in range(count):
            aggregate_step = next(self._step_gen)

            self._batch.append_single_experience(
                aggregate_step.state,
                aggregate_step.action,
                aggregate_step.reward,
                aggregate_step.done,
                aggregate_step.final_state
            )

    @property
    def num_episodes(self) -> int:
        return self._num_episodes


class VectorizedAggregateExperiencePopulationWrapper:
    """
    Convenience class that automates populating an aggregate experience batch with aggregate
    experiences from a vectorized gymnasium environment
    """

    @dataclass
    class AggregateExperience:
        state: torch.Tensor
        action: torch.Tensor
        reward: float
        done: bool
        final_state: torch.Tensor


    @dataclass
    class Step:
        starting_state: torch.Tensor
        action: torch.Tensor
        reward: float
        is_complete: bool


    def __init__(
            self, batch: AggregateExperienceBatch, env: gym.vector.VectorEnv, agent: Agent,
            aggregate_step_size: int, discount_factor: float,
            single_action: bool = True, state_dtype: torch.dtype = torch.float32,
            rolling_average_reward_window: int = 15, rolling_average_reward_default: int = -10000000
    ):

        if __debug__:
            assert isinstance(env, gym.vector.VectorEnv), f"env must be an instance of gym.Env, received {type(env)}"
            assert isinstance(agent, Agent), f"agent must be an instance of Agent, received {type(agent)}"
            assert isinstance(single_action, bool), f"single_action must be boolean, received {type(single_action)}"
            assert isinstance(state_dtype, torch.dtype), \
                f"state_dtype must be torch.dtype, received {type(state_dtype)}"
            assert isinstance(batch, AggregateExperienceBatch), \
                f"Batch must be of type AggregateExperienceBatch, received {type(batch)}"
            assert isinstance(aggregate_step_size, int) and aggregate_step_size > 0, \
                f"Aggregate step size must be a positive integer, received {aggregate_step_size}"
            assert isinstance(discount_factor, float) and 0 <= discount_factor <= 1, \
                f"Discount factor must be a float between 0 and 1, inclusive, received {discount_factor}"

        self._batch = batch
        self._env = env
        self._agent = agent

        self._aggregate_step_size = aggregate_step_size
        self._discount_factor = discount_factor

        self._state_dtype = state_dtype
        self._single_action = single_action

        self._step_gen = self.aggregate_step_generator()

        self._average_episode_reward = RollingAverage(rolling_average_reward_window, rolling_average_reward_default)
        self._num_episodes = 0

    @property
    def batch(self) -> AggregateExperienceBatch:
        """Returns the batch where experiences are being stored"""
        return self._batch

    @property
    def average_reward(self) -> float:
        return self._average_episode_reward.average

    # noinspection DuplicatedCode
    @torch.no_grad()
    def aggregate_step_generator(self) -> Generator[AggregateExperience, None, None]:
        current_state, _ = self._env.reset()
        current_state_tensor = torch.as_tensor(
            current_state, dtype=self._state_dtype, device=self._agent.device
        )

        prior_steps_queue: list[deque[VectorizedAggregateExperiencePopulationWrapper.Step]] = [
            deque(maxlen=self._aggregate_step_size) for _ in range(self._env.num_envs)
        ]

        total_episode_rewards = np.zeros(self._env.num_envs, dtype=np.float32)
        new_episode_started = np.zeros(self._env.num_envs, dtype=np.bool)

        while True:
            # Use the agent to select an action and step all environment
            agent_output = self._agent.choose_actions(current_state_tensor)
            cpu_actions = agent_output.detach().cpu().numpy()
            if self._single_action:
                cpu_actions = cpu_actions.squeeze(-1)
            new_states, rewards, is_term, is_trunc, _ = self._env.step(cpu_actions)

            is_completes = np.logical_or(is_term, is_trunc)

            new_state_tensor = torch.as_tensor(
                new_states, dtype=self._state_dtype, device=self._agent.device
            )

            total_episode_rewards += rewards

            for n in range(self._env.num_envs):
                if new_episode_started[n]:
                    # This environment was just reset, so this current observation is the initial observation
                    # and carries no reward/training merit.
                    new_episode_started[n] = False
                    continue

                prior_steps_queue[n].append(
                    VectorizedAggregateExperiencePopulationWrapper.Step(
                        starting_state=current_state_tensor[n],
                        action=agent_output[n],
                        reward=float(rewards[n]),
                        is_complete=is_completes[n],
                    )
                )

                if is_completes[n]:
                    self._num_episodes += 1
                    self._average_episode_reward.append(total_episode_rewards[n])
                    total_episode_rewards[n] = 0.0

                    total_rewards = [0.0] * (len(prior_steps_queue[n]) + 1)
                    for i, step in enumerate(reversed(prior_steps_queue[n])):
                        total_rewards[i] = total_rewards[i+1] * self._discount_factor + step.reward

                    for i in range(len(total_rewards) - 1):
                        yield VectorizedAggregateExperiencePopulationWrapper.AggregateExperience(
                            state=prior_steps_queue[n][0].starting_state,
                            action=prior_steps_queue[n][0].action,
                            reward=total_rewards[i],
                            done=True,
                            final_state=new_state_tensor[n]
                        )

                        prior_steps_queue[n].popleft()

                    new_episode_started[n] = True

                elif len(prior_steps_queue[n]) == self._aggregate_step_size:
                    total_reward = 0
                    for step in reversed(prior_steps_queue[n]):
                        total_reward = total_reward * self._discount_factor + step.reward

                    yield VectorizedAggregateExperiencePopulationWrapper.AggregateExperience(
                        state=prior_steps_queue[n][0].starting_state,
                        action=prior_steps_queue[n][0].action,
                        reward=total_reward,
                        done=False,
                        final_state=new_state_tensor[n]
                    )

            # Update current state
            current_state_tensor= new_state_tensor

    @torch.no_grad()
    def populate(self, count: int) -> None:
        """
        Adds the specified number of experiences to the batch

        :param count: The number of experiences to add
        """
        if __debug__:
            assert isinstance(count, int), f"count must be an integer, received {type(count)}"

        for _ in range(count):
            aggregate_step = next(self._step_gen)

            self._batch.append_single_experience(
                aggregate_step.state,
                aggregate_step.action,
                aggregate_step.reward,
                aggregate_step.done,
                aggregate_step.final_state
            )

    @property
    def num_episodes(self) -> int:
        return self._num_episodes