from collections import deque
from dataclasses import dataclass
from typing import Any, Generator

import gymnasium as gym

from utils.policy import Policy


@dataclass(frozen=True)
class Step:
    """
    Represents a step in the environment
    \n
    Parameter order: state | action | reward | done_trunc
    """

    state: Any
    """The current state of the environment"""

    action: Any
    """The action taken by the agent"""

    reward: float
    """The reward that the agent received after taking this action"""

    done_trunc: bool
    """True if the episode ended after this action was taken, and false otherwise"""


class ExperienceSource:
    """
    A convenience wrapper class that takes a gymnasium environment (or list of environments)
    and provides an iterator that yields environment steps.
    """

    def __init__(self, envs: gym.Env | list[gym.Env], policy: Policy, steps_per_yield: int = 1):
        """
        Constructor for the ExperienceSource class.
        \n
        This class is a convenience wrapper class that takes a gymnasium environment (or list of environments)
        and provides an iterator that yields environment steps.

        :param envs: The environment (or list of environments) to wrap. If a list is provided, environments MUST
                     be distinct.
        :param policy: The policy to use when selecting actions
        :param steps_per_yield: The maximum number of steps to yield at once (if an episode has ended, less steps
                                may be yielded)
        """
        if isinstance(envs, list):
            self.__envs_list = envs
        else:
            self.__envs_list = [envs]

        self.__agent = policy
        self.__steps_per_yield = steps_per_yield

        self.__total_rewards = [0.0 for _ in range(len(self.__envs_list))]
        self.__last_yielded_total_rewards: float | None = None


    def __iter__(self) -> Generator[tuple[Step, ...], None, None]:
        """
        Returns a generator that yields environment steps. Environment steps are yielded in a tuple of steps,
        whose size is based off of the steps_per_yield parameter in the constructor. Note that if multiple
        environments are provided, the iterator will alternate between then.
        \n
        Updating the policy will update future steps. However, since actions for all environments are generated
        together in one batch, updates may not propagate until all environments yield.
        \n
        Yields may have less items then specified if an episode has ended. Episodes will automatically be restarted
        when they end.
        """
        current_states = [env.reset()[0] for env in self.__envs_list]
        histories = [deque(maxlen=self.__steps_per_yield) for _ in range(len(self.__envs_list))]

        while True:
            # Select all actions in one batch query
            actions = self.__agent.compute(current_states)

            # Alternate between the different environments
            for i, current_env in enumerate(self.__envs_list):
                # Step the environment and add the result to the history for that environment
                new_state, reward, is_done, is_trunc, _ = current_env.step(actions[i])
                histories[i].append(Step(
                    state=current_states[i],
                    action=actions[i],
                    reward=float(reward),
                    done_trunc=is_done or is_trunc
                ))

                # Update total reward and the current state
                self.__total_rewards[i] += reward
                current_states[i] = new_state

                # Restart the environment if the episode is complete
                if is_done or is_trunc:
                    # First yield all partial histories
                    if len(histories[i]) > 0:
                        self.__last_yielded_total_rewards = self.__total_rewards[i]
                        yield tuple(histories[i])
                    self.__last_yielded_total_rewards = None
                    while len(histories[i]) > 2:
                        histories[i].popleft()
                        yield tuple(histories[i])

                    # Now reset
                    histories[i].clear()
                    current_states[i], _ = current_env.reset()
                    self.__total_rewards[i] = 0.0

                # Otherwise yield what we have so far
                elif len(histories[i]) == self.__steps_per_yield:
                    self.__last_yielded_total_rewards = None
                    yield tuple(histories[i])

    def get_total_undiscounted_episode_rewards(self):
        """
        If an episode just ended, this method will return the total undiscounted reward from that episode.
        If the episode is still in progress, this method will return None.
        """
        return self.__last_yielded_total_rewards


@dataclass(frozen=True)
class AggregateStep:
    """
    Represents multiple steps combined into one
    \n
    Parameter order: state | action | reward | final_state
    """

    initial_state: Any
    """ The initial state of the environment """

    action: Any
    """ The action taken by the agent in the initial state"""

    reward: float
    """ 
    The total discounted reward that the agent received after taking this action and 
    any subsequent actions requited to arrive in the final state
    """

    final_state: Any | None
    """ 
    The final state of the environment, or none if the episode ended before 
    the desired number of steps was reached 
    """


class AggregateExperienceSource(ExperienceSource):
    """
    This is a wrapper around ExperienceSource that combines multiple steps for when only the initial and final
    states are needed. For every set of steps, the discounted reward is returned.
    """
    def __init__(self, envs: gym.Env | list[gym.envs], policy: Policy, discount_factor: float, steps_per_yield: int = 2):
        """
        Constructor for the AggregateExperienceSource class.
        \n
        This class is a wrapper around ExperienceSource that combines multiple steps for when only the initial
         and final states are needed. For every set of steps, the discounted reward is returned.

        :param envs: The environment (or list of environments) to wrap. If a list is provided, environments MUST
                     be distinct.
        :param policy: The policy to use when selecting actions
        :param steps_per_yield: The maximum number of steps to combine per yield (must be at least 2)
        :param discount_factor: The discount factor to use when calculating rewards
        """
        assert steps_per_yield > 1
        super(AggregateExperienceSource, self).__init__(envs, policy, steps_per_yield)
        self.__discount_factor = discount_factor
        self.__steps_per_yield = steps_per_yield

    def __iter__(self) -> Generator[AggregateStep, None, None]:
        """
        Returns a generator that yields environment steps. Environment steps are yielded as an AggregateStep object,
        which combines a fixed number of steps from the  steps_per_yield parameter in the constructor. Note that if
        multiple environments are provided, the iterator will alternate between then.
        \n
        Updating the policy will update future steps. However, since actions for all environments are generated
        together in one batch, updates may not propagate until all environments yield.
        \n
        An AggregateStep may represent fewer steps than desired if the episode has ended. In this case, the final state
        will be None. Episodes will automatically be restarted when they end.
        """
        for steps in super(AggregateExperienceSource, self).__iter__():

            # If the episode ended early, set the final state to None
            if steps[-1].done_trunc and len(steps) <= self.__steps_per_yield:
                final_state = None
                steps_for_reward_calc = steps
            else:
                final_state = steps[-1].state
                steps_for_reward_calc = steps[:-1] # Don't count the final reward because it comes afterward

            # Now calculate the total reward (this could be numba JIT compiled,
            # but the steps are usually small enough that it is not worth it)
            reward = 0.0
            for step in reversed(steps_for_reward_calc): reward = step.reward + self.__discount_factor * reward

            yield AggregateStep(
                initial_state=steps[0].state,
                action=steps[0].action,
                reward=reward,
                final_state=final_state
            )