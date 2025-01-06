from collections import defaultdict
from typing import Any

from q_learning.structs import Parameters

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

class DiscreteQLearningAgent:

    def __init__(self, parameters: Parameters):
        """
        Represents an agent that maintains a q-table of discrete values

        It is assumed that the action type is a discrete integer in the specified range [0, parameters.num_actions)
        The state type can be any hashable type, and is specified by parameters.state_type
        """

        # Stores (state, action) -> expected reward (as a float)
        self.q_table: dict[tuple[parameters.state_type, int], float] = defaultdict(float)

        # Stores the inputted parameters
        self.parameters = parameters

    def get_best_action_and_value(self, state: Any) -> tuple[int, float]:
        """
        Returns what the agent believes to be the optimal action in this state and it's value

        :param state: the current state of the environment (the type of this variable should be specified
                      in parameters.state_type)
        :return: A tuple consisting of [the action (as an integer), its value (as a float)]
        """
        best_action, best_value = None, None

        # Find maximum value over all possible actions
        for action in range(self.parameters.num_actions):
            action_value = self.q_table[state, action]
            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action, best_value

    def update_q_table(self, old_state: Any, action: int, reward: float, new_state: Any) -> None:
        """
        Uses the specified event to update this agent's q-table.
        The learning rate is specified in parameters.learning_rate.
        Both state types should be specified in parameters.state_type.

        :param old_state: the initial state of the environment
        :param action:    the action taken
        :param reward:    the reward received by taking that action
        :param new_state: the new state of the environment
        """
        # First figure out the value of the new state
        _, new_state_value = self.get_best_action_and_value(new_state)

        # Now calculate the expected value of this action in this instance
        action_value = reward + self.parameters.discount_factor * new_state_value

        # Now update the current value in the q-table based on the specified learning rate
        self.q_table[old_state, action] = (
                action_value * self.parameters.learning_rate +
               self.q_table[old_state, action] * (1 - self.parameters.learning_rate)
       )


PARAMS = Parameters(
    num_actions=4,
    state_type=int,
    learning_rate=0.2,
    discount_factor=0.9,
)

NUM_STEPS_PER_EPOCH = 10
NUM_EPISODES_PER_TEST = 20


def play_random_steps(_agent: DiscreteQLearningAgent, _env: gym.Env, num_steps: int):
    state, _ = _env.reset()
    for _ in range(num_steps):
        action = _env.action_space.sample()
        new_state, reward, is_done, is_trunc, _ = _env.step(action)

        _agent.update_q_table(state, action, float(reward), new_state)

        if is_done or is_trunc:
            state, _ = _env.reset()
        else:
            state = new_state


def evaluate_agent(_agent: DiscreteQLearningAgent, _env: gym.Env, num_episodes: int):
    total_reward = 0

    for _ in range(num_episodes):
        state, _ = _env.reset()
        while True:
            action, _ = _agent.get_best_action_and_value(state)
            new_state, reward, is_done, is_trunc, _ = _env.step(action)
            total_reward += reward

            if is_done or is_trunc:
                break
            else:
                state = new_state

    return total_reward/num_episodes


VISUALIZE_AFTER_TRAINING = True

if __name__ == "__main__":

    env = gym.make('FrozenLake-v1')
    agent = DiscreteQLearningAgent(PARAMS)
    writer = SummaryWriter(log_dir='discrete_agent/logs')

    episodes = 0
    best_result = 0.0
    while True:
        episodes += 1

        # Play some rounds where we just pick random actions
        play_random_steps(agent, env, NUM_STEPS_PER_EPOCH)

        # Now play some rounds with the "best" action so we can evaluate performance
        result = evaluate_agent(agent, env, NUM_EPISODES_PER_TEST)
        writer.add_scalar("reward", result, episodes)
        if result > best_result:
            print(f"Training episode {episodes} return {result}")
            best_result = result
        if result > 0.8:
            break

    print("training complete!")
    writer.close()

    if VISUALIZE_AFTER_TRAINING:
        print("Visualizing! (just for fun)")
        env = gym.make('FrozenLake-v1', render_mode='human')

        evaluate_agent(agent, env, 5)




