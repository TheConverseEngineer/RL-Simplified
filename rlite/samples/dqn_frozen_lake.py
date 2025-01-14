"""
Implements the DQN algorithm on FrozenLake
Unfortunately, the convergence of this algorithm is unreliable, and more hyperparameter tuning is required
"""

import time

import torch
from dataclasses import dataclass

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dqn.dqn_base import EPSILON_INITIAL
from rlite.agent import Agent

import gymnasium as gym

from rlite.experience import AggregateExperienceBatch, AggregateExperiencePopulationWrapper


@dataclass
class Parameters:
    REPLAY_SIZE: int
    BATCH_SIZE: int
    LEARNING_RATE: float

    EPSILON_INITIAL: float
    MIN_EPSILON: float
    EPSILON_DECAY_RATE: float

    DISCOUNT_FACTOR: float
    UNROLL_STEPS: int
    TARGET_SYNC_PERIOD: int

class DQNAgent(nn.Module, Agent):

    def __init__(self, state_size, action_size, epsilon: float):
        super(DQNAgent, self).__init__()

        self.epsilon = epsilon

        self._model = nn.Sequential(
            nn.Linear(state_size, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, action_size),
        )

        self._action_size = action_size
        self._state_size = state_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.one_hot(x, self._state_size)
        return self._model(x.float())

    def choose_actions(self, states: torch.Tensor) -> torch.Tensor:
        model_results = self(states).argmax(dim=-1)

        epsilon_greedy_mask = torch.rand(states.shape[0]) < self.epsilon
        random_actions = torch.randint(self._action_size, (torch.sum(epsilon_greedy_mask).item(),))
        model_results[epsilon_greedy_mask] = random_actions

        return model_results


def main(parameters: Parameters):
    env = gym.make('FrozenLake-v1', map_name='4x4')
    assert env.observation_space.n == 16
    assert env.action_space.n == 4

    agent = DQNAgent(16, 4, EPSILON_INITIAL)
    target_agent = DQNAgent(16, 4, EPSILON_INITIAL)
    target_agent.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=parameters.LEARNING_RATE)

    experience = AggregateExperiencePopulationWrapper(
        AggregateExperienceBatch(
            parameters.REPLAY_SIZE,
            (), (),
            observation_dtype=torch.int64, action_dtype=torch.int64
        ), env, agent, parameters.UNROLL_STEPS, parameters.DISCOUNT_FACTOR,
        True, state_dtype=torch.int64, rolling_average_reward_window=100
    )

    writer = SummaryWriter(f'./logs/DQNFrozenLake-{time.time()}')
    # Start by fully populating the replay buffer
    experience.populate(parameters.REPLAY_SIZE)

    iteration = 0
    while True:
        # Add a new experience to the replay buffer and sample a batch
        experience.populate(1)
        training_batch, _ = experience.batch.randomly_select_batch(parameters.BATCH_SIZE)

        optimizer.zero_grad()

        # Use target agent to calculate final state q-values
        # (note that we set q-values in terminating states to 0)
        with torch.no_grad():
            final_q_values: torch.Tensor = target_agent(training_batch.final_observations).max(1)[0]
            training_batch.is_complete.squeeze_(-1)
            final_q_values[training_batch.is_complete] = 0.0
            final_q_values.detach_()

        # Now get the agent's q-values for the initial states
        initial_estimated_q_values: torch.Tensor = agent(training_batch.observations)
        training_batch.actions.unsqueeze_(-1)
        initial_estimated_q_values = initial_estimated_q_values.gather(1, training_batch.actions)
        initial_estimated_q_values.squeeze_(-1)

        # Now use the future q-values to calculate initial state q-values
        training_batch.rewards.squeeze_(-1)
        initial_theoretical_q_values: torch.Tensor = (training_batch.rewards +
                  final_q_values * parameters.DISCOUNT_FACTOR**parameters.UNROLL_STEPS)

        # Now calculate loss
        loss = nn.functional.mse_loss(initial_estimated_q_values, initial_theoretical_q_values)

        # Backpropagate loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 5)
        optimizer.step()

        # Decay epsilon
        agent.epsilon = max(parameters.MIN_EPSILON, agent.epsilon - parameters.EPSILON_DECAY_RATE)

        # Update target net
        if iteration % parameters.TARGET_SYNC_PERIOD == 0:
            target_agent.load_state_dict(agent.state_dict())
            print(f"{iteration} | {experience.num_episodes}: Epsilon: {agent.epsilon: .3f} Average reward: {experience.average_reward}")
            writer.add_scalar('Epsilon', agent.epsilon, iteration)
            writer.add_scalar('Average reward', experience.average_reward, iteration)
            writer.add_scalar('Loss', loss.item(), iteration)

        iteration += 1


    writer.close()


FROZEN_LAKE_PARAMETERS = Parameters(
    REPLAY_SIZE=5000,
    BATCH_SIZE=32,
    LEARNING_RATE=1e-3,

    EPSILON_INITIAL=1.0,
    MIN_EPSILON=0.04,
    EPSILON_DECAY_RATE=0.00025,

    DISCOUNT_FACTOR=0.99,
    UNROLL_STEPS=3,
    TARGET_SYNC_PERIOD=10,
)

if __name__ == '__main__':
    main(FROZEN_LAKE_PARAMETERS)