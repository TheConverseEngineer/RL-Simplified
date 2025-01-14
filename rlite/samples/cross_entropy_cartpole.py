"""
Implements the cross entropy method on CartPole.

Performance:
    - On cart pole, this method performs very well and converges in around 60 iterations
    - However, on LunarLander, this method gets stuck at a reward around 20 and cannot fully
      solve the environment
"""


import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rlite.agent import Agent
from rlite.experience import create_episode_batch, SimpleExperienceBatch

BATCH_SIZE = 16
DISCOUNT_FACTOR = 1.0
EPISODE_RETENTION_RATE = 70
LEARNING_RATE = 0.01

class CrossEntropyAgent(nn.Module, Agent):
    def __init__(self, input_size: int, output_size: int):
        super(CrossEntropyAgent, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self._model(x)

    def choose_actions(self, states: torch.Tensor) -> torch.Tensor:
        logits = self(states)
        actions = torch.distributions.categorical.Categorical(logits=logits).sample()
        actions.unsqueeze_(-1)
        return actions


def training_batch_generator(env: gym.Env, agent: CrossEntropyAgent):
    while True:
        complete_batch: list[SimpleExperienceBatch | None] = [None] * BATCH_SIZE
        undiscounted_reward = [None] * BATCH_SIZE
        with torch.no_grad():
            for i in range(BATCH_SIZE):
                complete_batch[i], undiscounted_reward[i] = \
                    create_episode_batch(env, agent, DISCOUNT_FACTOR, action_dtype=torch.long)

        # Compute cutoff
        total_episode_rewards = np.asarray([i.rewards[0,0] for i in complete_batch])
        cutoff = np.percentile(total_episode_rewards, EPISODE_RETENTION_RATE)

        # take the indices of the best episodes and compute total length
        total_training_item_length = 0
        best_episode_indices = []
        for i in range(BATCH_SIZE):
            if complete_batch[i].rewards[0,0] >= cutoff:
                total_training_item_length += len(complete_batch[i])
                best_episode_indices.append(i)

        # And now combine those episodes into a training batch
        yield SimpleExperienceBatch(
            torch.cat([complete_batch[i].observations for i in best_episode_indices]),
            torch.cat([complete_batch[i].actions for i in best_episode_indices]),
            torch.cat([complete_batch[i].rewards for i in best_episode_indices]),
            torch.cat([complete_batch[i].is_complete for i in best_episode_indices]),
        ), cutoff, np.mean(np.asarray(undiscounted_reward))


def main():
    env = gym.make('CartPole-v1')
    print(env.observation_space.shape, env.action_space.n)
    agent = CrossEntropyAgent(4, 2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(f'./logs/CrossEntropyCartPole-{time.time()}')

    iteration = 0

    for training_batch, percentile, average in training_batch_generator(env, agent):
        iteration += 1

        optimizer.zero_grad()
        model_output = agent(training_batch.observations)
        loss = nn.functional.cross_entropy(model_output, training_batch.actions)
        loss.backward()
        optimizer.step()

        print(f"Iteration {iteration}: Loss {loss.item(): .3f} Average {average: .4f}")
        writer.add_scalar('loss', loss.item(), iteration * BATCH_SIZE)
        writer.add_scalar('Average score', average, iteration * BATCH_SIZE)
        writer.add_scalar('Good Episode Cutoff', percentile, iteration * BATCH_SIZE)

        if average > 495:
            break

    print("Training complete!")
    writer.close()


if __name__ == '__main__':
    main()