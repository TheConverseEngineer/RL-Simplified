"""
Implements the REINFORCE method on CartPole, LunarLander, and Acrobot

Performance:
    - On CartPole, this method outperforms cross_entropy slightly, and reliably converges in 40-60 batches
    - On Lunar Lander, this method manages to converge fairly reliably (maybe some more hyperparameter
      tuning would help) in around 125-175 batches
    - On acrobot, the program sometimes gets stuck at -500 and never converges. However, it generally
      converges fairly quickly in around 100-175 batches.
"""


import time

import gymnasium as gym
import numpy as np
import torch
from attr import dataclass
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rlite.agent import Agent
from rlite.experience import create_episode_batch, SimpleExperienceBatch

@dataclass
class Parameters:
    ENV_NAME: str
    ENV_OBSERVATION_SIZE: int
    ENV_ACTION_SIZE: int
    TARGET_SCORE: float

    BATCH_SIZE: int
    DISCOUNT_FACTOR: float
    LEARNING_RATE: float


class ReinforceAgent(nn.Module, Agent):
    def __init__(self, input_size: int, output_size: int):
        super(ReinforceAgent, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self._model(x)

    def choose_actions(self, states: torch.Tensor) -> torch.Tensor:
        logits = self(states)
        actions = torch.distributions.categorical.Categorical(logits=logits).sample()
        actions.unsqueeze_(-1)
        return actions


def training_batch_generator(env: gym.Env, agent: ReinforceAgent, parameters: Parameters):
    while True:
        complete_batch: list[SimpleExperienceBatch | None] = [None] * parameters.BATCH_SIZE
        undiscounted_reward = [None] * parameters.BATCH_SIZE
        with torch.no_grad():
            for i in range(parameters.BATCH_SIZE):
                complete_batch[i], undiscounted_reward[i] = \
                    create_episode_batch(env, agent, parameters.DISCOUNT_FACTOR, action_dtype=torch.long)

        # And now combine those episodes into a training batch
        yield SimpleExperienceBatch(
            torch.cat([i.observations for i in complete_batch]),
            torch.cat([i.actions for i in complete_batch]),
            torch.cat([i.rewards for i in complete_batch]),
            torch.cat([i.is_complete for i in complete_batch]),
        ), np.mean(np.asarray(undiscounted_reward))


def main(parameters: Parameters):
    env = gym.make(parameters.ENV_NAME)
    print(env.observation_space.shape, env.action_space.n)
    agent = ReinforceAgent(parameters.ENV_OBSERVATION_SIZE, parameters.ENV_ACTION_SIZE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=parameters.LEARNING_RATE)
    writer = SummaryWriter(f'./logs/REINFORCE-NAIVE_{parameters.ENV_NAME}-{time.time()}')

    iteration = 0
    frame = 0

    start_time = time.time()
    for training_batch, average in training_batch_generator(env, agent, parameters):
        iteration += 1

        optimizer.zero_grad()
        model_output = agent(training_batch.observations)
        log_probabilities = nn.functional.log_softmax(model_output, dim=1)
        training_batch.actions.unsqueeze_(-1)
        action_log_probabilities = log_probabilities.gather(1, training_batch.actions)
        loss = -(training_batch.rewards * action_log_probabilities).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 5)
        optimizer.step()

        frame += len(training_batch.observations)

        fps = frame / (time.time() - start_time)
        print(f"Iteration {iteration} | {frame}: Loss {loss.item(): .3f} Average {average: .4f} FPS: {fps:.3f}")
        writer.add_scalar('loss', loss.item(), frame)
        writer.add_scalar('Average score', average, frame)
        writer.add_scalar('FPS', fps, frame)

        if average >= parameters.TARGET_SCORE:
            break

    print("Training complete!")
    writer.close()

CART_POLE_PARAMETERS = Parameters(
    ENV_NAME='CartPole-v1',
    ENV_OBSERVATION_SIZE=4,
    ENV_ACTION_SIZE=2,
    TARGET_SCORE=495,

    BATCH_SIZE = 16,
    DISCOUNT_FACTOR = 1.0,
    LEARNING_RATE = 0.01
)

LUNAR_LANDER_PARAMETERS = Parameters(
    ENV_NAME='LunarLander-v3',
    ENV_OBSERVATION_SIZE=8,
    ENV_ACTION_SIZE=4,
    TARGET_SCORE=200,

    BATCH_SIZE = 16,
    DISCOUNT_FACTOR = 0.98,
    LEARNING_RATE = 0.01
)

ACROBOT_PARAMETERS = Parameters(
    ENV_NAME='Acrobot-v1',
    ENV_OBSERVATION_SIZE=6,
    ENV_ACTION_SIZE=3,
    TARGET_SCORE=-100,

    BATCH_SIZE = 16,
    DISCOUNT_FACTOR = 1.0,
    LEARNING_RATE = 0.001
)

if __name__ == '__main__':
    main(LUNAR_LANDER_PARAMETERS)