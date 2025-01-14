"""
Implements the REINFORCE method on CartPole, LunarLander, and Acrobot

Unlike reinforce_naive.py, this version implements step unrolling and vectorized environments

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
from rlite.experience import create_episode_batch, SimpleExperienceBatch, AggregateExperienceBatch, \
                              VectorizedAggregateExperiencePopulationWrapper


@dataclass
class Parameters:
    ENV_NAME: str
    ENV_OBSERVATION_SHAPE: tuple[int, ...]
    ENV_ACTION_SIZE: int
    TARGET_SCORE: float

    BATCH_SIZE: int
    NUM_ENV: int

    DISCOUNT_FACTOR: float
    LEARNING_RATE: float
    STEP_SIZE: int
    ENTROPY_BETA: float


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
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(parameters.ENV_NAME) for _ in range(parameters.NUM_ENV)]
    )
    print(env.single_observation_space.shape, env.single_action_space.n)
    agent = ReinforceAgent(sum(parameters.ENV_OBSERVATION_SHAPE), parameters.ENV_ACTION_SIZE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=parameters.LEARNING_RATE)
    writer = SummaryWriter(f'./logs/REINFORCE_{parameters.ENV_NAME}-{time.time()}')
    batch_loader = VectorizedAggregateExperiencePopulationWrapper(
        AggregateExperienceBatch(
            parameters.BATCH_SIZE, parameters.ENV_OBSERVATION_SHAPE, (1,), # Discrete space
            action_dtype=torch.long
        ),
        env, agent, parameters.STEP_SIZE, parameters.DISCOUNT_FACTOR,
        rolling_average_reward_window=50, rolling_average_reward_default=-1000
    )

    iteration = 0

    start_time = time.time()

    average_rewards_sum = 0.0
    average_rewards_count = 0

    while True:
        iteration += 1

        # Create batch
        batch_loader.populate(parameters.BATCH_SIZE)
        training_batch = batch_loader.batch

        optimizer.zero_grad()
        model_output = agent(training_batch.observations)
        log_probabilities = nn.functional.log_softmax(model_output, dim=1)
        probabilities = nn.functional.softmax(model_output, dim=1)
        action_log_probabilities = log_probabilities.gather(1, training_batch.actions)

        average_rewards_sum  += training_batch.rewards.sum() / parameters.BATCH_SIZE
        average_rewards_count += 1

        policy_loss = -((training_batch.rewards - average_rewards_sum/average_rewards_count) * action_log_probabilities).mean()
        entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        entropy_loss = -parameters.ENTROPY_BETA * entropy

        loss = policy_loss + entropy_loss

        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.025)
        optimizer.step()

        # Calculate some additional statistics for reporting
        new_probabilities = nn.functional.softmax(agent(training_batch.observations), dim=1)
        kl_divergence = -((new_probabilities / probabilities).log() * probabilities).sum(dim=1).mean()

        max_grad = 0.0
        mean_grad = 0.0
        num_grad = 0
        for parameter in agent.parameters():
            max_grad = max(max_grad, parameter.grad.abs().max().item())
            mean_grad += (parameter.grad.square()).mean().sqrt().item()
            num_grad += 1


        fps = iteration * parameters.BATCH_SIZE / (time.time() - start_time)
        print(f"Iteration {iteration} | {iteration * parameters.BATCH_SIZE}: Loss {loss.item(): .3f} Average {batch_loader.average_reward: .4f} FPS: {fps:.3f}")
        writer.add_scalar('loss', loss.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Average score', batch_loader.average_reward, iteration * parameters.BATCH_SIZE)
        writer.add_scalar('FPS', fps, iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Entropy', entropy.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Gradient/KL Divergence', kl_divergence.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Gradient/Max grad', max_grad, iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Gradient/Mean grad', mean_grad / num_grad, iteration * parameters.BATCH_SIZE)

        if batch_loader.average_reward >= parameters.TARGET_SCORE:
            break

    print("Training complete!")
    writer.close()

CART_POLE_PARAMETERS = Parameters(
    ENV_NAME='CartPole-v1',
    ENV_OBSERVATION_SHAPE=(4,),
    ENV_ACTION_SIZE=2,
    TARGET_SCORE=485,

    BATCH_SIZE = 1000,
    NUM_ENV=1,

    DISCOUNT_FACTOR = 1.0,
    LEARNING_RATE = 0.0025,
    STEP_SIZE = 10,
    ENTROPY_BETA = 0.001
)

LUNAR_LANDER_PARAMETERS = Parameters(
    ENV_NAME='LunarLander-v3',
    ENV_OBSERVATION_SHAPE=(8,),
    ENV_ACTION_SIZE=4,
    TARGET_SCORE=200,

    BATCH_SIZE = 3500,
    NUM_ENV=1,

    DISCOUNT_FACTOR = 0.98,
    LEARNING_RATE = 0.01,
    STEP_SIZE = 25,
    ENTROPY_BETA = 0.01
)

ACROBOT_PARAMETERS = Parameters(
    ENV_NAME='Acrobot-v1',
    ENV_OBSERVATION_SHAPE=(6,),
    ENV_ACTION_SIZE=3,
    TARGET_SCORE=-100,

    BATCH_SIZE = 210,
    NUM_ENV=4,

    DISCOUNT_FACTOR = 1.0,
    LEARNING_RATE = 0.0005,
    STEP_SIZE = 15,
    ENTROPY_BETA = 0.01
)

if __name__ == '__main__':
    main(LUNAR_LANDER_PARAMETERS)

