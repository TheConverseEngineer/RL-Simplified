"""
Implements the REINFORCE method on CartPole, LunarLander, and Acrobot

Unlike reinforce_naive.py, this version implements step unrolling and vectorized environments

Performance:
    - This algorithm works on both CartPole and LunarLander, however some hyperparameter tuning may
      significantly improve performance.
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


class ActorCriticAgent(nn.Module, Agent):
    def __init__(self, input_size: int, output_size: int):
        super(ActorCriticAgent, self).__init__()

        self._shared_model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )

        self._actor_model = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

        self._critic_model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(1)
        partial_result = self._shared_model(x)
        return self._actor_model(partial_result), self._critic_model(partial_result)

    def choose_actions(self, states: torch.Tensor) -> torch.Tensor:
        logits = self(states)[0]
        actions = torch.distributions.categorical.Categorical(logits=logits).sample()
        actions.unsqueeze_(-1)
        return actions


def training_batch_generator(env: gym.Env, agent: ActorCriticAgent, parameters: Parameters):
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
        [lambda: gym.make(parameters.ENV_NAME, max_episode_steps=750) for _ in range(parameters.NUM_ENV)]
    )
    print(env.single_observation_space.shape, env.single_action_space.n)
    agent = ActorCriticAgent(sum(parameters.ENV_OBSERVATION_SHAPE), parameters.ENV_ACTION_SIZE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=parameters.LEARNING_RATE, eps=1e-3)
    writer = SummaryWriter(f'./logs/ACTOR-CRITIC_{parameters.ENV_NAME}-{time.time()}')
    batch_loader = VectorizedAggregateExperiencePopulationWrapper(
        AggregateExperienceBatch(
            parameters.BATCH_SIZE, parameters.ENV_OBSERVATION_SHAPE, (1,), # Discrete space
            action_dtype=torch.long
        ),
        env, agent, parameters.STEP_SIZE, parameters.DISCOUNT_FACTOR,
        rolling_average_reward_window=100, rolling_average_reward_default=-1000
    )

    iteration = 0

    start_time = time.time()

    while True:
        iteration += 1

        # Create batch
        batch_loader.populate(parameters.BATCH_SIZE)
        training_batch = batch_loader.batch

        # Calculate values of each state using unrolled bellman equation
        with torch.no_grad():
            final_state_values = agent(training_batch.final_observations)[1]
            final_state_values[training_batch.is_complete] = 0.0
            final_state_values *= parameters.DISCOUNT_FACTOR ** parameters.STEP_SIZE
            initial_state_q_values = training_batch.rewards + final_state_values
            initial_state_q_values.detach_()

        # Zero gradient and calculate model output
        optimizer.zero_grad()
        actor_output, critic_output = agent(training_batch.observations)
        log_probabilities = nn.functional.log_softmax(actor_output, dim=1)
        probabilities = nn.functional.softmax(actor_output, dim=1)
        action_log_probabilities = log_probabilities.gather(1, training_batch.actions)
        advantage = initial_state_q_values - critic_output.detach()

        advantage.detach_()

        value_loss = nn.functional.mse_loss(critic_output, initial_state_q_values)

        policy_loss = -(advantage * action_log_probabilities).mean()
        entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        entropy_loss = -parameters.ENTROPY_BETA * entropy


        loss = 0.5 * value_loss + policy_loss + entropy_loss
        loss.backward()


        nn.utils.clip_grad_norm_(agent.parameters(), 0.1)
        optimizer.step()

        # Calculate some additional statistics for reporting
        new_probabilities = nn.functional.softmax(agent(training_batch.observations)[0], dim=1)
        kl_divergence = -((new_probabilities / probabilities).log() * probabilities).sum(dim=1).mean()

        max_grad = 0.0
        mean_grad = 0.0
        num_grad = 0
        for parameter in agent.parameters():
            if parameter.grad is not None:
                max_grad = max(max_grad, parameter.grad.abs().max().item())
                mean_grad += (parameter.grad.square()).mean().sqrt().item()
                num_grad += 1


        fps = iteration * parameters.BATCH_SIZE / (time.time() - start_time)
        print(f"Iteration {iteration} | {iteration * parameters.BATCH_SIZE}: Loss {loss.item(): .3f} Average {batch_loader.average_reward: .4f} FPS: {fps:.3f}")
        writer.add_scalar('Loss/loss', loss.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Average score', batch_loader.average_reward, iteration * parameters.BATCH_SIZE)
        writer.add_scalar('FPS', fps, iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Loss/Entropy', entropy.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Loss/Value Loss', value_loss.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Loss/Policy Loss', policy_loss.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Gradient/KL Divergence', kl_divergence.item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Gradient/Max grad', max_grad, iteration * parameters.BATCH_SIZE)
        if num_grad != 0: writer.add_scalar('Gradient/Mean grad', mean_grad / num_grad, iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Loss/Mean Value', initial_state_q_values.mean().item(), iteration * parameters.BATCH_SIZE)
        writer.add_scalar('Loss/Mean Advantage', advantage.mean().item(), iteration * parameters.BATCH_SIZE)

        if batch_loader.average_reward >= parameters.TARGET_SCORE:
            break

    print("Training complete!")
    writer.close()


CART_POLE_PARAMETERS = Parameters(
    ENV_NAME='CartPole-v1',
    ENV_OBSERVATION_SHAPE=(4,),
    ENV_ACTION_SIZE=2,
    TARGET_SCORE=485,

    BATCH_SIZE = 150,
    NUM_ENV=1,

    DISCOUNT_FACTOR = 0.85,
    LEARNING_RATE = 0.001,
    STEP_SIZE = 3,
    ENTROPY_BETA = 0.01
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

