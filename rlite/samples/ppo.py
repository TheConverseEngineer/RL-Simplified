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

    EPSILON: float  = 0.2
    NUM_MINIBATCH: int = 5


class ActorCriticAgent(nn.Module, Agent):
    def __init__(self, input_size: int, output_size: int):
        super(ActorCriticAgent, self).__init__()

        self._shared_model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )

        self._actor_model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

        self._critic_model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(1)
        #partial_result = self._shared_model(x)
        return self._actor_model(x), self._critic_model(x)

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


def get_model_output(
        agent: ActorCriticAgent,
        training_batch: AggregateExperienceBatch,
        initial_state_q_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a batch of aggregate experiences and the current agent,
    computes the ratio in probability change for each action, the advantage of
    each chosen action, the agent entropy, the raw probabilities of each action, and critic output for each state
    :param agent:
    :param training_batch:
    :param initial_state_q_values:
    :return: A tuple of tensors containing the ratio in probability change for each action,
    the advantage of each chosen action, the agent entropy, the raw probabilities of
    each action from each visited state (primarily for KL divergence calculation),
    and the value outputs from the critic
    """
    # Get model output
    actor_output, critic_output = agent(training_batch.observations)

    # Apply softmax and select the probabilities corresponding to the given actions
    log_probabilities = nn.functional.log_softmax(actor_output, dim=1)
    action_log_probabilities = log_probabilities.gather(1, training_batch.actions)
    static_action_log_probabilities = action_log_probabilities.detach()

    # Calculate ratio and advantage
    ratio = (action_log_probabilities - static_action_log_probabilities).exp()
    advantage = initial_state_q_values - critic_output.detach()

    # Normalize the advantage for more stable training
    advantage = advantage - advantage.mean() / (advantage.std() + 1e-4) # 1e-4 term helps avoid divide-by-zero errors
    advantage.detach_()

    probabilities = nn.functional.softmax(actor_output, dim=1)

    entropy = -(probabilities * log_probabilities).sum(dim=1).mean()

    return ratio, advantage, entropy, probabilities, critic_output


def main(parameters: Parameters):
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(parameters.ENV_NAME, max_episode_steps=750) for _ in range(parameters.NUM_ENV)]
    )
    print(env.single_observation_space.shape, env.single_action_space.n)
    agent = ActorCriticAgent(sum(parameters.ENV_OBSERVATION_SHAPE), parameters.ENV_ACTION_SIZE)
    old_agent = ActorCriticAgent(sum(parameters.ENV_OBSERVATION_SHAPE), parameters.ENV_ACTION_SIZE)
    old_agent.load_state_dict(agent.state_dict())

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

        for idx, (mini_batch, _) in enumerate(training_batch.generate_random_minibatches(parameters.NUM_MINIBATCH)):
            # Calculate the expected q values for each starting state
            initial_state_q_values = calculate_initial_state_q_values(agent, parameters, mini_batch)

            # Zero gradient and calculate model output
            optimizer.zero_grad()
            ratio, advantage, entropy, probabilities, critic_output = \
                get_model_output(agent, mini_batch, initial_state_q_values)

            # Compute loss
            unclipped_policy_loss = advantage * ratio
            clipped_policy_loss = advantage * torch.clamp(ratio, 1 - parameters.EPSILON, 1 + parameters.EPSILON)
            policy_loss = -torch.min(clipped_policy_loss, unclipped_policy_loss).mean()

            value_loss = nn.functional.mse_loss(critic_output, initial_state_q_values)
            entropy_loss = -parameters.ENTROPY_BETA * entropy

            loss = 0.5 * value_loss + policy_loss + entropy_loss

            # Backpropagate loss
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.1)
            optimizer.step()

            # Calculate some additional statistics for reporting
            new_probabilities = nn.functional.softmax(agent(mini_batch.observations)[0], dim=1)
            kl_divergence = -((new_probabilities / probabilities).log() * probabilities).sum(dim=1).mean()

            max_grad = 0.0
            mean_grad = 0.0
            num_grad = 0
            for parameter in agent.parameters():
                if parameter.grad is not None:
                    max_grad = max(max_grad, parameter.grad.abs().max().item())
                    mean_grad += (parameter.grad.square()).mean().sqrt().item()
                    num_grad += 1

            frame = iteration * parameters.BATCH_SIZE + (parameters.BATCH_SIZE * idx / parameters.NUM_MINIBATCH)
            fps = frame / (time.time() - start_time)
            print(f"Iteration {iteration} | {frame}: Loss {loss.item(): .3f} Average {batch_loader.average_reward: .4f} FPS: {fps:.3f}")
            writer.add_scalar('Loss/loss', loss.item(), frame)
            writer.add_scalar('Average score', batch_loader.average_reward, frame)
            writer.add_scalar('FPS', fps, frame)
            writer.add_scalar('Loss/Entropy', entropy.item(), frame)
            writer.add_scalar('Loss/Value Loss', value_loss.item(), frame)
            writer.add_scalar('Loss/Policy Loss', policy_loss.item(), frame)
            writer.add_scalar('Gradient/KL Divergence', kl_divergence.item(), frame)
            writer.add_scalar('Gradient/Max grad', max_grad, frame)
            if num_grad != 0: writer.add_scalar('Gradient/Mean grad', mean_grad / num_grad, frame)
            writer.add_scalar('Loss/Mean Value', initial_state_q_values.mean().item(), frame)
            writer.add_scalar('Loss/Mean Advantage', advantage.mean().item(), frame)

        if batch_loader.average_reward >= parameters.TARGET_SCORE:
            break

    print("Training complete!")
    writer.close()


def calculate_initial_state_q_values(agent, parameters, training_batch):
    # Calculate values of each state using unrolled bellman equation
    with torch.no_grad():
        final_state_values = agent(training_batch.final_observations)[1]
        final_state_values[training_batch.is_complete] = 0.0
        final_state_values *= parameters.DISCOUNT_FACTOR ** parameters.STEP_SIZE
        initial_state_q_values = training_batch.rewards + final_state_values
        initial_state_q_values.detach_()
    return initial_state_q_values


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
    ENTROPY_BETA = 0.075  # 0.01
)

LUNAR_LANDER_PARAMETERS = Parameters(
    ENV_NAME='LunarLander-v3',
    ENV_OBSERVATION_SHAPE=(8,),
    ENV_ACTION_SIZE=4,
    TARGET_SCORE=200,

    BATCH_SIZE = 3500,
    NUM_ENV=1,

    DISCOUNT_FACTOR = 0.98,
    LEARNING_RATE = 0.0065, # 0.01
    STEP_SIZE = 25,
    ENTROPY_BETA = 0.3
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

