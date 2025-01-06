"""
Implements the vanilla policy gradient on the Cartpole Environment
Implementation modified from reinforce.py
"""
import time

import numpy as np
import torch
from torch  import nn

import gymnasium as gym

from policy_based.common import PolicyGradientNetwork
from utils.exploration import AggregateExperienceSource
from utils.policy import Policy
from utils.utils import CPU_DEVICE, RollingAverage

from torch.utils.tensorboard import SummaryWriter

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00075
TRAINING_BATCH_SIZE = 128

ENTROPY_BETA = 0.01
REWARD_UNROLL_STEPS = 11

class VanillaPolicyGradient(Policy):
    """Simple policy using the vanilla policy gradient"""
    def __init__(self, model: PolicyGradientNetwork, device: torch.device):
        self.__model = model
        self.__device = device

    @torch.no_grad()
    def compute(self, state_batch: list) -> list:
        tensor = torch.as_tensor(np.asarray(state_batch), dtype=torch.float32, device=self.__device)
        output = self.__model(tensor)
        return torch.distributions.categorical.Categorical(logits=output).sample().detach().cpu().numpy()


def main():
    # Set up the environment and device
    env = [gym.make('CartPole-v1')]
    device = CPU_DEVICE
    print(env[0].observation_space.shape)
    print(env[0].action_space.n)

    # Create the model, policy, and experience source
    network = PolicyGradientNetwork(env[0].observation_space.shape[0], env[0].action_space.n).to(device)
    policy = VanillaPolicyGradient(network, device)
    experience_source = AggregateExperienceSource(env, policy, DISCOUNT_FACTOR, REWARD_UNROLL_STEPS)
    experience_source_iterator = iter(experience_source)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    # Set up some logging things
    writer = SummaryWriter(f'./logs/CartPole-v1-vanilla-{time.time()}')
    average_total_rewards = RollingAverage(window_size=20)

    frame = 0
    episodes = 0
    reward_sum = 0
    while True:
        # Iterate through n complete episode
        batch_states = []
        batch_actions = []
        batch_short_term_discounted_rewards = []
        for _ in range(TRAINING_BATCH_SIZE):
            frame += 1
            step = next(experience_source_iterator)

            # Calculate baseline
            reward_sum += step.reward
            baseline = reward_sum / frame

            batch_states.append(step.initial_state)
            batch_actions.append(int(step.action))
            batch_short_term_discounted_rewards.append(step.reward - baseline)

            # Do some logging stuff if needed
            total_reward = experience_source.get_total_undiscounted_episode_rewards()
            if total_reward is not None:
                episodes += 1
                average_total_rewards.append(total_reward)
                avg = average_total_rewards.average
                writer.add_scalar('Total Reward', total_reward, frame)
                writer.add_scalar('Average Reward', avg, frame)
                writer.add_scalar('Baseline', baseline, frame)
                print(f'{frame} | Episode: {episodes} Reward: {total_reward:.2f} Average Reward: {avg:.3f} Baseline: {baseline:.3f}')
                if avg > 450:
                    print(f"Training complete in {episodes} episodes!")
                    return

        # Zero gradient and create tensors
        optimizer.zero_grad()
        states_tensor = torch.as_tensor(np.asarray(batch_states), dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(np.asarray(batch_actions), dtype=torch.int64, device=device)
        discounted_rewards_tensor = torch.as_tensor(np.asarray(batch_short_term_discounted_rewards), dtype=torch.float32, device=device)

        # Calculate loss
        logits = network(states_tensor)
        log_probabilities = nn.functional.log_softmax(logits, dim=1)
        probabilities = nn.functional.softmax(logits, dim=1)

        actions_tensor.unsqueeze_(-1)
        log_action_probabilities = log_probabilities.gather(dim=1, index=actions_tensor)
        log_action_probabilities.squeeze_(-1)

        policy_loss = -(discounted_rewards_tensor * log_action_probabilities).mean()
        entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        entropy_loss = -ENTROPY_BETA * entropy
        loss = policy_loss + entropy_loss

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
