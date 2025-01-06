import numpy as np
import torch
from torch  import nn

import gymnasium as gym

from policy_based.common import PolicyGradientNetwork
from utils.exploration import AggregateExperienceSource
from utils.policy import Policy
from utils.utils import CPU_DEVICE, calculate_total_rewards, RollingAverage

from torch.utils.tensorboard import SummaryWriter

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01
EPISODES_PER_TRAINING_BATCH = 4

class VanillaPolicyGradient(Policy):
    """Simple policy using the vanilla policy gradient"""
    def __init__(self, model: PolicyGradientNetwork, device: torch.device):
        self.__model = model
        self.__device = device

    @torch.no_grad() # TODO: Use this?
    def compute(self, state_batch: list) -> list:
        tensor = torch.as_tensor(np.asarray(state_batch), dtype=torch.float32, device=self.__device)
        output = self.__model(tensor)
        return torch.distributions.categorical.Categorical(logits=output).sample().detach().cpu().numpy()


def main():
    # Set up the environment and device
    env = gym.make('CartPole-v1')
    device = CPU_DEVICE
    print(env.observation_space.shape)
    print(env.action_space.n)

    # Create the model, policy, and experience source
    network = PolicyGradientNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    policy = VanillaPolicyGradient(network, device)
    experience_source = AggregateExperienceSource(env, policy, DISCOUNT_FACTOR, 2)
    experience_source_iterator = iter(experience_source)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    # Set up some logging things
    writer = SummaryWriter(log_dir='./logs')
    average_total_rewards = RollingAverage(window_size=20)

    frame = 0
    episodes = 0
    while True:
        # Iterate through n complete episode
        batch_states = []
        batch_actions = []
        batch_discounted_rewards = []
        for _ in range(EPISODES_PER_TRAINING_BATCH):
            undiscounted_rewards = []

            while True: # Loop through one episode
                frame += 1
                step = next(experience_source_iterator)
                batch_states.append(step.initial_state)
                batch_actions.append(step.action)
                undiscounted_rewards.append(step.reward)
                if step.final_state is None: break

            # Do some logging stuff
            episodes += 1
            total_reward = sum(undiscounted_rewards)
            average_total_rewards.append(total_reward)
            avg = average_total_rewards.average
            writer.add_scalar('Total Reward', total_reward, frame)
            writer.add_scalar('Average Reward', avg, frame)
            print(f'{frame} | Episode: {episodes} Reward: {total_reward:.2f} Average Reward: {avg:.3f}')
            if avg > 450:
                print(f"Training complete in {episodes} episodes!")
                return

            # Calculate the discounted total rewards
            batch_discounted_rewards.extend(calculate_total_rewards(np.array(undiscounted_rewards), DISCOUNT_FACTOR))

        # Zero gradient and create tensors
        optimizer.zero_grad()
        states_tensor = torch.as_tensor(np.asarray(batch_states), dtype=torch.float32, device=device)
        actions_tensor = torch.as_tensor(np.asarray(batch_actions), dtype=torch.int64, device=device)
        discounted_rewards_tensor = torch.as_tensor(np.asarray(batch_discounted_rewards), dtype=torch.float32, device=device)

        # Calculate loss
        log_probabilities = nn.functional.log_softmax(network(states_tensor), dim=1)

        actions_tensor.unsqueeze_(-1)
        log_action_probabilities = log_probabilities.gather(dim=1, index=actions_tensor)
        log_action_probabilities.squeeze_(-1)
        loss = -(discounted_rewards_tensor * log_action_probabilities).mean()

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
