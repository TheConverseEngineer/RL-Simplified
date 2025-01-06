"""
Implements the Advantage Actor-Critic algorithm on the Cartpole Environment.
Implementation modified from vanilla_pg.py
"""
import time

import numpy as np
import torch
from torch  import nn

import gymnasium as gym

from utils.exploration import AggregateExperienceSource
from utils.policy import Policy
from utils.utils import CPU_DEVICE, RollingAverage

from torch.utils.tensorboard import SummaryWriter

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00035
TRAINING_BATCH_SIZE = 128 * 4

ENTROPY_BETA = 0.01
REWARD_UNROLL_STEPS = 11

NUM_ENV = 4


class ActorCriticNetwork(nn.Module):
    """
    Represents the deep learning network used by our vanilla policy gradient
    """

    def __init__(self, input_size: int, num_actions: int):
        """
        Constructor for the PolicyGradientNetwork class
        This class represents the deep learning network used by our vanilla policy gradient
        \n
        Note that this network does NOT apply the softmax activation function

        :param input_size:  The size of the input
        :param num_actions:   The number of possible actions
        """
        super(ActorCriticNetwork, self).__init__()

        # Thankfully, cartpole requires a fairly simple model
        self.shared_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.value_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        partial =  self.shared_net(x)
        return self.policy_net(partial), self.value_net(partial)



class ActorCriticPolicy(Policy):
    """Simple policy using the vanilla policy gradient"""
    def __init__(self, model: ActorCriticNetwork, device: torch.device):
        self.__model = model
        self.__device = device

    @torch.no_grad()
    def compute(self, state_batch: list) -> list:
        tensor = torch.as_tensor(np.asarray(state_batch), dtype=torch.float32, device=self.__device)
        output, _ = self.__model(tensor)
        return torch.distributions.categorical.Categorical(logits=output).sample().detach().cpu().numpy()


def main():
    # Set up the environment and device
    envs = [gym.make('LunarLander-v3') for _ in range(NUM_ENV)]
    device = CPU_DEVICE
    print(envs[0].observation_space.shape)
    print(envs[0].action_space.n)

    # Create the model, policy, and experience source
    network = ActorCriticNetwork(envs[0].observation_space.shape[0], envs[0].action_space.n).to(device)
    policy = ActorCriticPolicy(network, device)
    experience_source = AggregateExperienceSource(envs, policy, DISCOUNT_FACTOR, REWARD_UNROLL_STEPS)
    experience_source_iterator = iter(experience_source)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=1e-4)

    # Set up some logging things
    writer = SummaryWriter(f'./logs/CartPole-v1-actor_critic-{time.time()}')
    average_total_rewards = RollingAverage(window_size=20)

    frame = 0
    episodes = 0
    while True:
        # Iterate through n complete episode
        batch_states = []
        batch_final_states = []
        batch_final_state_indices = []
        batch_actions = []
        batch_short_term_discounted_rewards = []
        for i in range(TRAINING_BATCH_SIZE):
            frame += 1
            step = next(experience_source_iterator)

            batch_states.append(step.initial_state)
            batch_actions.append(int(step.action))
            batch_short_term_discounted_rewards.append(step.reward)

            if step.final_state is not None:
                batch_final_states.append(step.final_state)
                batch_final_state_indices.append(i)

            # Do some logging stuff if needed
            total_reward = experience_source.get_total_undiscounted_episode_rewards()
            if total_reward is not None:
                episodes += 1
                average_total_rewards.append(total_reward)
                avg = average_total_rewards.average
                writer.add_scalar('Total Reward', total_reward, frame)
                writer.add_scalar('Average Reward', avg, frame)
                if avg > 450:
                    print(f"Training complete in {episodes} episodes!")
                    return


        # Zero gradient and create tensors
        optimizer.zero_grad()
        states_tensor = torch.as_tensor(np.asarray(batch_states), dtype=torch.float32, device=device)
        final_states_tensor = torch.as_tensor(np.asarray(batch_final_states), dtype=torch.float32, device=device)
        final_state_indices_tensor = torch.as_tensor(np.asarray(batch_final_state_indices), dtype=torch.int64, device=device)
        actions_tensor = torch.as_tensor(np.asarray(batch_actions), dtype=torch.int64, device=device)
        discounted_rewards_tensor = torch.as_tensor(np.asarray(batch_short_term_discounted_rewards), dtype=torch.float32, device=device)

        # Calculate loss
        logits, values = network(states_tensor)
        log_probabilities = nn.functional.log_softmax(logits, dim=1)
        probabilities = nn.functional.softmax(logits, dim=1)

        # Get the log probabilities of just the chosen actions
        actions_tensor.unsqueeze_(-1)
        log_action_probabilities = log_probabilities.gather(dim=1, index=actions_tensor)
        log_action_probabilities.squeeze_(-1)

        # Calculate the future reward of each action/state
        future_rewards = torch.zeros_like(discounted_rewards_tensor)
        if batch_final_states:
            _, future_state_value = network(final_states_tensor)
            future_state_value.squeeze_(-1)
            future_rewards[final_state_indices_tensor] = future_state_value
        discounted_rewards_tensor = discounted_rewards_tensor + (DISCOUNT_FACTOR ** REWARD_UNROLL_STEPS) * future_rewards

        policy_loss = -((discounted_rewards_tensor - values) * log_action_probabilities).mean()
        discounted_rewards_tensor.unsqueeze_(-1)
        value_loss = 0.5 * nn.functional.huber_loss(values, discounted_rewards_tensor)
        entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
        entropy_loss = -ENTROPY_BETA * entropy
        loss = policy_loss + entropy_loss + value_loss

        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/Policy Loss', policy_loss.item(), frame)
        writer.add_scalar('Loss/Value Loss', value_loss.item(), frame)
        writer.add_scalar('Loss/Entropy', entropy.item(), frame)
        torch.distributions.Categorical(logits=logits).entropy().mean()
        writer.add_scalar('Entropy', torch.distributions.Categorical(logits=logits).entropy().mean().item(), frame)
        print(f'{frame} | Episode: {episodes}  Average Reward: {average_total_rewards.average:.3f}')


if __name__ == '__main__':
    main()
