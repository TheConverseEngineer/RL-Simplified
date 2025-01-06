import time

import gymnasium as gym
import numba
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

NUM_ENV = 4
LEARNING_RATE = 0.001
MAX_ITERATIONS = 2000
BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.99
ENTROPY_BETA = 0.01
DEVICE = torch.device('cpu')

class Agent(nn.Module):
    def __init__(self, input_dim: int, action_space: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

    def forward(self, x):
        partial = self.shared(x)
        return self.actor(partial), self.critic(partial)

def telemetry_if_completed(info: dict, writer: SummaryWriter, frame: int):
    if "episode" in info:
        if 'r' in info["episode"]:
            for i, term in enumerate(info["episode"]['_r']):
                if not term: continue
                print(f"frame {frame}, return={info['episode']['r'][i]}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"][i], frame)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"][i], frame)


@numba.jit(nopython=True) # Since we are just looping over a numpy array, numba jit works great here
def calculate_total_rewards(reward: np.ndarray, discount_factor: float) -> np.ndarray:
    """
    Given the raw reward at each step, this method calculates the total discounted reward at each step

    :param reward: A 1-d array representing the reward at each step
    :param discount_factor: The discount factor to use
    :return:       A 1-d array representing the total (present and future) discounted reward obtained at each step

    Credit: https://stackoverflow.com/questions/78923906/efficient-computation-of-sum-of-discounted-rewards-in-rl
    """
    n = reward.shape[0]
    discounted = np.zeros(n + 1) # The final item has a value 0 for padding
    for i in np.arange(n - 1, -1, -1):
        discounted[i] = reward[i] + discount_factor * discounted[i + 1]
    return discounted[:-1] # Ignore the last item


def main():

    env = gym.vector.SyncVectorEnv(lambda: gym.wrappers.RecordEpisodeStatistics(gym.make('CartPole-v1')) for _ in range(NUM_ENV))
    assert len(env.single_observation_space.shape) == 1
    agent = Agent(env.single_observation_space.shape[0], env.single_action_space.n)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(f'logs/ActorCritic2-CartPole-{time.time()}')

    current_state, _ = env.reset()
    current_state = torch.as_tensor(current_state, dtype=torch.float32, device=DEVICE)

    frame = 0

    for iteration in range(MAX_ITERATIONS):
        starting_states = torch.zeros((BATCH_SIZE, NUM_ENV, env.single_observation_space.shape[0]), dtype=torch.float32, device=DEVICE)
        actions = torch.zeros((BATCH_SIZE, NUM_ENV), dtype=torch.int64, device=DEVICE)
        discounted_rewards = [[] for _ in range(NUM_ENV)]
        raw_rewards = [[] for _ in range(NUM_ENV)]

        # Run through parallel environments to get samples
        for batch_frame in range(BATCH_SIZE):
            frame += 1
            with torch.no_grad():
                logits, value = agent(current_state)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()

            new_observations, rewards, is_term, is_trunc, info = env.step(action.cpu().numpy())

            starting_states[batch_frame] = current_state
            actions[batch_frame] = action

            for i, reward in enumerate(rewards):
                raw_rewards[i].append(reward)

            for i in range(NUM_ENV):
                if is_term[i] or is_trunc[i]:
                    # Time to calculate discounted rewards!
                    calculated_discounted_rewards = calculate_total_rewards(np.asarray(raw_rewards[i]), discount_factor=DISCOUNT_FACTOR)
                    discounted_rewards[i].extend(calculated_discounted_rewards)
                    raw_rewards[i].clear()



            telemetry_if_completed(info, writer, frame)

        # Now bootstrap rewards for any partial episodes
        with torch.no_grad():
            _, final_state_values = agent(current_state)
        for i in range(NUM_ENV):
            if not raw_rewards[i]: continue
            calculated_discounted_rewards = calculate_total_rewards(np.asarray(raw_rewards[i]), discount_factor=DISCOUNT_FACTOR)
            calculated_discounted_rewards += (final_state_values[i].item() * (DISCOUNT_FACTOR ** len(raw_rewards[i])))
            discounted_rewards[i].extend(calculated_discounted_rewards)
            raw_rewards[i].clear()

        # Swap shape of this from (num env, batch size) -> (batch size * num env)
        discounted_rewards = np.asarray(discounted_rewards)
        np.swapaxes(discounted_rewards, 0, 1)
        discounted_reward_tensor = torch.as_tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
        discounted_reward_tensor = discounted_reward_tensor.flatten()
        discounted_reward_tensor.unsqueeze_(-1)

        # Now we need to flatten observations from (batch size, num env, observation shape) to (batch size * num env, observation shape)
        starting_states = starting_states.flatten(start_dim=0, end_dim=1)
        actions = actions.flatten()
        actions.unsqueeze_(-1)

        optimizer.zero_grad()

        # Run everything through the model
        actor_logits, critic_values = agent(starting_states)
        log_softmax_prob = nn.functional.log_softmax(actor_logits)
        softmax_prob = nn.functional.softmax(actor_logits)

        chosen_log_softmax = log_softmax_prob.gather(dim=1, index=actions)
        chosen_log_softmax.squeeze_(-1)

        # Time to calculate loss!
        critic_value_loss = nn.functional.mse_loss(discounted_reward_tensor, critic_values)
        actor_policy_loss = -((discounted_reward_tensor - critic_values) * chosen_log_softmax).mean()
        entropy = -(softmax_prob * log_softmax_prob).sum(dim=1).mean()

        loss = actor_policy_loss + critic_value_loss - ENTROPY_BETA * entropy
        loss.backward()
        optimizer.step()

        writer.add_scalar('Value Loss', critic_value_loss.item(), frame)




if __name__ == '__main__':
    main()