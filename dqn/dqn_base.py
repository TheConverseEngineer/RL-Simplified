import time

import ale_py
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

from utils.exploration import AggregateStep, AggregateExperienceSource
from utils.policy import Policy
from utils.utils import MPS_DEVICE, SizedBuffer, CPU_DEVICE, RollingAverage
from utils.wrappers import make_atari_env

import gymnasium as gym

""" Environment Constants """
ENV_NAME = 'PongNoFrameskip-v4'
TARGET_REWARD = 18.0
TARGET_DEVICE = MPS_DEVICE

""" DQN Constants"""
REPLAY_SIZE = 10_000
BATCH_SIZE = 32
TARGET_NETWORK_SYNC_PERIOD = 1_000

""" DQN Extension Constants """
N_STEPS_UNROLL = 3 # Track how many steps we should unroll when calculating rewards in the Bellman Equation (>= 1)

""" Learning Rate, Discount Factor, and Epsilon Constants"""
LEARNING_RATE = 7.82e-5
DISCOUNT_FACTOR = 0.98
EPSILON_INITIAL = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY_STEPS = 100_000
EPSILON_DECAY_RATE = (EPSILON_INITIAL - EPSILON_FINAL) / EPSILON_DECAY_STEPS

"""
Type assumptions:
- State: a numpy array of bytes
- Action: an integer
"""


def batch_to_tensor(batch: list[AggregateStep], device: torch.device) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts the inputted list of Aggregate Steps into tensors
    :param batch: A list of Aggregate Steps
    :param device: The device where tensors should be stored (gpu or cpu)
    :return: A tuple of tensors in the following format: initial_state (uint8) | actions (int64) | rewards (float32) |
             complete (bool) | final_state (uint8)
    """
    initial_states, final_states = [None]*len(batch), [None]*len(batch)
    actions, rewards = np.zeros(len(batch), dtype=np.int64), np.zeros(len(batch), dtype=np.float32)
    complete = np.zeros(len(batch), dtype=np.bool)

    for i, step in enumerate(batch):
        initial_states[i] = step.initial_state
        final_states[i] = step.final_state if step.final_state is not None else step.initial_state
        actions[i] = step.action
        rewards[i] = step.reward
        complete[i] = (step.final_state is None)

    return (
        torch.as_tensor(np.asarray(initial_states), dtype=torch.uint8, device=device),
        torch.as_tensor(actions, dtype=torch.int64, device=device),
        torch.as_tensor(rewards, dtype=torch.float32, device=device),
        torch.as_tensor(complete, dtype=torch.bool, device=device),
        torch.as_tensor(np.asarray(final_states), dtype=torch.uint8, device=device),
    )


def calculate_dqn_loss(batch: list[AggregateStep], network: nn.Module, target_network: nn.Module,
                       accumulated_discount_factor: float, device: torch.device) -> torch.Tensor:
    """
    Calculates the DQN network loss in this batch of steps

    :param batch: A list of Aggregate Steps
    :param network: The network whose loss should be calculated
    :param target_network: The target network that is used to calculate the "actual" next q-value
    :param accumulated_discount_factor: The discount factor accumulated over the number of unrolled steps
    :param device: The device where the models are stored (gpu or cpu)
    """
    # Convert batch to tensors
    initial_states, actions, rewards, complete, final_states = batch_to_tensor(batch, device)
    actions.unsqueeze_(-1)

    # Find out what value the network gave each chosen action
    predicted_action_values = network(initial_states).gather(1, actions)
    predicted_action_values.squeeze_(-1)

    # Compute the 'actual' value of the next state using the target network
    with torch.no_grad():
        next_state_values = target_network(final_states).max(1)[0]
        next_state_values[complete] = 0.0
        next_state_values.detach_()

    # Use the next state value to compute the "actual" value of each chosen action
    actual_action_values = next_state_values * accumulated_discount_factor + rewards

    return nn.functional.mse_loss(predicted_action_values, actual_action_values)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.ByteTensor):
        # scale on GPU
        xx = x / 255.0
        return self.fc(self.conv(xx))


class DQNPolicy(Policy):

    def __init__(self, model: DQN, device: torch.device, epsilon: float, num_actions: int):
        self.__model = model
        self.__device = device
        self.epsilon = epsilon
        self.num_actions = num_actions

    @torch.no_grad()
    def compute(self, state_batch: list) -> np.ndarray:
        state_tensor = torch.as_tensor(np.asarray(state_batch), dtype=torch.float32, device=self.__device)
        model_results = self.__model(state_tensor).argmax(dim=1)
        model_results = model_results.detach().cpu().numpy()

        mask = np.random.random(size=len(state_batch)) < self.epsilon
        random_actions = np.random.choice(self.num_actions, sum(mask))
        model_results[mask] = random_actions

        return model_results



def main():
    # Create the environment and specify which device to use
    gym.register_envs(ale_py)
    env = make_atari_env(ENV_NAME)
    print(f"Environment observation space shape: {env.observation_space.shape}")
    print(f"Environment action space dimension: {env.action_space.n}")

    # Create the network, target network, and optimizer
    net = DQN(env.observation_space.shape, env.action_space.n).to(TARGET_DEVICE)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(TARGET_DEVICE)
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    accumulated_discount_factor = DISCOUNT_FACTOR ** N_STEPS_UNROLL

    # Create the policy, experience source, and buffer
    policy = DQNPolicy(net, TARGET_DEVICE, EPSILON_INITIAL, num_actions=env.action_space.n)
    experience_source = AggregateExperienceSource(env, policy, DISCOUNT_FACTOR, N_STEPS_UNROLL+1)
    experience_source_iterator = iter(experience_source)
    buffer = SizedBuffer(REPLAY_SIZE)

    # Populate the experience buffer
    print(time.time())
    for _ in range(REPLAY_SIZE):
        buffer.append(next(experience_source_iterator))
    print(time.time())
    print("buffer populated!")

    # Create the summary writer and some performance metrics
    writer = SummaryWriter(log_dir='./logs')
    best_reward = -40
    average_reward = RollingAverage(window_size=100)

    # Time to train!
    frame = 0
    episodes_completed = 0
    start_time = time.time()
    while True:
        # Step the simulator and choose a random batch to train on
        buffer.append(next(experience_source_iterator))
        batch = buffer.get_random_content(BATCH_SIZE)

        # Training time
        optimizer.zero_grad()
        loss = calculate_dqn_loss(batch, net, target_net, accumulated_discount_factor, TARGET_DEVICE)
        loss.backward()
        optimizer.step()

        # Modify epsilon
        if frame < EPSILON_DECAY_STEPS: policy.epsilon -= EPSILON_DECAY_RATE

        # Sync target network
        if frame % TARGET_NETWORK_SYNC_PERIOD == 0:
            target_net.load_state_dict(net.state_dict())

        # Print out some metrics
        reward = experience_source.get_total_undiscounted_episode_rewards()
        if reward is not None:
            episodes_completed += 1
            fps = frame / (time.time() - start_time)
            average_reward.append(reward)
            avg = average_reward.average
            print(f"{frame} | Reward: {reward} Average: {avg:.2f} Epsilon: {policy.epsilon:.3f} Episodes {episodes_completed} FPS: {fps:.3f}")
            writer.add_scalar('Reward', reward, frame)
            writer.add_scalar('Average Reward', avg, frame)
            writer.add_scalar('FPS', episodes_completed, frame)
            writer.add_scalar('Episodes Completed', episodes_completed, frame)
            writer.add_scalar('Epsilon', policy.epsilon, frame)

            if round(avg, 1) > best_reward:
                best_reward = round(avg, 1)
                torch.save(net.state_dict(), f'./models/dqn_model_{best_reward}.pt')
                print(f"New best reward! {best_reward}")

        frame += 1


if __name__ == '__main__':
    main()








