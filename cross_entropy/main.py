import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cross_entropy.structs import Parameters
from cross_entropy.policy import CrossEntropyPolicyNetwork
from cross_entropy.training import generate_training_batch, filter_training_batch, visualize_run

import gymnasium as gym

import sys

"""
NOTES:
 - Works better with short episodes that give continuous rewards
 - Heavily reliant on actually finding a "good" episode
 - Struggles with complex tasks where a good action requires multiple steps and is hard to accidentally find
"""

PARAMS = Parameters(
    observation_space_size=4,
    action_space_size=2,
    hidden_layer_size=128,
    batch_size=16,
    episode_retention_percentile=70,
    reward_discount_factor=1.0,
    learning_rate=0.01,
)

def train(policy_to_train: CrossEntropyPolicyNetwork,):
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=policy_to_train.parameters(), lr=PARAMS.learning_rate)
    writer = SummaryWriter(comment="-cartpole")


    for iter_no, batch in enumerate(generate_training_batch(env, policy_to_train, PARAMS)):
        (
            train_observation_tensor,
            train_action_tensor,
            reward_mean,
            reward_std,
            reward_percentile_cutoff
        ) = filter_training_batch(batch, PARAMS)

        optimizer.zero_grad()
        action_scores = policy_to_train(train_observation_tensor)
        loss = objective(action_scores, train_action_tensor)
        loss.backward()
        optimizer.step()

        print(f"Iteration {iter_no}:"
              f"\tLoss: {loss.item()}"
              f"\tReward Mean: {reward_mean}"
              )
        writer.add_scalar("loss", loss.item(), iter_no)
        writer.add_scalar("reward_cutoff", reward_percentile_cutoff, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_std", reward_std, iter_no)

        if reward_mean > 475:
            print("Solved!")
            break

    writer.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        env = gym.make("CartPole-v1")
    else:
        env = gym.make("CartPole-v1", render_mode='human')

    assert env.observation_space.shape == (4,)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 2

    policy = CrossEntropyPolicyNetwork(PARAMS)

    if len(sys.argv) == 1:
        # Train the network
        train(policy)
        torch.save(policy.state_dict(), "models/policy.pth")

        print("Saved training file to models/policy.pth")
        print("Visualizing")

        env.close()
        env = gym.make("CartPole-v1", render_mode='human')

        visualize_run(env, policy)

        env.close()


    else:
        policy.load_state_dict(torch.load(sys.argv[1], weights_only=True))

        for i in range(5):
            visualize_run(env, policy)

    env.close()




