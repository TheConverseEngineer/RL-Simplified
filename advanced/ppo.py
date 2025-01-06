import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

"""Implementation based on the tutorial provided by CleanRL. All credit to them"""


@dataclass
class Args:
    env_id: str = "CartPole-v1"
    total_timesteps: int = 5000000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95

    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coefficient: float = 0.2
    entropy_loss_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def generate_batch(agent, envs: gym.vector.SyncVectorEnv, device: torch.device, num_steps: int, num_envs: int, args, writer: SummaryWriter):
    initial_states = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, device=device)
    log_softmax_probs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    is_dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
    state_estimated_values = torch.zeros((num_steps, num_envs), device=device)

    current_state, _ = envs.reset(seed=1)
    current_state = torch.Tensor(current_state, device=device)
    current_ended_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)

    while True:
        for step in range(0, num_steps):
            # Select and execute action
            with torch.no_grad():
                action, log_softmax_prob, _, value = agent.get_action_and_value(current_state)
            next_state, reward, is_term, is_trunc, infos = envs.step(action.cpu().numpy())

            # Update all values
            initial_states[step] = current_state
            is_dones[step] = current_ended_envs
            actions[step] = action
            log_softmax_probs[step] = log_softmax_prob
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            state_estimated_values[step] = value.flatten()

            current_state = torch.as_tensor(next_state, device=device)
            current_ended_envs = torch.as_tensor(np.logical_or(is_term, is_trunc), device=device)

            if "episode" in infos:
                if 'r' in infos["episode"]:
                    for i, term in enumerate(infos["episode"]['_r']):
                        if not term: continue
                        print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                        writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

        # bootstrap final values for partial episodes
        with torch.no_grad():
            next_value = agent.get_value(current_state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_advantage = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_nonterminals = ~current_ended_envs
                    next_values = next_value
                else:
                    next_nonterminals = ~is_dones[t + 1]
                    next_values = state_estimated_values[t + 1]

                delta = rewards[t] + args.gamma * next_values * next_nonterminals - state_estimated_values[t]
                advantages[t] = last_advantage = delta + args.gamma * args.gae_lambda * next_nonterminals * last_advantage
            returns = advantages + state_estimated_values

        yield current_state, current_ended_envs, initial_states, actions, log_softmax_probs, rewards, is_dones, state_estimated_values, returns, advantages


def generate_random_minibatches(batch_size: int, minibatch_size: int, batch_contents: tuple[torch.Tensor, ...]):
    indices = np.arange(batch_size)
    np.random.shuffle(indices)

    for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        yield tuple([sub_content[start:end] for sub_content in batch_contents])


if __name__ == "__main__":
    args = Args()
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = args.total_timesteps //batch_size
    run_name = f"PPO-{args.env_id}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    device = torch.device("cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id)) for i in range(args.num_envs)],
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_generator = generate_batch(agent, envs, device, args.num_steps, args.num_envs, args, writer)

    global_step = 0
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        percent_remaining = 1.0 - (iteration - 1.0) / num_iterations
        new_learning_rate = percent_remaining * args.learning_rate
        optimizer.param_groups[0]["lr"] = new_learning_rate

        current_state, current_completed_envs, initial_states, actions, log_softmax_probs, rewards, is_dones, estimated_values, returns, advantages = next(batch_generator)
        global_step += args.num_envs * args.num_steps

        # Flatten the tensors in the batch
        initial_states = initial_states.reshape((-1,) + envs.single_observation_space.shape)
        log_softmax_probs = log_softmax_probs.reshape(-1)
        actions = actions.reshape((-1,) + envs.single_action_space.shape)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        estimated_values = estimated_values.reshape(-1)

        # Training time!
        indices = np.arange(batch_size)
        for epoch in range(args.update_epochs):
            # Shuffle up all the items and then train on them in random minibatches
            for (minibatch_advantages, minibatch_initial_states, minibatch_actions, minibatch_log_probs,
                 minibatch_returns, minibatch_estimated_values) \
                in generate_random_minibatches(
                    batch_size, minibatch_size,
                    (advantages, initial_states, actions, log_softmax_probs, returns, estimated_values)
            ):
                _, minibatch_log_softmax_probs, entropy, minibatch_values = agent.get_action_and_value(
                    minibatch_initial_states,
                    minibatch_actions.long()
                )
                ratio = (minibatch_log_softmax_probs - minibatch_log_probs).exp()
                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - args.clip_coefficient, 1 + args.clip_coefficient)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                minibatch_values = minibatch_values.view(-1)
                v_loss_unclipped = (minibatch_values - minibatch_returns) ** 2
                v_clipped = minibatch_estimated_values + torch.clamp(minibatch_values - minibatch_estimated_values, -args.clip_coefficient, args.clip_coefficient)
                v_loss_clipped = (v_clipped - minibatch_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy and total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.entropy_loss_coefficient * entropy_loss + v_loss * args.value_loss_coefficient

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("loss", loss.item(), global_step)
        print("FPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("fps", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()