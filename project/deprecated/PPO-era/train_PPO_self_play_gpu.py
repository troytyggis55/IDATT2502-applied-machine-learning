import copy
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam

from environments.pong_environment_parallell import PongEnvironment
from ProximalPolicyOptimizationVectorized import ProximalPolicyOptimization

plt.ion()

# GPU-optimized Pong Environment using JAX
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = PongEnvironment(num_envs=1024)
model = ProximalPolicyOptimization(input_size=env.input_size, output_size=env.output_size,
                                   hidden_size=16, hidden_layers=2).to(device)
opponent = ProximalPolicyOptimization(input_size=env.input_size, output_size=env.output_size,
                                      hidden_size=16, hidden_layers=2).to(device)

random_index = 1
opponent_pool = [[copy.deepcopy(model), 0, 0, 0], [copy.deepcopy(opponent), 0, 0, 0]]
remove_percentage = 0.2

lr = 0.0003
optimizer = Adam(model.parameters(), lr=lr)

batch_size = 10
epochs = 2
mini_batch_size_percentage = 0.1

folder = f"models/{env.name}/{model.name}/{datetime.now().strftime('%d%m_%H%M')}"
info_file = f"{folder}/info.txt"

if not os.path.exists(folder):
    os.makedirs(folder)

with open(info_file, "w") as f:
    f.write(f"input_size\toutput_size\thidden_layers\thidden_size\tlr\tactivation\n")
    f.write(
        f"{env.input_size}\t{env.output_size}\t{model.layers}\t{model.hidden_size}\t{lr}\t{model.activation}\n")
    f.write(f"episode\tavg_reward\tbatch_reward\tbatch_loss\n")

seed = 0
infinity = np.iinfo(np.int32).max
action = {0: 0, 1: 0}

stat_rewards = []
stat_losses = []

batch_observations = []
batch_actions = []
batch_old_action_probs = []
batch_returns = []
batch_advantages = []
batch_rewards = []

for episode in range(0, infinity):
    steps = 0
    observations = []
    actions = []
    old_action_probs = []
    rewards = []
    values = []

    # Reset environment and seed
    random_key = (seed + episode) % (2 ** 32 - 1)
    observation_all, info_all = env.reset(seed=int(random_key))
    winner = None
    while winner is None:
        steps += 1

        # Extract observations for each agent
        obs_agent_0 = np.array(observation_all[0])
        obs_agent_1 = np.array(observation_all[1])

        # Get actions and action probabilities for each agent
        action_0, action_prob_0 = model.act(torch.tensor(obs_agent_0).float().to(device))
        action_1, action_prob_1 = opponent.act(torch.tensor(obs_agent_1).float().to(device))

        # Get value estimation for agent 0
        value = model.value(torch.tensor(obs_agent_0).float().to(device))

        # Append data for training
        observations.append(obs_agent_0)
        actions.append(action_0)
        old_action_probs.append(action_prob_0)
        values.append(value)

        # Prepare the action dictionary for the environment
        action = {0: action_0.cpu().numpy(), 1: action_1.cpu().numpy()}

        # Step the environment
        observation_all, reward, terminated, truncated, info = env.step(action)

        # Record rewards
        rewards.append(torch.tensor(reward).float().to(device))

        if terminated[0] or truncated[0]:
            winner = info.get("winner") if info.get("winner") is not None else 0
            opponent_pool[random_index][1] += winner
            opponent_pool[random_index][2] += 1

    # Convert to tensors
    rewards = torch.tensor(rewards).to(device)
    stat_rewards.append(rewards.sum().item())
    if len(stat_rewards) > 100:
        stat_rewards.pop(0)

    values = torch.stack(values).detach()
    returns = []
    advantages = []
    discounted_sum = 0
    for i in reversed(range(len(rewards))):
        discounted_sum = rewards[i] + 0.99 * discounted_sum
        returns.insert(0, discounted_sum)
        advantages.insert(0, discounted_sum - values[i].item())

    returns = torch.tensor(returns).to(device)
    advantages = torch.tensor(advantages).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_observations.extend(observations)
    batch_actions.extend(actions)
    batch_old_action_probs.extend(old_action_probs)
    batch_returns.extend(returns)
    batch_advantages.extend(advantages)
    batch_rewards.append(rewards.sum().item())

    episode += 1

    # Check if batch is ready for training
    if episode % batch_size == 0:
        # Convert batch data to tensors
        batch_observations = jnp.stack(batch_observations)
        batch_actions = torch.stack(batch_actions).to(device)
        batch_old_action_probs = torch.stack(batch_old_action_probs).to(device)
        batch_returns = torch.stack(batch_returns).to(device)
        batch_advantages = torch.stack(batch_advantages).to(device)
        batch_rewards = torch.tensor(batch_rewards).to(device)

        # Training with multiple epochs over the batch
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(np.array(batch_observations)), batch_actions, batch_old_action_probs,
            batch_returns, batch_advantages
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(
            len(batch_observations) * mini_batch_size_percentage), shuffle=True)  # mini_batch_size

        # Epochs
        for _ in range(epochs):
            for mini_batch in loader:
                mini_batch_observations, mini_batch_actions, mini_batch_old_action_probs, mini_batch_returns, mini_batch_advantages = mini_batch

                # Compute loss and update model
                loss, policy_loss, value_loss = model.compute_loss(
                    mini_batch_observations.to(device),
                    mini_batch_actions,
                    mini_batch_old_action_probs,
                    mini_batch_returns,
                    mini_batch_advantages
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stat_losses.append(loss.item())

        batch_loss = np.mean(stat_losses)
        batch_reward = batch_rewards.mean().item()
        avg_reward = np.mean(stat_rewards)

        print(f"Episode {episode}\t"
              f"Avg Reward: {avg_reward:.3f}\t"
              f"Batch reward: {batch_reward:.3f}\t"
              f"Batch loss: {batch_loss:.3f}",
              f"Opponent gen: {opponent_pool[random_index][3]}\t"
              f"Opponent games: {opponent_pool[random_index][2]}\t"
              f"Opponent winrate: {opponent_pool[random_index][1] / opponent_pool[random_index][2]:.3f}")

        plt.figure(1)
        plt.plot(episode, batch_reward, 'r.')
        plt.plot(episode, avg_reward, 'b.')
        plt.xlabel('Generation')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Over Generations')
        plt.draw()
        plt.pause(0.0001)
        stat_losses = []

        if episode % 100 == 0:
            with open(info_file, "a") as f:
                f.write(f"{episode}\t{avg_reward}\t{batch_reward}\t{batch_loss}\n")

        # Clear batch data
        batch_observations = []
        batch_actions = []
        batch_old_action_probs = []
        batch_returns = []
        batch_advantages = []
        batch_rewards = []

        # Remove worst models
        if opponent_pool[random_index][2] >= 20:
            winrate = opponent_pool[random_index][1] / opponent_pool[random_index][2]
            if winrate < remove_percentage:
                opponent_pool.pop(random_index)
                print(f"Removed model with winrate {winrate}")
                if len(opponent_pool) < 10:
                    opponent_pool.append((copy.deepcopy(model), 0, 0, episode))
                    print(f"Added model from generation {episode}")

        # Pick new random opponent
        random_index = np.random.randint(0, len(opponent_pool))
        opponent = opponent_pool[random_index][0]

    if episode % 1000 == 0:
        for file in os.listdir(f"{folder}"):
            if file.endswith("replace"):
                os.remove(f"{folder}/{file}")

        torch.save(model.state_dict(), f"{folder}/{episode}_replace")
        print(f"Saved model at generation {episode}")

    if episode & (episode - 1) == 0 and 10 <= episode.bit_length() - 1 <= 30:
        torch.save(model.state_dict(), f"{folder}/{episode}")
        print(f"Saved model at generation {episode}")
