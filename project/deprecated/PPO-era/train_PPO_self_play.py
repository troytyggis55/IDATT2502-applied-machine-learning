import copy
import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
plt.ion()

from environments.pong_environment import PongEnvironment
from ProximalPolicyOptimization import ProximalPolicyOptimization

env = PongEnvironment(max_steps=1000)
model = ProximalPolicyOptimization(input_size=env.input_size, output_size=env.output_size, hidden_size=8, hidden_layers=2)
opponent = ProximalPolicyOptimization(input_size=env.input_size, output_size=env.output_size, hidden_size=8, hidden_layers=2)

random_index = 1
opponent_pool = [[model.state_dict(), 0, 0, 0], [opponent.state_dict(), 0, 0, 0]]
remove_percentage = 0.2
opponent.load_state_dict(model.state_dict())


lr = 0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

batch_size = 5
epochs = 4
mini_batch_size = int(env.max_steps * batch_size * 0.1)

folder = f"models/{env.name}/{model.name}/{datetime.now().strftime('%d%m_%H%M')}"
info_file = f"{folder}/info.txt"

if not os.path.exists(folder):
    os.makedirs(folder)

with open(info_file, "w") as f:
    f.write(f"input_size\toutput_size\thidden_layers\thidden_size\tlr\tactivation\n")
    f.write(f"{env.input_size}\t{env.output_size}\t{model.layers}\t{model.hidden_size}\t{lr}\t{model.activation}\n")
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
    
    observation_all, info_all = env.reset(seed=abs(hash((seed, episode))), options={"render_mode": ("human" if episode == 0 else None)})
    winner = None
    while winner is None:
        steps += 1
        
        observation_model = torch.tensor(observation_all[0]).float()
        observation_opponent = torch.tensor(observation_all[1]).float()
        
        action[0], action_prob = model.act(observation_model)
        action[1], _ = opponent.act(observation_opponent)
        #action[1] = np.int8(0)
        
        value = model.value(observation_model)
        
        observations.append(observation_model)
        actions.append(torch.tensor(action[0]))
        old_action_probs.append(action_prob)
        values.append(value)
        
        observation_model, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(torch.tensor(reward[0]).float())
        
        if terminated[0]:
            round_winner = info.get("winner")
            opponent_pool[random_index][1] += round_winner
            opponent_pool[random_index][2] += 1
        
        if truncated[0]:
            winner = "Truncated"
            #print(f"Episode {episode} done in {steps} steps with reward {sum(rewards)} and winner {winner}")

    rewards = torch.tensor(rewards)
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

    returns = torch.tensor(returns)
    advantages = torch.tensor(advantages)
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
        batch_observations = torch.stack(batch_observations)
        batch_actions = torch.stack(batch_actions)
        batch_old_action_probs = torch.stack(batch_old_action_probs)
        batch_returns = torch.stack(batch_returns)
        batch_advantages = torch.stack(batch_advantages)
        batch_rewards = torch.tensor(batch_rewards)

        # Training with multiple epochs over the batch
        dataset = torch.utils.data.TensorDataset(batch_observations, batch_actions, batch_old_action_probs, batch_returns, batch_advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True) # mini_batch_size

        # Epochs
        for _ in range(epochs):
            for mini_batch in loader:
                mini_batch_observations, mini_batch_actions, mini_batch_old_action_probs, mini_batch_returns, mini_batch_advantages = mini_batch

                # Compute loss and update model
                loss, policy_loss, value_loss = model.compute_loss(
                    mini_batch_observations,
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
        # f"Avg Policy Loss: {avg_losses[1]:.2f}\t"
        # f"Avg Value Loss: {avg_losses[2]:.2f}")

        plt.figure(1)
        plt.plot(episode, batch_reward, 'r.')
        plt.plot(episode, avg_reward, 'b.')
        plt.xlabel('Generation')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Over Generations')
        plt.draw()

        #plt.figure(2)
        #plt.plot(episode, avg_loss, 'b.')
        #plt.xlabel('Generation')
        #plt.ylabel('Average Loss')
        #plt.title('Average Loss Over Generations')
        #plt.draw()

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
                if len(opponent_pool) < 5:
                    opponent_pool.append([copy.deepcopy(opponent.state_dict()), 0, 0, episode])
                    print(f"Added model from generation {episode}")
                
        # Pick new random opponent
        random_index = np.random.randint(0, len(opponent_pool))
        opponent.load_state_dict(opponent_pool[random_index][0])
    
    if episode % 1000 == 0:
        for file in os.listdir(f"{folder}"):
            if file.endswith("replace"):
                os.remove(f"{folder}/{file}")
                
        torch.save(model.state_dict(), f"{folder}/{episode}_replace")
        print(f"Saved model at generation {episode}")

    if episode & (episode - 1) == 0 and 10 <= episode.bit_length() - 1 <= 30:
        torch.save(model.state_dict(), f"{folder}/{episode}")
        print(f"Saved model at generation {episode}")


