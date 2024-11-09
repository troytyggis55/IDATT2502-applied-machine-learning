import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Define a simple Q-Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
        
    def forward(self, state):
        return self.model(state)

env = gym.make("Acrobot-v1")
agent = DQN()
optimizer = optim.Adam(agent.parameters(), lr=0.001)

epsilon = 1.0
gamma = 0.99
num_episodes = 5
losses = []
total_rewards = []

# Training loop
for episode in range(num_episodes):
    observations, info = env.reset()
    observations = torch.tensor(observations, dtype=torch.float32)
    total_reward = 0
    loss = 0
    episode_over = False

    while not episode_over:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = agent(observations)
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            episode_over = True
            break
            
        next_state = torch.tensor(next_state, dtype=torch.float32)

        total_reward += reward

        with torch.no_grad():
            target = reward if terminated else reward + gamma * agent(next_state).max().item()
            
        predicted = agent(observations)[action]
        loss = (predicted - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        observations = next_state

    total_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(0.01, epsilon * 0.995)

    print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss.item()}, Epsilon: {epsilon}")

env.close()

# Plot losses and rewards
fig = plt.figure()
plt.plot(losses, label="Loss")
plt.plot(total_rewards, label="Total Reward")
plt.show()
