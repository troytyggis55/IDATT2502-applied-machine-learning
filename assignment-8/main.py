import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

render_best_model = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AcrobatAgent(nn.Module):
    def __init__(self):
        super(AcrobatAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 3),
        )

    def policy(self, observation):
        logits = self.model(observation)
        action_probs = torch.softmax(logits, dim=0)
        return action_probs

    def act(self, observation):
        action_probs = self.policy(observation)
        action = torch.multinomial(action_probs, 1).item()
        return action, torch.log(action_probs[action])


agent = AcrobatAgent().to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)

env = gym.make("Acrobot-v1")
observation, info = env.reset()
observation = torch.tensor(observation).float().to(device)

episode_over = False
total_reward = 0
time = 0
exploration_sum = 0
platau_count = 0

generation = 0
log_probs = []
rewards = []
total_rewards = []
losses = []

best_model = agent.state_dict()
best_reward = -500

while not episode_over and generation < 1000:

    action, log_prob = agent.act(observation)
    log_probs.append(log_prob)

    observation, reward, terminated, truncated, info = env.step(action)
    observation = torch.tensor(observation).float().to(device)
    rewards.append(reward)
    total_reward += reward
    time += 1

    if terminated or truncated:
        best_model = agent.state_dict() if total_reward > best_reward else best_model
        best_reward = total_reward if total_reward > best_reward else best_reward

        exploration_sum += 10 if total_reward == -500 else -exploration_sum
        penalty = 2 if total_reward == -500 else 1
        treat = 0.5 if total_reward > -100 else 1

        loss = - penalty * treat * (total_reward / 500) * sum(log_probs) - exploration_sum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total_rewards.append(total_reward)

        print(f"Generation {generation} finished with loss {loss.item()}, total reward "
              f"{total_reward}, time {time}, exploration sum {exploration_sum}, platau count {platau_count}")

        platau_count = platau_count + 1 if total_reward == -500 else 0
        if platau_count > 20:
            agent = AcrobatAgent().to(device)
            optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
            platau_count = 0
            exploration_sum = 0

        observation, info = env.reset()
        observation = torch.tensor(observation).float().to(device)
        total_reward = 0
        time = 0
        generation += 1
        log_probs = []
        rewards = []

agent.load_state_dict(best_model)

while render_best_model:
    env = gym.make("Acrobot-v1", render_mode="human")
    observation, info = env.reset()
    observation = torch.tensor(observation).float().to(device)
    total_reward = 0
    episode_over = False

    while not episode_over:
        action, _ = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        observation = torch.tensor(observation).float().to(device)
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Total reward with best model: {total_reward}")

env.close()

# Viktig: for å plotte, kjør i jetbrains pycharm default runner, for å rendre miljø, kjør i terminal
fig = plt.figure()
plt.plot(losses, label="Loss")
plt.plot(total_rewards, label="Total Reward")
plt.axhline(y=-100, color='r', linestyle='--')
plt.show()