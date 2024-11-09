from datetime import datetime
import os

import numpy as np
import torch

from env.pong_environment import PongEnvironment
from default_policy import DefaultPolicyAgent

seed = 0

gen = 0
total_reward = 0
total_steps = 0
log_probs = []
last_x_results = []
last_x_total_steps = []

max_episodes = np.inf
render_per_x = np.inf
replace_per_x = 20000
replace_if = 75
save_per_x = 2000
stats_len = 500

now = datetime.now()
now = now.strftime("%d%m_%H%M")
print(now)

layers = 0
nodes = 16
lr = 0.0003

# Create folder for saving models with the current date and time
if not os.path.exists("models"):
    os.makedirs("models")
    
if not os.path.exists(f"models/{now}"):
    os.makedirs(f"models/{now}")
    #make a file and write the layers and nodes to it
    with open(f"models/{now}/info.txt", "w") as f:
        f.write(f"{layers}\t{nodes}\t{lr}")

env = PongEnvironment()
observations, infos = env.reset(seed=abs(hash((seed, gen))), options={"render_mode": "human"})

agent = DefaultPolicyAgent(layers=layers, hidden_size=nodes)
agent_copy = DefaultPolicyAgent(layers=layers, hidden_size=nodes)
agent_copy.load_state_dict(agent.state_dict())

optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

actions = {0: 0, 1: 0}

while gen <= max_episodes:
    actions[0], log_prob = agent.act(torch.tensor(observations[0]).float())
    actions[1], _ = agent_copy.act(torch.tensor(observations[1]).float())
    
    log_probs.append(log_prob)

    observations, rewards, terminated, truncated, infos = env.step(actions)

    #total_reward += rewards[0]
    total_steps += 1

    if terminated.get(0) or truncated.get(0):
        winner = infos.get(0).get("id")
        #total_reward = (2 * win_ratio - 1) * len(last_x_results) / stats_len #rewards[0]
        total_reward = rewards[0]
        #print(rewards)
        
        last_x_results.append([(winner == 0) * 1.0, total_reward, total_steps])
        
        if len(last_x_results) > stats_len:
            last_x_results.pop(0)
        
        win_ratio = sum([x[0] for x in last_x_results]) * 100 / len(last_x_results)
        avg_reward = sum([x[1] for x in last_x_results]) / len(last_x_results)
        avg_steps = sum([x[2] for x in last_x_results]) / len(last_x_results)
        
        if gen % stats_len == 0:
            with open(f"models/{now}/stats.txt", "a") as f:
                f.write(f"{gen}\t{win_ratio:.2f}\t{avg_reward:.2f}\t{avg_steps:.2f}\n")
        
        if gen % replace_per_x == 0 or (len(last_x_results) >= stats_len and win_ratio >= replace_if):
            agent_copy.load_state_dict(agent.state_dict())
            print(f"Replaced model at generation {gen} with win ratio {win_ratio:.2f}")
            last_x_results = []
            
        if gen % save_per_x == 0:
            #Delete last saved model
            for file in os.listdir(f"models/{now}"):
                if file.endswith("replace"):
                    os.remove(f"models/{now}/{file}")
                
            torch.save(agent.state_dict(), f"models/{now}/{gen}_replace")
            print(f"Saved model at generation {gen}")
            
        if gen & (gen - 1) == 0 and 10 <= gen.bit_length() - 1 <= 30:
            torch.save(agent.state_dict(), f"models/{now}/{gen}")
            print(f"Saved model at generation {gen}")

        loss = agent.loss(log_probs, avg_reward, total_steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Generation {gen}\t"
                f"Total reward: {total_reward:.2f}\t"
                f"Avg reward: {avg_reward:.2f}\t"
                f"Steps: {total_steps}\t"
                f"Avg steps: {avg_steps:.2f}\t"
                f"Winner: {winner}\t"
                f"Win ratio: {win_ratio:.2f}\t"
                f"Loss: {loss.item():.2f}")


        total_steps, total_reward, gen, log_probs = 0, 0, gen + 1, []
        
        render_mode = "human" if (gen + 10) % render_per_x == 0 else None
        observations, infos = env.reset(seed=abs(hash((seed, gen))), options={"render_mode": render_mode})
