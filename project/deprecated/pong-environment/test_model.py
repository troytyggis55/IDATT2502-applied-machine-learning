import \
    os

import \
    torch

from default_policy import DefaultPolicyAgent
from env.pong_environment import PongEnvironment

seed = 0

gen = 0
total_reward = 0
total_steps = 0
log_probs = []
last_x_results = []
last_x_total_steps = []

max_episodes = 1000
render_per_x = max_episodes
stats_len = max_episodes

env = PongEnvironment()
observations, infos = env.reset(seed=abs(hash((seed, gen))), options={"render_mode": "human"})

dir1 = "0411_1424"
dir2 = "0411_1424"

gen1 = "65536"
gen2 = "65536"

model1 = f"models/{dir1}/{gen1}"
model2 = f"models/{dir2}/{gen2}"

with open(f"models/{dir1}/info.txt", "r") as f:
    content = f.read().strip()
    layers1, nodes1, lr = content.split("\t")
    
with open(f"models/{dir2}/info.txt", "r") as f:
    content = f.read().strip()
    layers2, nodes2, lr = content.split("\t")

agent = DefaultPolicyAgent(layers=int(layers1), hidden_size=int(nodes1))
agent.load_state_dict(torch.load(model1))
agent_copy = DefaultPolicyAgent(layers=int(layers2), hidden_size=int(nodes2))
agent_copy.load_state_dict(torch.load(model2))

optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-7, weight_decay=1e-5)

actions = {0: 0, 1: 0}

while gen <= max_episodes:
    actions[0], _ = agent.act(torch.tensor(observations[0]).float())
    actions[1], _ = agent_copy.act(torch.tensor(observations[1]).float())
    
    observations, rewards, terminated, truncated, infos = env.step(actions)

    #total_reward += rewards[0]
    total_steps += 1

    if terminated.get(0) or truncated.get(0):
        winner = infos.get(0).get("id")
        
        last_x_results.append((winner == 0) * 1.0)
        last_x_total_steps.append(total_steps)
        
        if len(last_x_results) > stats_len:
            last_x_results.pop(0)
            last_x_total_steps.pop(0)
        
        win_ratio = sum(last_x_results) * 100 / len(last_x_results)    
        
        total_reward = rewards[0]


        print(f"Generation {gen}\t"
                f"Total reward: {total_reward:.2f}\t"
                f"Steps: {total_steps}\t"
                f"Avg steps: {sum(last_x_total_steps) / len(last_x_total_steps):.2f}"
                f"Winner: {winner}\t"
                f"Win ratio: {win_ratio:.2f}\t")

        total_steps, total_reward, gen, log_probs = 0, 0, gen + 1, []
        
        render_mode = "human" if (gen + 10) % render_per_x == 0 else None
        observations, infos = env.reset(seed=abs(hash((seed, gen))), options={"render_mode": render_mode})
