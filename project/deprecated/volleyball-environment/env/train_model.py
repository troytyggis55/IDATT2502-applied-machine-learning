import os

import numpy as np
import torch

from volleyball_environment import VolleyballEnvironment
from default_policy import DefaultPolicyAgent

# Constants for the environment and training
episodes = 100000
render_per_x = np.inf
replace_per_x = 1
save_per_x = 1000
stats_len = replace_per_x
generation = 0

log_probs = []
last_x_results = []
last_x_total_steps = []
total_steps = 0
total_reward = 0

def reset_environment(env, generation, render_per_x):
    render_mode = "human" if (generation + 10) % render_per_x == 0 else None
    return env.reset(options={"render_mode": render_mode, "gen": generation})

env = VolleyballEnvironment()
observations, infos = reset_environment(env, generation=0, render_per_x=1)

# Initializing actions, agent, optimizer, and logs
agent = DefaultPolicyAgent(layers=3)
# agent.load_state_dict(torch.load("wdlc_4000.pth"))
agent_copy = DefaultPolicyAgent(layers=3)
agent_copy.load_state_dict(agent.state_dict())

optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-7,
                             weight_decay=1e-5)


# Make the current date and time into a stringname that can be used to save the model
# Format MM_DD_HH_MM
import datetime

now = datetime.datetime.now()
now = now.strftime("%d%m_%H%M")
print(now)

# Training loop
while generation <= episodes:
    # Get action masks for both agents
    left_action_mask = env.action_mask("left")
    right_action_mask = env.action_mask("mirrored_right")

    # Select actions while respecting action masks
    actions = {}
    actions["left"], log_prob = agent.act(torch.tensor(observations["left"]).float(), action_mask=left_action_mask)
    log_probs.append(log_prob)

    actions["mirrored_right"], _ = agent_copy.act(
        torch.tensor(observations["mirrored_right"]).float(), action_mask=right_action_mask)

    observations, rewards, terminated, truncated, infos = env.step(actions)
    total_steps += 1
    total_reward += rewards["left"]

    if terminated["left"] or truncated["left"]:
        winner = infos["left"].get("winner")

        last_x_results.append((winner == "left") * 1.0)
        last_x_total_steps.append(total_steps)
        if len(last_x_results) > stats_len:
            last_x_results.pop(0)
            last_x_total_steps.pop(0)

        win_ratio = sum(last_x_results) * 100 / len(last_x_results)
        avg_steps = sum(last_x_total_steps) / len(last_x_total_steps)

        loss = agent.loss(log_probs, total_reward, total_steps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Generation {generation}\t"
              f"reward {total_reward:.3f}\t"
              f"steps {total_steps}\t"
              f"avg steps {avg_steps:.3f}\t"
              f"{winner}\t"
              f"{win_ratio:.3f}%\t"
              f"loss {loss:.3f}")

        observations, infos = reset_environment(env, generation, render_per_x=render_per_x)

        if generation % replace_per_x == 0:
            agent_copy.load_state_dict(agent.state_dict())
            #print("""
            #    ###################################
            #    Replaced Model
            #    ###################################
            #    """)
            last_x_results = []

        if generation % save_per_x == 0:
            print("""
                ###################################
                Saving Model
                ###################################
                """)
            # Delete files that start with 'wdlc'
            for file in os.listdir():
                if file.startswith(now):
                    os.remove(file)

            title = f"{now}_{generation}.pth"
            torch.save(agent.state_dict(), title)

        if generation & (generation - 1) == 0 and 10 <= generation.bit_length() - 1 <= 30:
            print("""
                ###################################
                Saving Model
                ###################################
                """)

            title = f"safe_{now}_{generation}.pth"
            torch.save(agent.state_dict(), title)

        total_steps, total_reward, generation, log_probs = 0, 0, generation + 1, []
