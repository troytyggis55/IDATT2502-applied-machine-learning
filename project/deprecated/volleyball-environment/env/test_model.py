import os

import numpy as np
import torch

from volleyball_environment import VolleyballEnvironment
from default_policy import DefaultPolicyAgent

def reset_environment(env, generation, render_per_x):
    render_mode = "human" if generation % render_per_x == 0 else None
    return env.reset(seed=1, options={"render_mode": render_mode, "gen": generation})

env = VolleyballEnvironment()
observations, infos = reset_environment(env, generation=0, render_per_x=2000)

# Initializing actions, agent, optimizer, and logs
actions = {"left": 0, "mirrored_right": 0}
agent = DefaultPolicyAgent(layers=3)
agent.load_state_dict(torch.load("0311_1824_100000.pth"))
agent_copy = DefaultPolicyAgent(layers=3)
agent_copy.load_state_dict(torch.load("0311_1824_100000.pth"))
 
last_x_results = []

# Constants for the environment and training
episodes = 1000
render_per_x = 999
win_percentage_len = episodes
generation = 0
total_reward, total_steps = 0, 0

# Training loop
while generation <= episodes:
    left_action_mask = env.action_mask("left")
    right_action_mask = env.action_mask("mirrored_right")

    actions = {}
    actions["left"], _ = agent.act(torch.tensor(observations["left"]).float(), action_mask=left_action_mask)

    actions["mirrored_right"], _ = agent_copy.act(torch.tensor(observations["mirrored_right"]).float(), action_mask=right_action_mask)
    
    observations, rewards, terminated, truncated, infos = env.step(actions)
    total_steps += 1

    if terminated["left"] or truncated["left"]:
        winner = infos["left"].get("winner")

        last_x_results.append((winner == "left") * 1.0)
        if len(last_x_results) > win_percentage_len:
            last_x_results.pop(0)

        win_ratio = sum(last_x_results) * 100 / len(last_x_results)

        reward = rewards["left"]

        print(f"Generation {generation}\tfinal {reward:.3f}\ttotal {total_reward:.3f}\tsteps"
              f" {total_steps}"
              f"\t{infos['left'].get('winner')}\t"
              f"{win_ratio:.3f}%")

        observations, infos = reset_environment(env, generation, render_per_x=render_per_x)

        total_reward, total_steps, generation, log_probs = 0, 0, generation + 1, []
