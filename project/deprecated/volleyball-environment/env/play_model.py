import os
import numpy as np
import torch
import pygame
from volleyball_environment import VolleyballEnvironment
from default_policy import DefaultPolicyAgent

def reset_environment(env, generation, render_per_x):
    render_mode = "human" if generation % render_per_x == 0 else None
    return env.reset(options={"render_mode": render_mode, "gen": generation})

# Initialize pygame for user input
pygame.init()

# Set up the environment and agents
env = VolleyballEnvironment()
observations, infos = reset_environment(env, generation=0, render_per_x=2000)

# Initializing actions, agent, and logs
actions = {"left": 0, "mirrored_right": 0}
agent_copy = DefaultPolicyAgent()
agent_copy.load_state_dict(torch.load("0311_1206_51000.pth"))

last_x_results = []

# Constants for the environment and training
episodes = 1000
render_per_x = 999
win_percentage_len = episodes
generation = 0
total_reward, total_steps = 0, 0

# Training loop
while generation <= episodes:
    # Handle user input with pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    keys = pygame.key.get_pressed()

    # Map key inputs to actions
    if keys[pygame.K_w]:  # Up
        actions["left"] = 3
    elif keys[pygame.K_s]:  # Down
        actions["left"] = 4
    elif keys[pygame.K_a]:  # Left
        actions["left"] = 1
    elif keys[pygame.K_d]:  # Right
        actions["left"] = 2
    else:
        actions["left"] = 0  # No action

    # Agent copy acts for the right player
    actions["mirrored_right"], _ = agent_copy.act(torch.tensor(observations["mirrored_right"]).float())

    # Step the environment
    observations, rewards, terminated, truncated, infos = env.step(actions)

    total_reward += rewards["left"]
    total_steps += 1

    # Check if the episode has ended
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

        total_reward, total_steps, generation = 0, 0, generation + 1

# Quit pygame when done
pygame.quit()
