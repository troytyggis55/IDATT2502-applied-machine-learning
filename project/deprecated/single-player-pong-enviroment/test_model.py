from datetime import datetime

import \
    torch

from default_policy import DefaultPolicyAgent
import \
    pygame

from env.single_player_pong_environment_v1 import SinglePlayerPongEnvironment

now = datetime.now()

env = SinglePlayerPongEnvironment()
observations, infos = env.reset(seed=abs(hash((now,0))), options={"render_mode": "human"})

dir1 = "0511_1321"
gen1 = "200000_replace"

model1 = f"models/{dir1}/{gen1}"

with open(f"models/{dir1}/info.txt", "r") as f:
    content = f.read().strip()
    layers1, nodes1, lr = content.split("\t")

agent = DefaultPolicyAgent(layers=int(layers1), hidden_size=int(nodes1))
agent.load_state_dict(torch.load(model1))

actions = {0: 0}

total_reward = 0
total_steps = 0
gen = 0

episodes = 500
last_x_results = []
stats_len = 100

while gen <= episodes:
    actions[0], _ = agent.act(torch.tensor(observations[0]).float())

    observations, rewards, terminated, truncated, infos = env.step(actions)

    total_reward += \
        rewards[
            0]
    total_steps += 1

    if terminated.get(0) or truncated.get(0):
        last_x_results.append(total_reward)

        if len(last_x_results) > stats_len:
            last_x_results.pop(0)

        avg_reward = sum([
            x
            for
            x
            in
            last_x_results]) / len(last_x_results)

        print(f"Generation {gen}\t"
              f"Total reward: {total_reward:.2f}\t"
              f"Avg reward: {avg_reward:.2f}\t"
              f"Steps: {total_steps}\t")

        total_steps, total_reward, gen, log_probs = 0, 0, gen + 1, []

        observations, infos = env.reset(seed=abs(hash((now, gen))),
                                        options={"render_mode": "human" if (gen == episodes - 1) else None})
