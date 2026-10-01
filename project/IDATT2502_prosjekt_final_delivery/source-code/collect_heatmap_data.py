import os
import sys

import stable_baselines3
from stable_baselines3.common.utils import set_random_seed

from volleyball_environment import VolleyballEnvironment

if __name__ == "__main__":
    file1 = sys.argv[1] if len(sys.argv) > 1 else None
    file2 = sys.argv[2] if len(sys.argv) > 2 else None
    episode_amount = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if not file1:
        print("Please provide at least one model file")
        sys.exit()

    model1 = stable_baselines3.PPO.load(file1, device="cpu")
    model2 = stable_baselines3.PPO.load(file2, device="cpu") if file2 else None

    if not model2:
        print("Warning: No second model file provided")

    heatmap_file = "heatmap_data.txt"
    if model1 is not None:
        if model2 is not None:
            heatmap_file = f"heatmap_data_{os.path.basename(file1).split('.')[0]}_{os.path.basename(file2).split('.')[0]}.txt"
        else:
            heatmap_file = f"heatmap_data_{os.path.basename(file1).split('.')[0]}.txt"
    
    with open(heatmap_file, "w") as f:
        f.write("ball_x,ball_y,player_left_x,player_left_y,player_right_x,player_right_y\n")

    episode = 0
    wins = 0

    set_random_seed(42)

    env = VolleyballEnvironment(render_mode=None)
    obs, info = env.reset(seed=0)
    mirrored_obs = env.get_obs_mirror()

    while episode < episode_amount:
        left_action = int(model1.predict(obs, deterministic=True)[0])
        right_action = 0
        if model2 is not None:
            right_action = int(model2.predict(mirrored_obs, deterministic=True)[0])

        obs, _, terminated, truncated, info = env.step(left_action, right_action)
        mirrored_obs = env.get_obs_mirror()

        with open(heatmap_file, "a") as f:
            f.write(f"{obs[0]},{obs[1]},{obs[4]},{obs[5]},{obs[6]},{obs[7]}\n")

        if terminated or truncated:
            wins += 1 if info.get("winner") == "left" else 0
            print(f"Episode {episode + 1}\tWins: {wins}", end="\r")

            episode += 1
            obs, info = env.reset(seed=episode)
            mirrored_obs = env.get_obs_mirror()

    print(f"Episode {episode}\tWins: {wins}")
