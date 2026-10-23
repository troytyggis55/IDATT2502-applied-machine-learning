import os
import sys

import pygame
import stable_baselines3
from stable_baselines3.common.utils import set_random_seed

from volleyball_environment_debug import VolleyballEnvironment


if __name__ == "__main__":

    file1 = None
    file2 = None

    mode = []
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].split(",")
    if len(sys.argv) > 2:
        file1 = sys.argv[2]
    if len(sys.argv) > 3:
        file2 = sys.argv[3]
    
    model1 = None
    model2 = None

    if file1 is not None:
        model1 = stable_baselines3.PPO.load(file1, device="cpu")

    if file2 is not None:
        model2 = stable_baselines3.PPO.load(file2, device="cpu")


    if "trace" in mode:
        env = VolleyballEnvironment(render_mode="trace")
    elif "render" in mode:
        env = VolleyballEnvironment(render_mode="human")
    else:
        env = VolleyballEnvironment(render_mode=None)
        
    if os.path.exists("heatmap_data") is False:
        os.mkdir("heatmap_data")
        
    heatmap_file = "heatmap_data/heatmap_data.txt"
    if "heatmap" in mode:
        if model1 is not None and model2 is not None:
            heatmap_file = f"heatmap_data/heatmap_data_{os.path.basename(file1).split('.')[0]}_{os.path.basename(file2).split('.')[0]}.txt"
        with open(heatmap_file, "w") as f:
            f.write("ball_x,ball_y,player_left_x,player_left_y,player_right_x,player_right_y\n")
    
    
    episode = 0
    wins = 0
    end_episode = 1
    
    obs, info = env.reset(seed=0)
    mirrored_obs = env.get_obs_mirror()

    if "seed" in mode:
        set_random_seed(42)
    
    for types in mode:
        if types.startswith("ep"):
            end_episode = int(types[2:])
    
    while episode < end_episode:
        if "step" in mode:
            stepped = False
            while not stepped:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    stepped = True
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
        
        left_action = env.action_space.sample()
        right_action = None

        if model1 is not None:
            left_action = int(model1.predict(obs, deterministic=True)[0])
        elif "render" in mode:
            left_action = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                left_action = 3
            elif keys[pygame.K_s]:
                left_action = 4
            elif keys[pygame.K_a]:
                left_action = 1
            elif keys[pygame.K_d]:
                left_action = 2

        if model2 is not None:
            right_action = int(model2.predict(mirrored_obs, deterministic=True)[0])
        elif "render" in mode:
            right_action = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                right_action = 3
            elif keys[pygame.K_DOWN]:
                right_action = 4
            elif keys[pygame.K_LEFT]:
                right_action = 2
            elif keys[pygame.K_RIGHT]:
                right_action = 1

        obs, _, terminated, truncated, info = env.step(left_action, right_action)
        mirrored_obs = env.get_obs_mirror()
        
        if "heatmap" in mode:
            with open(heatmap_file, "a") as f:
                f.write(f"{obs[0]},{obs[1]},{obs[4]},{obs[5]},{obs[6]},{obs[7]}\n")
        
        if terminated or truncated:
            wins += 1 if info.get("winner") == "left" else 0
            print(f"Episode {episode + 1}\tWins: {wins}", end="\r")
            
            episode += 1
            obs, info = env.reset(seed=episode)
            mirrored_obs = env.get_obs_mirror()
            
    print(f"Episode {episode}\tWins: {wins}")
