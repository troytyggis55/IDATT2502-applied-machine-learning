import sys

import pygame
import stable_baselines3

from volleyball_environment import VolleyballEnvironment

if __name__ == "__main__":

    file1 = None
    file2 = None

    if len(sys.argv) > 1:
        file1 = sys.argv[1]
    if len(sys.argv) > 2:
        file2 = sys.argv[2]

    model1 = None
    model2 = None

    if file1 is not None:
        model1 = stable_baselines3.PPO.load(file1, device="cpu")

    if file2 is not None:
        model2 = stable_baselines3.PPO.load(file2, device="cpu")

    env = VolleyballEnvironment(render_mode="human")
    obs, info = env.reset()
    mirrored_obs = env.get_obs_mirror()

    while True:
        if pygame.event.get(pygame.QUIT):
            break
        
        left_action = 0
        right_action = 0

        if model1 is not None:
            left_action = int(model1.predict(obs, deterministic=True)[0])
        else:
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
        else:
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
        
        if terminated or truncated:
            print(f"Winner: {info['winner']}")
            obs, info = env.reset()
            mirrored_obs = env.get_obs_mirror()
