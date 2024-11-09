from datetime import datetime

import \
    pygame

from env.pong_environment import PongEnvironment

env = PongEnvironment()
observations, infos = env.reset(seed=abs(hash(datetime.now())), options={"render_mode": "human"})

actions = {0: 0, 1: 0}
episode_over = False

total_reward = 0
total_steps = 0

while not episode_over:
    keys = pygame.key.get_pressed()
    
    actions[0] = 0
    actions[1] = 0
    
    if keys[pygame.K_w]:
        actions[0] = 2
    elif keys[pygame.K_s]:
        actions[0] = 1
    
    if keys[pygame.K_UP]:
        actions[1] = 2
    elif keys[pygame.K_DOWN]:
        actions[1] = 1

    observations, rewards, terminated, truncated, infos = env.step(actions)

    total_reward += rewards[0]
    total_steps += 1

    if terminated.get(0) or truncated.get(0):
        print(f"Episode finished with total reward {total_reward}, total steps {total_steps}")
        break