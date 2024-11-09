from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

from self_play.envs.volleyball_environment import VolleyballEnvironment

env = VolleyballEnvironment()
obs, info = env.reset()

model = DQN("MlpPolicy", env, verbose=1)


class RewardTrackingCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.avg_rewards = []
        self.episode_rewards = []

    def _on_step(self):
        # Accumulate rewards from each step
        reward = self.locals["rewards"]
        self.episode_rewards.append(reward)

        # Calculate average reward periodically
        if self.n_calls % self.check_freq == 0:
            avg_reward = np.mean(self.episode_rewards)
            self.avg_rewards.append(avg_reward)
            self.episode_rewards = []  # Reset for the next interval

            if len(self.avg_rewards) > 20:
                self.avg_rewards.pop(0)

            plt.figure(1)
            plt.plot(self.n_calls, avg_reward, 'r.')
            plt.plot(self.n_calls, np.mean(self.avg_rewards), 'b.')
            plt.xlabel('Steps')
            plt.ylabel('Average Reward')
            plt.title('Average Reward Over Generations')
            plt.draw()

            if self.verbose > 0:
                print(f"Step {self.n_calls}: average reward = {avg_reward}")

        plt.pause(0.0001)

        return True

reward_callback = RewardTrackingCallback(check_freq=100)

now = datetime.now()

while True:
    model.learn(total_timesteps=1, reset_num_timesteps=False, progress_bar=True, callback=reward_callback)
    model.save(f"{now.strftime('%d%m_%H%M')}_{model.num_timesteps}")
    
    env.render_mode = "human"
    
    obs, info = env.reset()
    
    env.render()
    for i in range(1000):
        action = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
            
        if terminated or truncated:
            obs, info = env.reset()
    
env.close()