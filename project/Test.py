from datetime import datetime

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback

plt.ion()

from self_play.envs.volleyball_environment import VolleyballEnvironment

env = VolleyballEnvironment()


# Custom Callback for tracking average reward
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


model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)

total_timesteps = np.iinfo(np.int32).max
reward_callback = RewardTrackingCallback(check_freq=100)
model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=reward_callback)
model.save(f"{datetime.now().strftime('%d%m_%H%M')}_{total_timesteps}")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100000000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

