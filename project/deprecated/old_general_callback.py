import os
import time
from copy import deepcopy
from datetime import datetime

import matplotlib as mpl
import numpy as np

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class GeneralCallback(BaseCallback):
    def __init__(
            self,
            verbose=1,
            total_trainingsteps=10_000_000,
            plot_every_n_steps=1000,
            steps_per_update=100_000,
            random_opponent_every_n_updates=10,
            modelname=None):
        np.random.seed(42)
        
        super(GeneralCallback, self).__init__(verbose)
        self.total_trainingsteps = total_trainingsteps
        
        self.reward_sum = 0
        self.reward_sums = []
        
        self.rollouts = 0
        
        self.steps_per_update = steps_per_update
        self.random_opponent_every_n_steps = random_opponent_every_n_updates * steps_per_update
        self.plot_every_n_steps = plot_every_n_steps
        
        self.training = False
        self.modelname = modelname
        self.last_n_results = []
        
    
        plt.xlabel('Steps')
        plt.ylabel('Reward sum')
        plt.title('Total rewards during training')
        plt.grid(True)

    def _on_training_start(self) -> None:
        if self.modelname is not None:
            self.folder = f"{self.modelname}"
        else:
            self.folder = f"{datetime.now().strftime('%d%m_%H%M')}"
        
        # Create model_pool folder if it doesn't exist
        if not os.path.exists(f"{self.folder}"):
            os.makedirs(f"{self.folder}")
        if not os.path.exists(f"{self.folder}/model_pool"):
            os.makedirs(f"{self.folder}/model_pool")
        
        self.dictionary_file = f"{self.folder}/dictionary.txt"
        self.model.save(f"{self.folder}/initial.zip")
        
        # Init dictionary file
        with open(self.dictionary_file, "w") as f:
            for key, value in self.model.__dict__.items():
                f.write(f"{key}: {value}\n")
        
        self.rollout_and_training_time = time.time()
        self.init_time = time.time()
        
        self.opponent_pool = [(deepcopy(self.model.policy.state_dict()), 0)]
        self.opponent = self.opponent_pool[0]
        self.training_env.set_attr("opponent", deepcopy(self.model.policy))
        print("Opponent set")
        

    def _on_step(self):
        self.wins += sum(1 for info in self.locals["infos"] if info.get("winner") == "left")
        self.losses += sum(1 for info in self.locals["infos"] if info.get("winner") == "right")
        
        for info in self.locals["infos"]:
            winner = info.get("winner")
            if winner == "left":
                self.last_n_results.append(1)
            elif winner == "right":
                self.last_n_results.append(0)
                
        while len(self.last_n_results) > 1000:
            self.last_n_results.pop(0)
        
        #Plot the rewards
        if self.num_timesteps % self.plot_every_n_steps < self.training_env.num_envs:
            self.reward_sums.append(self.reward_sum)

            if len(self.reward_sums) > 100:
                self.reward_sums.pop(0)

            plt.plot(self.num_timesteps, self.reward_sum, 'r.', label=f'Sum of rewards last {self.plot_every_n_steps} steps', markersize=3)
            plt.plot(self.num_timesteps, np.mean(self.reward_sums), 'b.', label='Average of the last 100 plots', markersize=3)
            plt.show(block=False)
            plt.pause(0.000001)

            self.reward_sum = 0

        # Update the opponent every steps_per_update steps
        if self.num_timesteps % self.steps_per_update < self.training_env.num_envs:
            winrate = np.mean(self.last_n_results)
            
            if winrate > 0.6:
                self.model.save(f"{self.folder}/model_pool/{self.num_timesteps}.zip") # Save the model if the winrate is above 60%
            
            self.opponent_pool.append((deepcopy(self.model.policy.state_dict()), self.rollouts)) # Add the current model to the pool
            
            if self.num_timesteps % self.random_opponent_every_n_steps < self.training_env.num_envs: # Random opponent 20% of the time
                pool_size = len(self.opponent_pool)
                probabilities = np.arange(1, pool_size + 1) / np.arange(1, pool_size + 1).sum()
                chosen_index = np.random.choice(pool_size, p=probabilities)
                self.opponent = self.opponent_pool[chosen_index]
                
                plt.axvline(x=self.num_timesteps, color='y', linestyle='--', linewidth=0.8)
            else: # Choose the latest opponent 80% of the time
                self.opponent = self.opponent_pool[-1]
                plt.axvline(x=self.num_timesteps, color='k', linestyle='--', linewidth=0.8)

            self.training_env.env_method("update_opponent", self.opponent[0])
            print(f"Opponent updated to rollout {self.opponent[1]}")
            
        if self.num_timesteps % (self.total_trainingsteps * 0.1) < self.training_env.num_envs:
            self.model.save(f"{self.folder}/model_pool/{self.num_timesteps}.zip") # Save the model every 10% of the total steps

        if self.training:
            elapsed_training_time = time.strftime("%M:%S", time.gmtime(time.time() - self.start_training_time))
            elapsed_rollout_and_training_time = time.strftime("%M:%S", time.gmtime(time.time() - self.rollout_and_training_time))
            self.rollout_and_training_time = time.time()

            used_time = time.time() - self.init_time
            progress = self.num_timesteps / self.total_trainingsteps * 100
            formatted_time_left = time.strftime("%H:%M:%S", time.gmtime(used_time / progress * (100 - progress)))
            print(f"Progress: {progress:.2f}%\tTime left: {formatted_time_left}\tTraining time: {elapsed_training_time}\tRollout time: {elapsed_rollout_and_training_time}")
            
            self.training = False

        return True

    def _on_rollout_start(self) -> None:
        self.rollouts += 1
        
    def _on_rollout_end(self) -> None:
        self.start_training_time = time.time()
        self.training = True
        
        print(f"Rollout: {self.rollouts}\tSteps: {self.num_timesteps}\tTime trained: {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.init_time))}")

    def _on_training_end(self) -> None:
        self.model.save(f"{self.folder}/final.zip")
        plt.savefig(f"{self.folder}/training_plot.png")
        plt.close()
        print("Training done")
