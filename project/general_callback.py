import os
import time
from copy import deepcopy
from datetime import datetime

import matplotlib as mpl
import numpy as np
from stable_baselines3 import PPO

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
        
        self.rollouts = 0
        
        self.steps_per_update = steps_per_update
        self.random_opponent_every_n_steps = random_opponent_every_n_updates * steps_per_update
        self.plot_every_n_steps = plot_every_n_steps
        
        self.modelname = modelname
        self.last_n_results = []
        self.wins = 0
        self.losses = 0
        
        #plt.xlabel('Steps')
        #plt.ylabel('Reward sum')
        #plt.title('Total rewards during training')
        #plt.grid(True)
        
    def _get_state_dict_from_file(self, file):
        return PPO.load(file, device='cpu').policy.state_dict()

    def _on_training_start(self) -> None:
        if self.modelname is not None:
            self.folder = f"{self.modelname}"
        else:
            self.folder = f"{datetime.now().strftime('%d%m_%H%M')}"
        
        # Create model_pool folder if it doesn't exist
        if not os.path.exists(f"{self.folder}"):
            os.makedirs(f"{self.folder}")
        if not os.path.exists(f"{self.folder}/milestone_plots"):
            os.makedirs(f"{self.folder}/milestone_plots")
        if not os.path.exists(f"{self.folder}/model_pool_plots"):
            os.makedirs(f"{self.folder}/model_pool_plots")
        
        self.dictionary_file = f"{self.folder}/dictionary.txt"
        
        # Init dictionary file
        with open(self.dictionary_file, "w") as f:
            for key, value in self.model.__dict__.items():
                f.write(f"{key}: {value}\n")
                
        self.stats_file = f"{self.folder}/stats.txt"
        with open(self.stats_file, "w") as f:
            f.write("num_timesteps,wins,losses,winrate,avg_winrate,opponent\n")
        
        self.init_time = time.time()
        
        model_name = f"{self.folder}/model_pool/0.zip"
        self.model.save(model_name)
        self.training_env.set_attr("opponent", deepcopy(self.model.policy))
        print("Opponent set")
        self.opponent_name = "0"

    def _on_step(self):
        for info in self.locals["infos"]:
            winner = info.get("winner")
            if winner == "left":
                self.last_n_results.append(1)
                self.wins += 1
            elif winner == "right":
                self.losses += 1
                self.last_n_results.append(0)
                
        while len(self.last_n_results) > 200:
            self.last_n_results.pop(0)

        # Update the opponent every steps_per_update steps
        if self.num_timesteps % self.steps_per_update < self.training_env.num_envs:
            winrate = np.mean(self.last_n_results)
            print(f"Winrate: {winrate}")
            
            if self.num_timesteps % self.random_opponent_every_n_steps < self.training_env.num_envs: # Random opponent 20% of the time                
                files = [f"{self.folder}/model_pool/{file}" for file in os.listdir(f"{self.folder}/model_pool") if file.endswith(".zip")]
                files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
                #
                #indices = np.arange(len(files))
                #decay_rate = 0.1
                #probabilities = np.exp(decay_rate * indices)
                #probabilities = probabilities / probabilities.sum()
                #
                #chosen_index = np.random.choice(indices, p=probabilities)
                chosen_index = np.random.randint(0, len(files))
                self.opponent = self._get_state_dict_from_file(files[chosen_index])
                print(f"Chose random opponent: {files[chosen_index]}")
                self.opponent_name = files[chosen_index].split("/")[-1].split(".")[0]
                
                #plt.axvline(x=self.num_timesteps, color='y', linestyle='--', linewidth=0.6)
            else: # Choose the latest opponent 80% of the time
                #if winrate > 0.75:
                model_name = f"{self.folder}/model_pool/{self.num_timesteps}.zip"
                self.model.save(model_name)
                    #plt.savefig(f"{self.folder}/model_pool_plots/{self.num_timesteps}.png")
                    
                files = [f"{self.folder}/model_pool/{file}" for file in os.listdir(f"{self.folder}/model_pool") if file.endswith(".zip")]
                files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
                self.opponent = self._get_state_dict_from_file(files[-1])
                print(f"Chose latest opponent: {files[-1]}")
                self.opponent_name = files[-1].split("/")[-1].split(".")[0]
                #plt.axvline(x=self.num_timesteps, color='k', linestyle='--', linewidth=0.6)

            self.training_env.env_method("update_opponent", self.opponent)
        
        #Plot the rewards
        if self.num_timesteps % self.plot_every_n_steps < self.training_env.num_envs:
            #plt.plot(self.num_timesteps, self.wins / (self.wins + self.losses), 'r.', markersize=3)
            
            #plt.plot(self.num_timesteps, np.mean(self.last_n_results), 'b.', markersize=3)
            #plt.show(block=False)
            #plt.pause(0.000001)
            #
            with open(self.stats_file, "a") as f:
                f.write(f"{self.num_timesteps},{self.wins},{self.losses},{self.wins / (self.wins + self.losses)},{np.mean(self.last_n_results)},{self.opponent_name}\n")
            self.wins = 0
            self.losses = 0
            
        if self.num_timesteps % 1_000_000 < self.training_env.num_envs:
            self.model.save(f"{self.folder}/milestone_models/{self.num_timesteps}.zip")
            #plt.savefig(f"{self.folder}/milestone_plots/{self.num_timesteps}.png")

        return True

    def _on_rollout_start(self) -> None:
        self.rollouts += 1
        
    def _on_rollout_end(self) -> None:
        print(f"Rollout: {self.rollouts}\tSteps: {self.num_timesteps}\tTime trained: {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.init_time))}")
