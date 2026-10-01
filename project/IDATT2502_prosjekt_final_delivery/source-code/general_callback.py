import os
import time
from copy import deepcopy
from datetime import datetime

import matplotlib as mpl
import numpy as np
from stable_baselines3 import PPO

mpl.use('TkAgg')
from stable_baselines3.common.callbacks import BaseCallback


def _get_state_dict_from_file(file):
    return PPO.load(file, device='cpu').policy.state_dict()


class GeneralCallback(BaseCallback):
    def __init__(
            self,
            verbose=1,
            total_trainingsteps=10_000_000,
            plot_every_n_steps=1000,
            steps_per_update=100_000,
            random_opponent_every_n_updates=10,
            winrate_window=200,
            winrate_threshold=0.75,
            milestone_every_n_steps=1_000_000,
            folder_name=None):
        np.random.seed(42)

        super(GeneralCallback, self).__init__(verbose)
        self.total_trainingsteps = total_trainingsteps

        self.rollouts = 0

        self.steps_per_update = steps_per_update
        self.random_opponent_every_n_steps = random_opponent_every_n_updates * steps_per_update
        self.plot_every_n_steps = plot_every_n_steps
        self.milestone_every_n_steps = milestone_every_n_steps

        self.winrate_window = winrate_window
        self.winrate_threshold = winrate_threshold

        self.folder_name = folder_name
        self.last_n_results = []
        self.wins = 0
        self.losses = 0

    def _on_training_start(self) -> None:
        if self.folder_name is None:
            self.folder_name = f"{datetime.now().strftime('%d%m_%H%M')}"

        # Create model_pool folder if it doesn't exist
        if not os.path.exists(f"{self.folder_name}"):
            os.makedirs(f"{self.folder_name}")

        self.dictionary_file = f"{self.folder_name}/dictionary.txt"

        # Init dictionary file
        with open(self.dictionary_file, "w") as f:
            for key, value in self.model.__dict__.items():
                f.write(f"{key}: {value}\n")

        self.stats_file = f"{self.folder_name}/stats.txt"
        with open(self.stats_file, "w") as f:
            f.write("num_timesteps,wins,losses,winrate,avg_winrate,opponent\n")

        self.init_time = time.time()

        model_name = f"{self.folder_name}/model_pool/0.zip"
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

        while len(self.last_n_results) > self.winrate_window:
            self.last_n_results.pop(0)

        # Update the opponent every steps_per_update steps
        if self.num_timesteps % self.steps_per_update < self.training_env.num_envs:
            winrate = np.mean(self.last_n_results)
            print(f"Winrate: {winrate}")

            if winrate > self.winrate_threshold:
                model_name = f"{self.folder_name}/model_pool/{self.num_timesteps}.zip"
                self.model.save(model_name)

            files = [f"{self.folder_name}/model_pool/{file}" for file in
                     os.listdir(f"{self.folder_name}/model_pool") if file.endswith(".zip")]
            files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
            if self.num_timesteps % self.random_opponent_every_n_steps < self.training_env.num_envs:  # Random opponent 20% of the time
                
                indices = np.arange(len(files))
                decay_rate = 0.1
                probabilities = np.exp(decay_rate * indices)
                probabilities = probabilities / probabilities.sum()
                
                chosen_index = np.random.choice(indices, p=probabilities) # Exponential sampling
                # chosen_index = np.random.randint(0, len(files)) # Uniform sampling
                
                self.opponent = _get_state_dict_from_file(files[chosen_index])
                print(f"Chose random opponent: {files[chosen_index]}")
                self.opponent_name = files[chosen_index].split("/")[-1].split(".")[0]
            else:  # Choose the latest opponent 80% of the time

                self.opponent = _get_state_dict_from_file(files[-1])
                print(f"Chose latest opponent: {files[-1]}")
                self.opponent_name = files[-1].split("/")[-1].split(".")[0]

            self.training_env.env_method("update_opponent", self.opponent)

        if self.num_timesteps % self.plot_every_n_steps < self.training_env.num_envs:
            with open(self.stats_file, "a") as f:
                f.write(
                    f"{self.num_timesteps},{self.wins},{self.losses},{self.wins / (self.wins + self.losses)},{np.mean(self.last_n_results)},{self.opponent_name}\n")
            self.wins = 0
            self.losses = 0

        if self.num_timesteps % self.milestone_every_n_steps < self.training_env.num_envs:
            self.model.save(f"{self.folder_name}/milestone_models/{self.num_timesteps}.zip")

        return True

    def _on_rollout_start(self) -> None:
        self.rollouts += 1

    def _on_rollout_end(self) -> None:
        time_left = (time.time() - self.init_time) / self.num_timesteps * (self.total_trainingsteps - self.num_timesteps)
        
        print(
            f"Rollout: {self.rollouts}\tSteps: {self.num_timesteps}\tTime trained: {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.init_time))}\tTime left: {time.strftime('%H:%M:%S', time.gmtime(time_left))}")
