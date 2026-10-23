import os

import matplotlib as mpl
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

from general_callback import GeneralCallback

mpl.use('TkAgg')

import stable_baselines3

from volleyball_environment import VolleyballEnvironment

from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env(opponent=None):
    def _init():
        env = VolleyballEnvironment(render_mode=None, opponent=opponent)
        return env

    return _init


def train_model(
        seed=42,
        steps_per_update=40_000,
        random_opponent_every_n_updates=5,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        total_trainingsteps=1_000_000,
        num_envs=4,
        name=None):
    
    if name is None:
        print("Please provide a name for the model")
        return
    
    set_random_seed(seed)


    actual_n_steps = n_steps // num_envs # Only used for setting the model
    
    callback = GeneralCallback(
        total_trainingsteps=total_trainingsteps,
        plot_every_n_steps=1000,
        steps_per_update=steps_per_update,
        random_opponent_every_n_updates=random_opponent_every_n_updates,
        modelname=name)
    
    env_fns = [make_env() for _ in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = stable_baselines3.PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        device="cpu",
        tensorboard_log=f"{name}/tensorboard",
        learning_rate=learning_rate,
        n_steps=actual_n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde
    )
    
    model.learn(total_timesteps=total_trainingsteps, reset_num_timesteps=False, callback=callback)
    
if __name__ == "__main__":
    train_model(name="generalized_models", total_trainingsteps=np.iinfo(np.int32).max, seed=0, num_envs=8)