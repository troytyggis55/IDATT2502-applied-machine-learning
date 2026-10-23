from datetime import datetime

import matplotlib as mpl

mpl.use('TkAgg')

import stable_baselines3

from volleyball_environment import VolleyballEnvironment
from general_callback import GeneralCallback

train_env = VolleyballEnvironment(render_mode=None)
obs, info = train_env.reset()

model = stable_baselines3.PPO("MlpPolicy", train_env, verbose=1, device="cpu")
callback = GeneralCallback
now = datetime.now()

folder = f"models/{now.strftime('%d%m_%H%M')}"

while True:
    model.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=True, callback=callback)
    model.save(f"{folder}/{model.num_timesteps}")
