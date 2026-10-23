import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from volleyball_environment import VolleyballEnvironment

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a folder")
        sys.exit()
        
    models_folder = f"{sys.argv[1]}/milestone_models"

    model_files = [f"{models_folder}/{file}" for file in os.listdir(f"{models_folder}") if file.endswith(".zip")]
    model_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    
    result_file = f"{sys.argv[1]}/evaluated_results.txt"
    with open(result_file, "w") as f:
        f.write("left_model,right_model,wins\n")

    # make each model play against the previous one and log the results after 1000 games
    # Switch the opponent side
    for left_file in model_files:
        for right_file in model_files:
            set_random_seed(42)
            
            left_model = PPO.load(left_file, device="cpu")
            right_model = PPO.load(right_file, device="cpu")
            
            seed = 0
            env = VolleyballEnvironment()
            obs, info = env.reset(seed=seed)
            mirror_obs = env.get_obs_mirror()
            
            wins = 0
            losses = 0
            
            while wins + losses < 100:
                left_action = int(left_model.predict(obs, deterministic=True)[0])
                right_action = int(right_model.predict(mirror_obs, deterministic=True)[0])
                
                obs, rewards, terminated, truncated, info = env.step(left_action, right_action)
                mirror_obs = env.get_obs_mirror()
                
                if terminated or truncated:
                    if info.get("winner") == "left":
                        wins += 1
                    else:
                        losses += 1
                    print(f"{left_file} vs {right_file}: {wins} - {losses}, seed: {seed}", end="\r")
                    
                    
                    seed += 1
                    obs, info = env.reset(seed=seed)
                    mirror_obs = env.get_obs_mirror()
                    
            print(f"\n{left_file} vs {right_file}: {wins} - {losses}\n")
            with open(result_file, "a") as f:
                f.write(f"{left_file.split('/')[-1].split('.')[0]},{right_file.split('/')[-1].split('.')[0]},{wins}\n")
