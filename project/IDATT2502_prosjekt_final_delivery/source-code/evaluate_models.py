import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from volleyball_environment import VolleyballEnvironment


def evaluate_models(args):
    left_file, right_file, episode_amount = args
    set_random_seed(42)

    left_model = PPO.load(left_file, device="cpu")
    right_model = PPO.load(right_file, device="cpu")

    seed = 0
    env = VolleyballEnvironment()
    obs, info = env.reset(seed=seed)
    mirror_obs = env.get_obs_mirror()

    wins = 0
    losses = 0

    while wins + losses < episode_amount:
        left_action = int(left_model.predict(obs, deterministic=True)[0])
        right_action = int(right_model.predict(mirror_obs, deterministic=True)[0])

        obs, rewards, terminated, truncated, info = env.step(left_action, right_action)
        mirror_obs = env.get_obs_mirror()

        if terminated or truncated:
            if info.get("winner") == "left":
                wins += 1
            else:
                losses += 1
            seed += 1
            obs, info = env.reset(seed=seed)
            mirror_obs = env.get_obs_mirror()
            
    result_line = f"{os.path.basename(left_file).split('.')[0]},{os.path.basename(right_file).split('.')[0]},{wins}\n"
    print(result_line)
    return result_line


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a folder")
        sys.exit()
        
    if len(sys.argv) < 3:
        episode_amount = 1000
    else:
        episode_amount = int(sys.argv[2])

    models_folder = sys.argv[1]
        
    model_files = [os.path.join(models_folder, file) for file in os.listdir(models_folder) if
                   file.endswith(".zip")]
    model_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))

    result_file = os.path.join(models_folder, "evaluated_results.txt")
    with open(result_file, "w") as f:
        f.write("left_model,right_model,wins\n")

    # Generate all pairs
    model_pairs = [(left_file, right_file, episode_amount) for left_file in model_files for right_file in
                   model_files]
    
    print("Starting evaluation")
    print("Warning: This will utilize all available CPU cores to the maximum and make your computer slow when running")
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_models, pair): pair for pair in model_pairs}
        with open(result_file, "a") as f:
            for future in as_completed(futures):
                result_line = future.result()
                f.write(result_line)
