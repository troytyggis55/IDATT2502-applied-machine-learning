import os
import sys

from stable_baselines3.common.vec_env import DummyVecEnv

from volleyball_environment import VolleyballEnvironment
import stable_baselines3


evaluated_file = "old_models/old_models_1M/benchmark/final.zip"
benchmark_file = "old_models/old_models_1M/benchmark/final.zip"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluated_file = sys.argv[1]
    if len(sys.argv) > 2:
        benchmark_file = sys.argv[2]
        
evaluated_model = stable_baselines3.PPO.load(evaluated_file, device="cpu")
benchmark_model = stable_baselines3.PPO.load(benchmark_file, device="cpu")

total_episodes = 200
episodes = 1
wins_v_random = 0

env = VolleyballEnvironment(opponent=None)
vec_env = DummyVecEnv([lambda: env])
vec_env.seed(42)
obs = vec_env.reset()

while episodes <= total_episodes:
    action = int(evaluated_model.predict(obs, deterministic=True)[0])
    obs, rewards, done, info = vec_env.step([action])
    
    if done:
        if info[0]["winner"] == "left":
            wins_v_random += 1
        
        print(f"\rEpisode {episodes} done. Total wins: {wins_v_random}. Progress: {episodes / total_episodes * 100:.2f}%", end="")
        
        episodes += 1
        obs = vec_env.reset()

vec_env.close()

print(f"\nWinrate against random opponent: {wins_v_random / total_episodes}")

episodes = 1
wins_v_benchmark = 0

env = VolleyballEnvironment(opponent=benchmark_model.policy)
vec_env = DummyVecEnv([lambda: env])
vec_env.seed(42)
obs = vec_env.reset()

while episodes <= total_episodes:
    action = int(evaluated_model.predict(obs, deterministic=True)[0])
    obs, rewards, done, info = vec_env.step([action])
    
    if done:
        if info[0]["winner"] == "left":
            wins_v_benchmark += 1
        
        print(f"\rEpisode {episodes} done. Total wins: {wins_v_benchmark}. Progress: {episodes / total_episodes * 100:.2f}%", end="")
        
        episodes += 1
        obs = vec_env.reset()

print(f"\nModel winrate against benchmark opponent: {wins_v_benchmark / total_episodes}")

evaluated_name = evaluated_file.split("/")[-2].split(".")[0]

if not os.path.exists("evals"):
    os.makedirs("evals")

with open(f"deprecated/evals/{evaluated_name}.txt", "w") as f:
    f.write(f"{evaluated_model.get_parameters()['policy.optimizer']['param_groups']}\n")
    f.write(f"Total episodes per evaluation: {total_episodes}\n")
    f.write(f"Model winrate against random opponent: {wins_v_random / total_episodes}\n")
    f.write(f"Model winrate against benchmark opponent: {wins_v_benchmark / total_episodes}\n")