import sys

import torch

from ProximalPolicyOptimization import ProximalPolicyOptimization
from environments.single_player_pong_environment_v1 import SinglePlayerPongEnvironment

env = SinglePlayerPongEnvironment()

date = "0711_1206"
gen = "4096"

folder = f"models/{env.name}/{date}"
info_file = f"{folder}/info.txt"
model_file = f"{folder}/{gen}"

def define_file_paths(file_path):
    global folder, info_file, model_file
    model_file = file_path
    folder = "/".join(model_file.split("/")[:-1])
    info_file = f"{folder}/info.txt"
    
    print(folder, info_file, model_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        define_file_paths(sys.argv[1])

with open(info_file, "r") as f:
    f.readline()
    input_size, output_size, hidden_layers, hidden_size, lr, activation = f.readline().strip().split("\t")
    
model = ProximalPolicyOptimization(input_size=int(input_size), output_size=int(output_size), hidden_layers=int(hidden_layers), hidden_size=int(hidden_size))
model.load_state_dict(torch.load(model_file))

seed = 0
max_eps = 1000#np.iinfo(np.int32).max
action = {0: 0}

for episode in range(0, max_eps):
    steps = 0
    rewards = 0
    
    observation, info = env.reset(seed=abs(hash((seed, episode))), options={"render_mode": ("human" if episode == 0 or episode == max_eps - 1 else None)})
    done = None
    
    while done is None:
        steps += 1
        
        observation = torch.tensor(observation[0]).float()
        action[0], _ = model.act(observation)
        
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward[0]
        
        if terminated[0] or truncated[0]:
            done = "Terminated" if terminated[0] else "Truncated"
            
    print(f"Episode {episode} done in {steps} steps with reward {rewards} and winner {done}")