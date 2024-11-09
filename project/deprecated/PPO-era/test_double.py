import sys

import torch

from ProximalPolicyOptimization import ProximalPolicyOptimization
from environments.pong_environment import PongEnvironment

env = PongEnvironment()

date1 = "0711_1206"
date2 = "0711_1206"

gen1 = "4096"
gen2 = "4096"

folder1 = f"models/{env.name}/{date1}"
folder2 = f"models/{env.name}/{date2}"
info_file1 = f"{folder1}/info.txt"
info_file2 = f"{folder2}/info.txt"
model_file2 = f"{folder2}/{gen2}"
model_file1 = f"{folder1}/{gen1}"

def define_file_paths(file_path1, file_path2):
    global folder1, folder2, info_file1, info_file2, model_file1, model_file2
    model_file1 = file_path1
    model_file2 = file_path2
    folder1 = "/".join(model_file1.split("/")[:-1])
    folder2 = "/".join(model_file2.split("/")[:-1])
    info_file1 = f"{folder1}/info.txt"
    info_file2 = f"{folder2}/info.txt"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_path>")
    else:
        define_file_paths(sys.argv[1], sys.argv[2])

with open(info_file1, "r") as f:
    f.readline()
    input_size1, output_size1, hidden_layers1, hidden_size1, lr1, activation1 = f.readline().strip().split("\t")
    
with open(info_file2, "r") as f:
    f.readline()
    input_size2, output_size2, hidden_layers2, hidden_size2, lr2, activation2 = f.readline().strip().split("\t")
    
model1 = ProximalPolicyOptimization(input_size=int(input_size1), output_size=int(output_size1), hidden_layers=int(hidden_layers1), hidden_size=int(hidden_size1))
model1.load_state_dict(torch.load(model_file1))

model2 = ProximalPolicyOptimization(input_size=int(input_size2), output_size=int(output_size2), hidden_layers=int(hidden_layers2), hidden_size=int(hidden_size2))
model2.load_state_dict(torch.load(model_file2))

seed = 0
max_eps = 1000#np.iinfo(np.int32).max
action = {0: 0, 1: 0}

winrate = 0

for episode in range(0, max_eps):
    steps = 0
    rewards = 0
    
    observation, info = env.reset(seed=abs(hash((seed, episode))), options={"render_mode": ("human" if episode == 0 or episode == max_eps - 1 else None)})
    
    winner = None
    while winner is None:
        steps += 1
        
        observation1 = torch.tensor(observation[0]).float()
        observation2 = torch.tensor(observation[1]).float()
        
        action[0], _ = model1.act(observation1)
        action[1], _ = model2.act(observation2)
        
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward[0]
        
        if terminated[0] or truncated[0]:
            winner = info.get("winner")
            winrate += 1 if winner == 0 else 0
            
    print(f"Episode {episode} done in {steps} steps with reward {rewards} and winner {winner} (winrate: {winrate / (episode + 1)})")